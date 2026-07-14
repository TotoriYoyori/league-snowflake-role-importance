import os

import pandas as pd
import streamlit as st

from settings import Settings
from src import query
from src.model import (
    eda,
    evaluation,
    importance,
    prep,
    predict_win_probability,
    predictor_slider_bounds
)


# --------------- CONSTANTS ---------------
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "sample_data")

# SiS session token file only exists inside Snowflake -> use Snowflake live data.
IS_LOCAL: bool = not os.path.isfile("/snowflake/session/token")
CACHE_TTL = 600

# --------------- MOCK FETCH (LOCAL MODE ONLY) ---------------
DIFF_INTERVAL_CSV = "s1000_diff_interval_state.csv"
MATCH_SUMMARY_CSV = "s1000_match_team_stats_summary.csv"

def _mock_diff_interval_by_match(
    minute: int,
    team: str,
    min_game_duration: int,
) -> pd.DataFrame:
    diff = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, DIFF_INTERVAL_CSV))
    match = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, MATCH_SUMMARY_CSV))

    d = diff[(diff["MINUTE"] == minute) & (diff["TEAM"] == team)]
    m = match[match["GAME_DURATION"] >= min_game_duration]

    return (d
        .merge(m, on="MATCH_ID", how="inner")
        [["MATCH_ID", "LANE", "GOLD_DIFF", "WINNING_TEAM", "AVERAGE_RANK", "GAME_DATE", "GAME_DURATION"]]
        .reset_index(drop=True)
    )


def _failed(message: str) -> pd.DataFrame:
    df = pd.DataFrame()
    df.attrs["error"] = message
    return df


def load_error(df: pd.DataFrame):
    return df.attrs.get("error")


@st.cache_resource
def get_session():
    if IS_LOCAL:
        return None
    conn = st.connection("snowflake", ttl=None)
    return conn.session()


def _run(session, sql: str) -> pd.DataFrame:
    return session.sql(sql).to_pandas()


@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading gold-diff data…")
def load_diff_interval_by_match(minute, team, min_game_duration):
    try:
        if IS_LOCAL:
            return _mock_diff_interval_by_match(minute=minute, team=team, min_game_duration=min_game_duration)
        diff_query = query.DiffIntervalByMatch(minute=minute, team=team, min_game_duration=min_game_duration)
        return _run(get_session(), diff_query.build())
    except Exception as e:
        return _failed(str(e))


@st.cache_data(ttl=CACHE_TTL)
def get_pivoted_data(settings: Settings, minute, team, min_game_duration):
    raw_df = load_diff_interval_by_match(minute, team, min_game_duration)
    error = load_error(raw_df)
    if error is not None:
        return _failed(error), 0
    pivoted_df, n_dropped = prep.pivot_diff_interval(raw_df, win_reference_team=team)
    scaled_df = prep.scale_gold_diff(pivoted_df, list(settings.feature_cols), scale=settings.gold_scale)
    return scaled_df, n_dropped


@st.cache_data(ttl=CACHE_TTL)
def get_eda_bundle(settings: Settings, minute, team, min_game_duration):
    scaled_df, _ = get_pivoted_data(settings, minute, team, min_game_duration)
    error = load_error(scaled_df)
    if error is not None:
        return {"error": error}
    target_col = f"{team.upper()}_WIN"
    feature_cols = list(settings.feature_cols)
    return {
        "distributions": eda.feature_distributions(scaled_df, feature_cols),
        "correlation": eda.feature_correlation(scaled_df, feature_cols),
        "balance": eda.class_balance(scaled_df, target_col),
    }


@st.cache_data(ttl=CACHE_TTL)
def get_eval_bundle(settings: Settings, minute, team, min_game_duration):
    scaled_df, _ = get_pivoted_data(settings, minute, team, min_game_duration)
    error = load_error(scaled_df)
    if error is not None:
        return {"error": error}
    target_col = f"{team.upper()}_WIN"
    feature_cols = list(settings.feature_cols)
    train_df, test_df = prep.split_train_test(scaled_df, target_col, random_state=settings.random_state)
    model = evaluation.fit_logistic_regression(train_df, feature_cols, target_col)
    metrics = evaluation.evaluate_on_test(model, test_df, feature_cols, target_col, threshold=settings.win_loss_threshold)
    return {
        "train_df": train_df,
        "test_df": test_df,
        "coef_df": evaluation.report_model_coefficients(model, feature_cols),
        "sm_table": evaluation.statsmodels_coefficient_table(train_df, feature_cols, target_col),
        "metrics": metrics,
        "roc_df": evaluation.roc_curve_data(test_df[target_col], metrics["y_proba"]),
        "preview_df": evaluation.preview_predictions(test_df, metrics["y_proba"], target_col, threshold=settings.win_loss_threshold),
    }


@st.cache_resource(ttl=CACHE_TTL)
def get_full_model(settings: Settings, minute, team, min_game_duration):
    """cache_resource, not cache_data: returns a live sklearn model object,
    reused across reruns — the Predictor tab scores against this directly.
    ttl matches get_pivoted_data's — without it this never expired, so the
    Predictor tab's model would silently freeze on stale data forever while
    every other tab kept refreshing on the same 10-minute cadence.

    Do not mutate the returned model in place (e.g. calling .fit() again on
    it) — cache_resource hands back the same object reference on every hit,
    unlike cache_data, so an in-place mutation would corrupt this cache slot
    for every other caller sharing the same key."""
    scaled_df, _ = get_pivoted_data(settings, minute, team, min_game_duration)
    if load_error(scaled_df) is not None:
        return None
    target_col = f"{team.upper()}_WIN"
    feature_cols = list(settings.feature_cols)
    return importance.fit_full_data_model(scaled_df, feature_cols, target_col)


@st.cache_data(ttl=CACHE_TTL)
def get_full_model_coefficients(settings: Settings, minute, team, min_game_duration):
    model = get_full_model(settings, minute, team, min_game_duration)
    if model is None:
        return _failed("Upstream data failed to load.")
    feature_cols = list(settings.feature_cols)
    return evaluation.report_model_coefficients(model, feature_cols)


@st.cache_data(ttl=CACHE_TTL)
def get_stability_summary(settings: Settings, minute, team, n_splits, n_repeats, min_game_duration):
    scaled_df, _ = get_pivoted_data(settings, minute, team, min_game_duration)
    error = load_error(scaled_df)
    if error is not None:
        return _failed(error)
    target_col = f"{team.upper()}_WIN"
    feature_cols = list(settings.feature_cols)
    return importance.coefficient_stability_cv(scaled_df, feature_cols, target_col, n_splits=n_splits, n_repeats=n_repeats, random_state=settings.random_state)


@st.cache_data(ttl=CACHE_TTL)
def get_lane_importance(settings: Settings, minute, team, n_splits, n_repeats, min_game_duration):
    stability_summary = get_stability_summary(settings, minute, team, n_splits, n_repeats, min_game_duration)
    error = load_error(stability_summary)
    if error is not None:
        return _failed(error)
    return importance.lane_importance(stability_summary)


@st.cache_data(ttl=CACHE_TTL)
def get_predictor_bounds(settings: Settings, minute, team, min_game_duration):
    scaled_df, _ = get_pivoted_data(settings, minute, team, min_game_duration)
    if load_error(scaled_df) is not None:
        return {}
    feature_cols = list(settings.feature_cols)
    return predictor_slider_bounds(scaled_df, feature_cols)

import os

import pandas as pd
import streamlit as st

from settings import Settings
from src import query as q
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
    """Mirrors query.py::DiffIntervalByMatch.build() + its execution, but reads
    from the two vendored CSVs and filters/joins in pandas instead of SQL."""
    diff = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, DIFF_INTERVAL_CSV))
    match = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, MATCH_SUMMARY_CSV))

    d = diff[(diff["MINUTE"] == minute) & (diff["TEAM"] == team)]
    m = match[match["GAME_DURATION"] >= min_game_duration]

    return (d
        .merge(m, on="MATCH_ID", how="inner")
        [["MATCH_ID", "LANE", "GOLD_DIFF", "WINNING_TEAM", "AVERAGE_RANK", "GAME_DATE", "GAME_DURATION"]]
        .reset_index(drop=True)
    )


# --------------- FAILURE HANDLING ---------------
def _failed(message: str) -> pd.DataFrame:
    df = pd.DataFrame()
    df.attrs["error"] = message

    return df


def load_error(df: pd.DataFrame) -> str | None:
    """The error message if `df` came from a failed load, else None."""
    return df.attrs.get("error")


# --------------- SNOWFLAKE SESSION ---------------
@st.cache_resource
def get_session():
    """Return the Snowpark session in SiS/live mode, or None when local."""
    if IS_LOCAL:
        return None

    conn = st.connection("snowflake", ttl=None)  # None = forever
    return conn.session()


def _run(session, sql: str) -> pd.DataFrame:
    return session.sql(sql).to_pandas()


# --------------- RAW FETCH (LIVE OR MOCK) ---------------
def load_diff_interval_by_match(
    minute: int,
    team: str,
    min_game_duration: int,
) -> pd.DataFrame:
    """Long-format rows: one per (match, lane), joined to match outcome."""

    @st.cache_data(
        ttl=CACHE_TTL,
        show_spinner=f"Loading gold-diff data at minute {minute} ({team})…",
    )
    def _load(
        minute: int,
        team: str,
        min_game_duration: int
    ) -> pd.DataFrame:
        try:
            if IS_LOCAL:
                return _mock_diff_interval_by_match(
                    minute=minute,
                    team=team,
                    min_game_duration=min_game_duration,
                )
            query = q.DiffIntervalByMatch(
                minute=minute,
                team=team,
                min_game_duration=min_game_duration,
            )
            return _run(get_session(), query.build())
        except Exception as e:
            return _failed(str(e))

    return _load(minute, team, min_game_duration)


# --------------- STAGE 1: PIVOT + SCALE ---------------
def get_pivoted_data(
    settings: Settings,
    minute: int,
    team: str,
    min_game_duration: int,
) -> tuple[pd.DataFrame, int]:
    """Returns (scaled_df, n_dropped). The foundation every stage below
    re-derives from, keyed on (minute, team, min_game_duration)."""

    @st.cache_data(ttl=CACHE_TTL)
    def _pivot(minute: int, team: str, min_game_duration: int) -> tuple[pd.DataFrame, int]:
        raw_df = load_diff_interval_by_match(minute, team, min_game_duration)
        error = load_error(raw_df)
        if error is not None:
            return _failed(error), 0

        pivoted_df, n_dropped = prep.pivot_diff_interval(raw_df, win_reference_team=team)
        scaled_df = prep.scale_gold_diff(
            pivoted_df, list(settings.feature_cols), scale=settings.gold_scale
        )
        return scaled_df, n_dropped

    return _pivot(minute, team, min_game_duration)


# --------------- STAGE 2: EDA ---------------
def get_eda_bundle(
    settings: Settings,
    minute: int,
    team: str,
    min_game_duration: int,
) -> dict:
    @st.cache_data(ttl=CACHE_TTL)
    def _eda(minute: int, team: str, min_game_duration: int) -> dict:
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

    return _eda(minute, team, min_game_duration)


# --------------- STAGE 3: EVAL (TRAIN/TEST SPLIT) ---------------
def get_eval_bundle(
    settings: Settings,
    minute: int,
    team: str,
    min_game_duration: int,
) -> dict:
    """Keyed on (minute, team, min_game_duration) — test_size/random_state are
    fixed constants in Settings (not user-adjustable per current scope)."""

    @st.cache_data(ttl=CACHE_TTL)
    def _eval(minute: int, team: str, min_game_duration: int) -> dict:
        scaled_df, _ = get_pivoted_data(settings, minute, team, min_game_duration)
        error = load_error(scaled_df)
        if error is not None:
            return {"error": error}

        target_col = f"{team.upper()}_WIN"
        feature_cols = list(settings.feature_cols)

        train_df, test_df = prep.split_train_test(
            scaled_df, target_col,
            test_size=settings.test_size,
            random_state=settings.random_state,
        )
        model = evaluation.fit_logistic_regression(train_df, feature_cols, target_col)
        metrics = evaluation.evaluate_on_test(model, test_df, feature_cols, target_col)
        return {
            "train_df": train_df,
            "test_df": test_df,
            "coef_df": evaluation.report_model_coefficients(model, feature_cols),
            "sm_table": evaluation.statsmodels_coefficient_table(train_df, feature_cols, target_col),
            "metrics": metrics,
            "roc_df": evaluation.roc_curve_data(test_df[target_col], metrics["y_proba"]),
            "preview_df": evaluation.preview_predictions(test_df, metrics["y_proba"], target_col),
        }

    return _eval(minute, team, min_game_duration)


# --------------- STAGE 4: FULL-DATA MODEL (MODEL OF RECORD) ---------------
def get_full_model(
    settings: Settings,
    minute: int,
    team: str,
    min_game_duration: int,
):
    """cache_resource, not cache_data: returns a live sklearn model object,
    reused across reruns — the Predictor tab scores against this directly."""

    @st.cache_resource
    def _full_model(minute: int, team: str, min_game_duration: int):
        scaled_df, _ = get_pivoted_data(settings, minute, team, min_game_duration)
        if load_error(scaled_df) is not None:
            return None  # sentinel: upstream data failed to load

        target_col = f"{team.upper()}_WIN"
        feature_cols = list(settings.feature_cols)
        return importance.fit_full_data_model(scaled_df, feature_cols, target_col)

    return _full_model(minute, team, min_game_duration)


def get_full_model_coefficients(
    settings: Settings,
    minute: int,
    team: str,
    min_game_duration: int,
) -> pd.DataFrame:
    @st.cache_data(ttl=CACHE_TTL)
    def _coef(minute: int, team: str, min_game_duration: int) -> pd.DataFrame:
        model = get_full_model(settings, minute, team, min_game_duration)
        if model is None:
            return _failed("Upstream data failed to load.")

        feature_cols = list(settings.feature_cols)
        return evaluation.report_model_coefficients(model, feature_cols)

    return _coef(minute, team, min_game_duration)


# --------------- STAGE 5: CV COEFFICIENT STABILITY ---------------
def get_stability_summary(
    settings: Settings,
    minute: int,
    team: str,
    n_splits: int,
    n_repeats: int,
    min_game_duration: int,
) -> pd.DataFrame:
    """Keyed on (minute, team, n_splits, n_repeats, min_game_duration) — every
    sidebar control that can change the underlying data or the CV fit."""

    @st.cache_data(ttl=CACHE_TTL)
    def _stability(minute: int, team: str, n_splits: int, n_repeats: int, min_game_duration: int) -> pd.DataFrame:
        scaled_df, _ = get_pivoted_data(settings, minute, team, min_game_duration)
        error = load_error(scaled_df)
        if error is not None:
            return _failed(error)

        target_col = f"{team.upper()}_WIN"
        feature_cols = list(settings.feature_cols)
        return importance.coefficient_stability_cv(
            scaled_df, feature_cols, target_col,
            n_splits=n_splits, n_repeats=n_repeats,
            random_state=settings.random_state,
        )

    return _stability(minute, team, n_splits, n_repeats, min_game_duration)


def get_lane_importance(
    settings: Settings,
    minute: int,
    team: str,
    n_splits: int,
    n_repeats: int,
    min_game_duration: int,
) -> pd.DataFrame:
    @st.cache_data(ttl=CACHE_TTL)
    def _importance(minute: int, team: str, n_splits: int, n_repeats: int, min_game_duration: int) -> pd.DataFrame:
        stability_summary = get_stability_summary(settings, minute, team, n_splits, n_repeats, min_game_duration)
        error = load_error(stability_summary)
        if error is not None:
            return _failed(error)

        return importance.lane_importance(stability_summary)

    return _importance(minute, team, n_splits, n_repeats, min_game_duration)


# --------------- STAGE 6: PREDICTOR SLIDER BOUNDS ---------------
def get_predictor_bounds(
    settings: Settings,
    minute: int,
    team: str,
    min_game_duration: int,
) -> dict[str, tuple[float, float]]:
    @st.cache_data(ttl=CACHE_TTL)
    def _bounds(minute: int, team: str, min_game_duration: int) -> dict[str, tuple[float, float]]:
        scaled_df, _ = get_pivoted_data(settings, minute, team, min_game_duration)
        if load_error(scaled_df) is not None:
            return {}  # render_predictor_tab falls back to fixed default bounds per feature

        feature_cols = list(settings.feature_cols)
        return predictor_slider_bounds(scaled_df, feature_cols)

    return _bounds(minute, team, min_game_duration)

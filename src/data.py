import pandas as pd
import streamlit as st

from settings import Settings
from src import mock, query as q
from src.model import (
    eda,
    evaluation,
    importance,
    prep,
    predict_win_probability,
    predictor_slider_bounds
)


# --------------- SNOWFLAKE SESSION ---------------
@st.cache_resource
def get_session(_settings: Settings):
    """Return the Snowpark session in SiS/live mode, or None when local."""
    if _settings.is_local:
        return None
    conn = st.connection("snowflake", ttl=_settings.snowflake.connection_ttl)
    return conn.session()


def _run(session, sql: str) -> pd.DataFrame:
    return session.sql(sql).to_pandas()


# --------------- RAW FETCH (LIVE OR MOCK) ---------------
def load_diff_interval_by_match(
    settings: Settings,
    minute: int,
    team: str,
) -> pd.DataFrame:
    """Long-format rows: one per (match, lane), joined to match outcome."""

    @st.cache_data(
        ttl=settings.cache.source_data_ttl,
        show_spinner=f"Loading gold-diff data at minute {minute} ({team})…",
    )
    def _load(minute: int, team: str) -> pd.DataFrame:
        if settings.is_local:
            return mock.diff_interval_by_match(
                sample_data_dir=settings.sample_data_dir,
                minute=minute,
                team=team,
                min_game_duration=settings.model.min_game_duration,
            )
        query = q.DiffIntervalByMatch(
            database=settings.snowflake.database,
            schema_=settings.snowflake.schema_,
            minute=minute,
            team=team,
            min_game_duration=settings.model.min_game_duration,
        )
        return _run(get_session(settings), query.build())

    return _load(minute, team)


# --------------- STAGE 1: PIVOT + SCALE ---------------
def get_pivoted_data(
    settings: Settings,
    minute: int,
    team: str,
) -> tuple[pd.DataFrame, int]:
    """Returns (scaled_df, n_dropped). The foundation every stage below
    re-derives from, keyed only on (minute, team)."""

    @st.cache_data(ttl=settings.cache.source_data_ttl)
    def _pivot(minute: int, team: str) -> tuple[pd.DataFrame, int]:
        raw_df = load_diff_interval_by_match(settings, minute, team)
        pivoted_df, n_dropped = prep.pivot_diff_interval(raw_df, win_reference_team=team)
        scaled_df = prep.scale_gold_diff(
            pivoted_df, list(settings.model.feature_cols), scale=settings.model.gold_scale
        )
        return scaled_df, n_dropped

    return _pivot(minute, team)


# --------------- STAGE 2: EDA ---------------
def get_eda_bundle(
    settings: Settings,
    minute: int,
    team: str,
) -> dict:
    @st.cache_data(ttl=settings.cache.source_data_ttl)
    def _eda(minute: int, team: str) -> dict:
        scaled_df, _ = get_pivoted_data(settings, minute, team)
        target_col = f"{team.upper()}_WIN"
        feature_cols = list(settings.model.feature_cols)
        return {
            "distributions": eda.feature_distributions(scaled_df, feature_cols),
            "correlation": eda.feature_correlation(scaled_df, feature_cols),
            "balance": eda.class_balance(scaled_df, target_col),
        }

    return _eda(minute, team)


# --------------- STAGE 3: EVAL (TRAIN/TEST SPLIT) ---------------
def get_eval_bundle(
    settings: Settings,
    minute: int,
    team: str,
) -> dict:
    """Keyed on (minute, team) only — test_size/random_state are fixed
    constants in ModelSettings (not user-adjustable per current scope)."""

    @st.cache_data(ttl=settings.cache.baseline_fit_ttl)
    def _eval(minute: int, team: str) -> dict:
        scaled_df, _ = get_pivoted_data(settings, minute, team)
        target_col = f"{team.upper()}_WIN"
        feature_cols = list(settings.model.feature_cols)

        train_df, test_df = prep.split_train_test(
            scaled_df, target_col,
            test_size=settings.model.test_size,
            random_state=settings.model.random_state,
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

    return _eval(minute, team)


# --------------- STAGE 4: FULL-DATA MODEL (MODEL OF RECORD) ---------------
def get_full_model(
    settings: Settings,
    minute: int,
    team: str,
):
    """cache_resource, not cache_data: returns a live sklearn model object,
    reused across reruns — the Predictor tab scores against this directly."""

    @st.cache_resource
    def _full_model(minute: int, team: str):
        scaled_df, _ = get_pivoted_data(settings, minute, team)
        target_col = f"{team.upper()}_WIN"
        feature_cols = list(settings.model.feature_cols)
        return importance.fit_full_data_model(scaled_df, feature_cols, target_col)

    return _full_model(minute, team)


def get_full_model_coefficients(
    settings: Settings,
    minute: int,
    team: str,
) -> pd.DataFrame:
    @st.cache_data(ttl=settings.cache.full_fit_ttl)
    def _coef(minute: int, team: str) -> pd.DataFrame:
        model = get_full_model(settings, minute, team)
        feature_cols = list(settings.model.feature_cols)
        return evaluation.report_model_coefficients(model, feature_cols)

    return _coef(minute, team)


# --------------- STAGE 5: CV COEFFICIENT STABILITY ---------------
def get_stability_summary(
    settings: Settings,
    minute: int,
    team: str,
    n_splits: int,
    n_repeats: int,
) -> pd.DataFrame:
    """Keyed on (minute, team, n_splits, n_repeats) — the only stage whose
    cache key includes the CV sidebar controls."""

    @st.cache_data(ttl=settings.cache.cv_stability_ttl)
    def _stability(minute: int, team: str, n_splits: int, n_repeats: int) -> pd.DataFrame:
        scaled_df, _ = get_pivoted_data(settings, minute, team)
        target_col = f"{team.upper()}_WIN"
        feature_cols = list(settings.model.feature_cols)
        return importance.coefficient_stability_cv(
            scaled_df, feature_cols, target_col,
            n_splits=n_splits, n_repeats=n_repeats,
            random_state=settings.model.random_state,
        )

    return _stability(minute, team, n_splits, n_repeats)


def get_lane_importance(
    settings: Settings,
    minute: int,
    team: str,
    n_splits: int,
    n_repeats: int,
) -> pd.DataFrame:
    @st.cache_data(ttl=settings.cache.cv_stability_ttl)
    def _importance(minute: int, team: str, n_splits: int, n_repeats: int) -> pd.DataFrame:
        stability_summary = get_stability_summary(settings, minute, team, n_splits, n_repeats)
        return importance.lane_importance(stability_summary)

    return _importance(minute, team, n_splits, n_repeats)


# --------------- STAGE 6: PREDICTOR SLIDER BOUNDS ---------------
def get_predictor_bounds(
    settings: Settings,
    minute: int,
    team: str,
) -> dict[str, tuple[float, float]]:
    @st.cache_data(ttl=settings.cache.source_data_ttl)
    def _bounds(minute: int, team: str) -> dict[str, tuple[float, float]]:
        scaled_df, _ = get_pivoted_data(settings, minute, team)
        feature_cols = list(settings.model.feature_cols)
        return predictor_slider_bounds(scaled_df, feature_cols)

    return _bounds(minute, team)

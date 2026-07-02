import streamlit as st

from settings import get_settings
from src import theme, ui

st.set_page_config(
    page_title="LEAGUE_SNOWFLAKE Role Importance",
    layout="wide",
)

settings = get_settings()
theme.inject(st)
ui.render_header(settings)

with st.sidebar:
    st.markdown("### Configuration · 配置")
    st.caption(f"Environment: **{settings.env}**")

    minute = st.select_slider(
        "Minute",
        options=list(settings.model.minute_options),
        value=settings.model.default_minute,
    )
    team = st.radio(
        "Team",
        options=list(settings.model.team_options),
        index=list(settings.model.team_options).index(settings.model.default_team),
        horizontal=True,
    )

    st.divider()
    st.markdown("#### Cross-Validation · 交叉验证")
    n_splits = st.number_input(
        "n_splits",
        min_value=settings.model.cv_n_splits_range[0],
        max_value=settings.model.cv_n_splits_range[1],
        value=settings.model.cv_n_splits_default,
    )
    n_repeats = st.number_input(
        "n_repeats",
        min_value=settings.model.cv_n_repeats_range[0],
        max_value=settings.model.cv_n_repeats_range[1],
        value=settings.model.cv_n_repeats_default,
    )

    st.divider()
    st.caption(
        "Gold diffs are scaled to per-1,000g units throughout. "
        "Lane Importance and Predictor share one model fit on all "
        "available data; Model Evaluation uses a held-out test split."
    )

tab_eda, tab_eval, tab_importance, tab_predictor = st.tabs(
    [f"{en} · {zh}" for en, zh in settings.ui.tab_labels]
)

with tab_eda:
    ui.render_eda_tab(settings, minute, team)

with tab_eval:
    ui.render_evaluation_tab(settings, minute, team)

with tab_importance:
    ui.render_importance_tab(settings, minute, team, n_splits, n_repeats)

with tab_predictor:
    ui.render_predictor_tab(settings, minute, team)

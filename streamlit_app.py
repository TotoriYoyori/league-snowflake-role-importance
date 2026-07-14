import streamlit as st

from settings import settings
from src import data, theme, ui


# --------------- CONSTANTS ---------------
SECONDS_PER_MINUTE = 60


# --------------- APP ---------------
st.set_page_config(
    page_title="LEAGUE_SNOWFLAKE Role Importance",
    layout="wide",
)

with st.sidebar:
    st.markdown("### Configuration · 配置")
    st.caption(f"Environment · 环境: **{'local' if data.IS_LOCAL else 'production'}**")

    minute = st.select_slider(
        "Minute · 分钟",
        options=list(settings.minute_options),
        value=settings.default_minute,
    )
    team = st.radio(
        "Team · 方",
        options=list(settings.team_options),
        format_func=lambda t: f"{t} · {'蓝' if t == 'Blue' else '红'}",
        index=list(settings.team_options).index(settings.default_team),
        horizontal=True,
    )
    min_game_duration_minutes = st.select_slider(
        "Min match length (min) · 最短比赛时长（分钟）",
        options=list(settings.min_game_duration_options),
        value=settings.default_min_game_duration_minutes,
        help="Excludes matches shorter than this — e.g. remakes or very early forfeits · "
             "排除短于此时长的对局，例如重开局或早期投降",
    )
    min_game_duration = min_game_duration_minutes * SECONDS_PER_MINUTE  # DB column is in seconds

    st.divider()
    st.markdown("#### Cross-Validation · 交叉验证")
    n_splits = st.number_input(
        "n_splits · 分裂次数",
        min_value=settings.cv_n_splits_range[0],
        max_value=settings.cv_n_splits_range[1],
        value=settings.cv_n_splits_default,
    )
    n_repeats = st.number_input(
        "n_repeats · 重复次数",
        min_value=settings.cv_n_repeats_range[0],
        max_value=settings.cv_n_repeats_range[1],
        value=settings.cv_n_repeats_default,
    )

    st.divider()
    st.caption(
        "Gold diffs are scaled to per-1,000g units throughout. "
        "Lane Importance and Predictor share one model fit on all "
        "available data; Model Evaluation uses a held-out test split.\n\n"
        "金币差均以每 1000 金为单位缩放。分路重要性与预测器共用同一个基于全部数据拟合的模型；"
        "模型评估则使用留出测试集。"
    )

theme.inject(st)
ui.render_header()

tab_eda, tab_eval, tab_importance, tab_predictor = st.tabs(
    [f"{en} · {zh}" for en, zh in ui.TAB_LABELS]
)

with tab_eda:
    ui.render_eda_tab(settings, minute, team, min_game_duration)

with tab_eval:
    ui.render_evaluation_tab(settings, minute, team, min_game_duration)

with tab_importance:
    ui.render_importance_tab(settings, minute, team, n_splits, n_repeats, min_game_duration)

with tab_predictor:
    ui.render_predictor_tab(settings, minute, team, min_game_duration)

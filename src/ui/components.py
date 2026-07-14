import textwrap
from typing import Callable

import pandas as pd
import streamlit as st

from src import data

# --------------- APP LABELING ---------------
APP_TITLE = "Role Importance"
APP_SUBTITLE_EN = "League of Legends Lane Quest Impact — Win Coefficient Tracker"
APP_SUBTITLE_ZH = "英雄联盟分路任务 — 其影响以及胜率系数追踪"
TAB_LABELS = (
    ("EDA", "分析一览"),
    ("Model Evaluation", "模型评估"),
    ("Lane Importance", "分路重要性"),
    ("Predictor", "预测器"),
)

# --------------- STATUS PILLS (CSS classes defined in theme.py) ---------------
PILL_OK = "ri-pill-ok"
PILL_WARN = "ri-pill-warn"
PILL_FAIL = "ri-pill-fail"
PILL_NEUTRAL = "ri-pill-neutral"


def _pill(text: str, level: str = PILL_NEUTRAL) -> str:
    return f'<span class="ri-pill {level}">{text}</span>'


# --------------- HEADER ---------------
def render_header() -> None:
    mode_label = "Local / Mock" if data.IS_LOCAL else "Snowflake (live)"
    mode_level = PILL_NEUTRAL if data.IS_LOCAL else PILL_OK
    st.markdown(
        textwrap.dedent(f"""
        <div class="ri-header">
            <div class="ri-title">{APP_TITLE} {_pill(mode_label, mode_level)}</div>
            <div class="ri-subtitle-en">{APP_SUBTITLE_EN}</div>
            <div class="ri-subtitle-zh">{APP_SUBTITLE_ZH}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_section_header(en: str, zh: str) -> None:
    st.markdown(
        textwrap.dedent(f"""
        <div class="ri-section-header">
            <span class="ri-section-en">{en}</span>
            <span class="ri-section-zh">{zh}</span>
        </div>
        """),
        unsafe_allow_html=True,
    )


# --------------- CARD (generalized: body is a render callback, not just a df) ---------------
def render_card(
    name_en: str,
    name_zh: str,
    desc_en: str,
    body: Callable[[], object],
    status_text: str | None = None,
    status_level: str = PILL_NEUTRAL,
) -> None:
    pill_html = _pill(status_text, status_level) if status_text else ""
    with st.container(border=True):
        st.markdown(
            textwrap.dedent(f"""
            <div class="ri-card-title">
                <div>
                    <div class="ri-card-name">{name_en} · <span style="font-weight:400;color:var(--ink-soft);">{name_zh}</span></div>
                    <div class="ri-card-desc">{desc_en}</div>
                </div>
                {pill_html}
            </div>
            """),
            unsafe_allow_html=True,
        )
        body()


def _df_body(
    df: pd.DataFrame,
    hide_index: bool = True
) -> Callable[[], object]:
    return lambda: st.dataframe(df, hide_index=hide_index)


# --------------- CONTEXT CAPTION (AVERAGE_RANK / GAME_DATE — informational only) ---------------
def render_context_caption(df: pd.DataFrame) -> None:
    if df.empty:
        return
    date_min, date_max = df["GAME_DATE"].min(), df["GAME_DATE"].max()
    date_min_str = pd.to_datetime(date_min).strftime("%Y-%m-%d")
    date_max_str = pd.to_datetime(date_max).strftime("%Y-%m-%d")

    rank_counts = df["AVERAGE_RANK"].value_counts()
    top_rank = rank_counts.index[0] if not rank_counts.empty else "n/a"
    st.caption(
        f"Data spans {date_min_str} to {date_max_str} · 数据时间范围 {date_min_str} 至 {date_max_str}"
    )
    st.caption(
        f"{len(df)} matches · most common rank tier: {top_rank} · 共 {len(df)} 场比赛，最常见段位：{top_rank}"
    )

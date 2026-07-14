import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from settings import Settings
from src import data
from src.ui.components import PILL_FAIL, PILL_OK, PILL_WARN, _df_body, render_card, render_context_caption
from src.ui.theme import DEFAULT_MARGIN, PALETTE, hex_to_rgba

# --------------- COLORING THRESHOLDS ---------------
CORR_NEAR_PERFECT = 0.99
CORR_HIGH = 0.7
CORR_MODERATE = 0.5
BALANCE_LOWER = 0.4
BALANCE_UPPER = 0.6


def _highlight_correlation(val: float) -> str:
    """Background color for a correlation cell."""
    if pd.isna(val):
        return ""

    abs_val = abs(val)
    if abs_val >= CORR_NEAR_PERFECT:
        return f"background-color: {hex_to_rgba(PALETTE['ink_soft'], 0.15)};"
    elif abs_val > CORR_HIGH:
        return f"background-color: {hex_to_rgba(PALETTE['red'], 0.18)}; font-weight: 600;"
    elif abs_val > CORR_MODERATE:
        return f"background-color: {hex_to_rgba(PALETTE['amber'], 0.18)}; font-weight: 600;"
    else:
        return ""


# --------------- TAB 1: EDA ---------------
def render_eda_tab(
    settings: Settings,
    minute: int,
    team: str,
    min_game_duration: int
) -> None:
    scaled_df, n_dropped = data.get_pivoted_data(settings, minute, team, min_game_duration)
    error = data.load_error(scaled_df)
    if error is not None:
        st.error(f"Failed to load EDA data: {error}")
        return

    bundle = data.get_eda_bundle(settings, minute, team, min_game_duration)
    feature_cols = list(settings.feature_cols)
    lane_labels = settings.lane_labels

    render_context_caption(scaled_df)
    if n_dropped:
        st.caption(
            f"{n_dropped} matches excluded for incomplete lane data at minute {minute}. · "
            f"因分路数据不完整，已剔除 {n_dropped} 场比赛（第 {minute} 分钟）。"
        )

    # ----- Class Balance: bar chart instead of table -----
    balance = bundle["balance"]
    win_rate = float(balance["proportion"].iloc[0]) if not balance.empty else 0.0
    is_balanced = BALANCE_LOWER <= win_rate <= BALANCE_UPPER

    def _balance_body():
        labels = {
            0: "Loss",
            1: "Win"
        }
        order = [1, 0]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[labels[idx] for idx in order],
            y=[balance.loc[idx, "count"] if idx in balance.index else 0 for idx in order],
            text=[f"{balance.loc[idx, 'proportion']:.1%}" if idx in balance.index else "0.0%" for idx in order],
            textposition="outside",
            marker=dict(color=[PALETTE["green"], PALETTE["dark_red"]]),
        ))
        fig.update_layout(
            yaxis_title="Match count", height=280,
            margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
        )
        st.plotly_chart(fig)

    render_card(
        "Base Win Rate", "基本胜率",
        f"{team} win rate across the pulled matches.",
        _balance_body,
        status_text=f"{win_rate:.1%}",
        status_level=PILL_OK if is_balanced else PILL_WARN,
    )

    # ----- Feature Correlation -----
    corr = bundle["correlation"]
    max_offdiag = corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max() \
        if len(corr) > 1 \
        else 0.0

    def _corr_body():
        styled = (corr
            .round(3)
            .style.map(_highlight_correlation)
        )
        st.dataframe(styled, hide_index=False)

    if max_offdiag > CORR_HIGH:
        corr_pill_level = PILL_FAIL
    elif max_offdiag > CORR_MODERATE:
        corr_pill_level = PILL_WARN
    else:
        corr_pill_level = PILL_OK

    render_card(
        "Feature Correlation", "特征相关性",
        "Pairwise correlation between lane gold-diff features (multicollinearity check). "
        f"Row and column both represent the same 5 features. Cells above |{CORR_MODERATE}| are "
        f"highlighted yellow, above |{CORR_HIGH}| red.",
        _corr_body,
        status_text=f"max |r| = {max_offdiag:.2f}",
        status_level=corr_pill_level,
    )

    # ----- Feature Distributions -----
    render_card(
        "Feature Distributions", "特征分布",
        "Mean / std / skew of each lane's gold diff at this minute.",
        _df_body(bundle["distributions"].round(2), hide_index=False),
    )

    # ----- Gold Diff Distribution by Lane -----
    def _hist_body():
        fig = go.Figure()
        for col in feature_cols:
            en, zh = lane_labels[col]
            fig.add_trace(go.Histogram(
                x=scaled_df[col] * settings.gold_scale,
                name=f"{en} ({zh})",
                opacity=0.6,
            ))
        fig.update_layout(
            barmode="overlay",
            height=350,
            margin=DEFAULT_MARGIN,
            xaxis_title="Gold diff (g) · 金币差别",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.0,
                xanchor="center", x=0.5,
            ),
        )
        st.plotly_chart(fig)
        st.caption("Tip: click a lane in the legend to hide/show it · 按住上面分路按钮则展示/隐藏其分布")

    render_card(
        "Gold Diff Distribution by Lane", "分路和其金币分布",
        f"Histogram of lane gold diff at minute {minute}, per {settings.gold_scale:.0f}g unit.",
        _hist_body,
    )

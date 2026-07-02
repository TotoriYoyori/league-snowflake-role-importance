import textwrap
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from settings import Settings
from src import data as d
from src.model import predict_win_probability

# --------------- UI PILLS ---------------
PILL_OK = "ri-pill-ok"
PILL_WARN = "ri-pill-warn"
PILL_FAIL = "ri-pill-fail"
PILL_NEUTRAL = "ri-pill-neutral"


def _pill(text: str, level: str = PILL_NEUTRAL) -> str:
    return f'<span class="ri-pill {level}">{text}</span>'


# --------------- HEADER ---------------
def render_header(settings: Settings) -> None:
    mode_label = "Local / Mock" if settings.is_local else "Snowflake (live)"
    mode_level = PILL_NEUTRAL if settings.is_local else PILL_OK
    st.markdown(
        textwrap.dedent(f"""
        <div class="ri-header">
            <div class="ri-title">{settings.ui.app_title} {_pill(mode_label, mode_level)}</div>
            <div class="ri-subtitle-en">{settings.ui.app_subtitle_en}</div>
            <div class="ri-subtitle-zh">{settings.ui.app_subtitle_zh}</div>
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

print('')
# --------------- CARD (generalized: body is a render callback, not just a df) ---------------
def render_card(
    name_en: str,
    name_zh: str,
    desc_en: str,
    body: Callable[[], None],
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
) -> Callable[[], None]:
    return lambda: st.dataframe(
        df,
        width='stretch',
        height='stretch',
        hide_index=hide_index
    )

def _highlight_correlation(val: float) -> str:
    """Background color for a correlation cell:"""
    if pd.isna(val):
        return ""

    abs_val = abs(val)
    if abs_val >= 0.99:
        return "background-color: rgba(136, 136, 136, 0.15);"
    elif abs_val > 0.7:
        return "background-color: rgba(204, 31, 31, 0.18); font-weight: 600;"
    elif abs_val > 0.5:
        return "background-color: rgba(224, 138, 30, 0.18); font-weight: 600;"
    else:
        return ""


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

# ===========================================================================
# TAB 1: EDA
# ===========================================================================
def render_eda_tab(settings: Settings, minute: int, team: str) -> None:
    scaled_df, n_dropped = d.get_pivoted_data(settings, minute, team)
    bundle = d.get_eda_bundle(settings, minute, team)
    feature_cols = list(settings.model.feature_cols)
    lane_labels = settings.model.lane_labels

    render_context_caption(scaled_df)
    if n_dropped:
        st.caption(
            f"{n_dropped} matches excluded for incomplete lane data at minute {minute}. · "
            f"因分路数据不完整，已剔除 {n_dropped} 场比赛（第 {minute} 分钟）。"
        )

    # ----- Class Balance: bar chart instead of table -----
    balance = bundle["balance"]
    win_rate = float(balance["proportion"].iloc[0]) if not balance.empty else 0.0
    is_balanced = 0.4 <= win_rate <= 0.6

    def _balance_body():
        labels = {0: "Loss", 1: "Win"}
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[labels.get(idx, str(idx)) for idx in balance.index],
            y=balance["count"],
            text=[f"{p:.1%}" for p in balance["proportion"]],
            textposition="outside",
            marker_color=["#a81818" if labels.get(idx) == "Loss" else "#2f9e63" for idx in balance.index],
        ))
        fig.update_layout(
            yaxis_title="Match count", height=280,
            margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
        )
        st.plotly_chart(fig, width='stretch')

    render_card(
        "Base Win Rate", "基本胜率",
        f"{team} win rate across the pulled matches.",
        _balance_body,
        status_text=f"{win_rate:.1%}",
        status_level=PILL_OK if is_balanced else PILL_WARN,
    )

    # ----- Feature Correlation: index no longer hidden, so row labels (GOLD_DIFF_X) are visible -----
    corr = bundle["correlation"]
    max_offdiag = corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max() \
        if len(corr) > 1 \
        else 0.0

    def _corr_body():
        styled = (corr
            .round(3)
            .style.map(_highlight_correlation)
        )
        st.dataframe(styled, width='stretch', hide_index=False)

    if max_offdiag > 0.7:
        corr_pill_level = PILL_FAIL
    elif max_offdiag > 0.5:
        corr_pill_level = PILL_WARN
    else:
        corr_pill_level = PILL_OK

    render_card(
        "Feature Correlation", "特征相关性",
        "Pairwise correlation between lane gold-diff features (multicollinearity check). Row and column both represent the same 5 features. Cells above |0.5| are highlighted yellow, above |0.7| red.",
        _corr_body,
        status_text=f"max |r| = {max_offdiag:.2f}",
        status_level=corr_pill_level,
    )

    # ----- Feature Distributions: index no longer hidden, so mean/std/skew are attributable -----
    render_card(
        "Feature Distributions", "特征分布",
        "Mean / std / skew of each lane's gold diff at this minute.",
        _df_body(bundle["distributions"].round(2), hide_index=False),
    )

    # ----- Gold Diff Distribution by Lane: added legend-interaction hint -----
    def _hist_body():
        fig = go.Figure()
        for col in feature_cols:
            en, zh = lane_labels[col]
            fig.add_trace(go.Histogram(
                x=scaled_df[col] * settings.model.gold_scale,
                name=f"{en} ({zh})",
                opacity=0.6,
            ))
        fig.update_layout(
            barmode="overlay",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Gold diff (g) · 金币差别",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.0,
                xanchor="center", x=0.5,
            ),
        )
        st.plotly_chart(fig, width='stretch')
        st.caption("Tip: click a lane in the legend to hide/show it · 按住上面分路按钮则展示/隐藏其分布")

    render_card(
        "Gold Diff Distribution by Lane", "分路和其金币分布",
        f"Histogram of lane gold diff at minute {minute}, per {settings.model.gold_scale:.0f}g unit.",
        _hist_body,
    )


# ===========================================================================
# TAB 2: MODEL EVALUATION
# ===========================================================================
def render_evaluation_tab(settings: Settings, minute: int, team: str) -> None:
    bundle = d.get_eval_bundle(settings, minute, team)
    metrics = bundle["metrics"]
    auc = metrics["auc"]

    if auc >= settings.model.auc_ok_threshold:
        auc_level, auc_text = PILL_OK, f"AUC {auc:.3f} — OK"
    elif auc >= settings.model.auc_warn_threshold:
        auc_level, auc_text = PILL_WARN, f"AUC {auc:.3f} — WARN"
    else:
        auc_level, auc_text = PILL_FAIL, f"AUC {auc:.3f} — FAIL"

    c1, c2, c3 = st.columns(3)
    c1.metric("Test AUC", f"{auc:.3f}")
    c2.metric("Train matches", len(bundle["train_df"]))
    c3.metric("Test matches", len(bundle["test_df"]))

    def _roc_body():
        roc_df = bundle["roc_df"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc_df["fpr"], y=roc_df["tpr"], mode="lines", name=f"AUC = {auc:.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Chance"))
        fig.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=380, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width='stretch')

    render_card(
        "ROC Curve", "ROC 曲线",
        "Held-out test set performance. Early-minute models tend to score lower — there's more game left to play.",
        _roc_body,
        status_text=auc_text,
        status_level=auc_level,
    )

    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual: Loss", "Actual: Win"], columns=["Pred: Loss", "Pred: Win"])
    render_card(
        "Confusion Matrix", "混淆矩阵",
        "Held-out test set predictions at the 0.5 probability threshold.",
        _df_body(cm_df, hide_index=False),
    )

    report_df = pd.DataFrame(metrics["classification_report"]).T.round(3)
    render_card(
        "Classification Report", "分类报告",
        "Precision / recall / F1 per class, held-out test set. Rows 0/1 = Loss/Win class metrics.",
        _df_body(report_df, hide_index=False),
    )

    sm_table_labeled = bundle["sm_table"].round(4).rename(columns={"coef": "coef (train-split)"})
    render_card(
        "Coefficient Significance (statsmodels)", "系数显著性（statsmodels）",
        "p-values and 95% CIs from the 70% train split — for statistical significance "
        "testing only. Differs from the Lane Importance tab, since this is based on training set.",
        _df_body(sm_table_labeled, hide_index=False),
    )

    render_card(
        "Sample Predictions", "预测样例",
        "First 10 held-out matches: actual vs. predicted outcome.",
        _df_body(bundle["preview_df"].head(10).round(3)),
    )


# ===========================================================================
# TAB 3: LANE IMPORTANCE
# ===========================================================================
def render_importance_tab(settings: Settings, minute: int, team: str, n_splits: int, n_repeats: int) -> None:
    lane_df = d.get_lane_importance(settings, minute, team, n_splits, n_repeats)
    coef_df = d.get_full_model_coefficients(settings, minute, team)

    def _importance_body():
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=lane_df["odds_ratio"],
            y=lane_df["lane"],
            orientation="h",
            error_x=dict(type="data", array=lane_df["odds_ratio_err"]),
            marker_color="#a81818",
        ))
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title=f"Odds ratio per {settings.model.gold_scale:.0f}g lead",
            yaxis=dict(autorange="reversed"),
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width='stretch')

    render_card(
        "Lane Importance", "分路重要性",
        f"Odds multiplier per {settings.model.gold_scale:.0f}g lead, mean ± 95% CI over {n_splits}x{n_repeats} CV, "
        "fit on all available data. Track this across patches to see which lane's gold lead most decides game outcome. "
        "For train-split p-value significance testing, see Model Evaluation.",
        _importance_body,
    )

    render_card(
        "CV Coefficient Summary", "交叉验证系数汇总",
        "Full numeric stability summary underlying the chart above. Interpret as per ± 1,000g.",
        _df_body(lane_df.round(4)),
    )

    render_card(
        "Full-Data Model Coefficients", "全量数据模型系数",
        "Single fit on all available rows. Interpret as per ± 1,000g. ",
        _df_body(coef_df.round(4)),
    )


# ===========================================================================
# TAB 4: PREDICTOR
# ===========================================================================
PREDICTOR_PRESETS = {
    "ADC - Actually Doing Carry": {
        "GOLD_DIFF_BOTTOM": 0.7,
        "GOLD_DIFF_SUPPORT": 0.5,
        "GOLD_DIFF_JUNGLE": 0.0,
        "GOLD_DIFF_MIDDLE": -0.3,
        "GOLD_DIFF_TOP": -0.3,
    },
    "Jungle/Mid E-Couple": {
        "GOLD_DIFF_JUNGLE": 0.6,
        "GOLD_DIFF_MIDDLE": 0.6,
        "GOLD_DIFF_TOP": -0.7,
        "GOLD_DIFF_BOTTOM": -0.2,
        "GOLD_DIFF_SUPPORT": -0.2,
    },
    "Talon Camps Top": {
        "GOLD_DIFF_MIDDLE": 0.6,
        "GOLD_DIFF_TOP": 0.6,
        "GOLD_DIFF_JUNGLE": -0.3,
        "GOLD_DIFF_BOTTOM": -0.1,
        "GOLD_DIFF_SUPPORT": 0.0,
    },
    "Jungle Diff": {
        "GOLD_DIFF_JUNGLE": -0.8,
        "GOLD_DIFF_TOP": -0.1,
        "GOLD_DIFF_MIDDLE": 0.0,
        "GOLD_DIFF_BOTTOM": -0.1,
        "GOLD_DIFF_SUPPORT": 0.0,
    },
    "Fed Top": {
        "GOLD_DIFF_TOP": 0.8,
        "GOLD_DIFF_JUNGLE": -0.2,
        "GOLD_DIFF_MIDDLE": -0.5,
        "GOLD_DIFF_BOTTOM": -0.3,
        "GOLD_DIFF_SUPPORT": -0.2,
    },
}


def _apply_preset(
    preset_name: str,
    feature_cols: list[str],
    bounds: dict,
    step: float,
    gold_scale: float
) -> None:
    preset = PREDICTOR_PRESETS[preset_name]
    for feature in feature_cols:
        frac = preset.get(feature, 0.0)
        lo_scaled, hi_scaled = bounds[feature]
        lo_raw = round((lo_scaled * gold_scale) / step) * step
        hi_raw = round((hi_scaled * gold_scale) / step) * step

        target = hi_raw * frac if frac >= 0 else lo_raw * abs(frac)
        target = round(target / step) * step
        st.session_state[f"predictor_{feature}"] = float(target)


def render_predictor_tab(settings: Settings, minute: int, team: str) -> None:
    feature_cols = list(settings.model.feature_cols)
    lane_labels = settings.model.lane_labels
    bounds = d.get_predictor_bounds(settings, minute, team)
    model = d.get_full_model(settings, minute, team)
    gold_scale = settings.model.gold_scale
    step = settings.model.predictor_slider_step

    render_section_header("Lane Gold Diff", "各分路经济差")
    st.caption(
        "Each slider is that lane's gold difference against its direct lane opponent — "
        "positive = ahead, negative = behind — at the selected minute."
    )
    st.caption(
        "New here? Click a scenario below to auto-fill the sliders with a realistic "
        "example, then tweak individual lanes to explore your own what-if."
    )

    preset_cols = st.columns(len(PREDICTOR_PRESETS))
    for preset_col, preset_name in zip(preset_cols, PREDICTOR_PRESETS):
        with preset_col:
            if st.button(preset_name, width="stretch"):
                _apply_preset(preset_name, feature_cols, bounds, step, gold_scale)
                st.rerun()

    gold_diffs = {}
    cols = st.columns(len(feature_cols))
    for col, feature in zip(cols, feature_cols):
        en, zh = lane_labels[feature]
        lo_scaled, hi_scaled = bounds.get(
            feature,
            (settings.model.predictor_slider_min / gold_scale, settings.model.predictor_slider_max / gold_scale),
        )
        lo_raw = round((lo_scaled * gold_scale) / step) * step
        hi_raw = round((hi_scaled * gold_scale) / step) * step

        slider_key = f"predictor_{feature}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = 0.0  # seed once, only if not already set

        with col:
            raw_value = st.slider(
                f"{en} ({zh})",
                min_value=float(lo_raw), max_value=float(hi_raw),
                step=float(step),
                format="%.0f",
                key=slider_key,
            )
        gold_diffs[feature] = raw_value / gold_scale

    proba = predict_win_probability(model, feature_cols, gold_diffs)
    st.metric(f"P({team} wins)", f"{proba:.1%}")

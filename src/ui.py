
import textwrap
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from settings import Settings
from src import data as d

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


def _df_body(df: pd.DataFrame, height: int | None = None) -> Callable[[], None]:
    return lambda: st.dataframe(df, use_container_width=True, height=height or "content", hide_index=True)


# --------------- CONTEXT CAPTION (AVERAGE_RANK / GAME_DATE — informational only) ---------------
def render_context_caption(df: pd.DataFrame) -> None:
    if df.empty:
        return
    date_min, date_max = df["GAME_DATE"].min(), df["GAME_DATE"].max()
    rank_counts = df["AVERAGE_RANK"].value_counts()
    top_rank = rank_counts.index[0] if not rank_counts.empty else "n/a"
    st.caption(
        f"Data spans {date_min} to {date_max} · "
        f"{len(df)} matches · most common rank tier: {top_rank}"
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
        st.caption(f"{n_dropped} matches excluded for incomplete lane data at minute {minute}.")

    balance = bundle["balance"]
    win_rate = float(balance["proportion"].iloc[0]) if not balance.empty else 0.0
    is_balanced = 0.4 <= win_rate <= 0.6
    render_card(
        "Class Balance", "类别平衡",
        f"{team} win rate across the pulled matches.",
        _df_body(balance),
        status_text=f"{win_rate:.1%}",
        status_level=PILL_OK if is_balanced else PILL_WARN,
    )

    corr = bundle["correlation"]
    max_offdiag = corr.where(~np.eye(len(corr), dtype=bool)).abs().max().max() \
        if len(corr) > 1 \
        else 0.0
    render_card(
        "Feature Correlation", "特征相关性",
        "Pairwise correlation between lane gold-diff features (multicollinearity check).",
        _df_body(corr.round(3)),
        status_text=f"max |r| = {max_offdiag:.2f}",
        status_level=PILL_WARN if max_offdiag > 0.7 else PILL_OK,
    )

    render_card(
        "Feature Distributions", "特征分布",
        "Mean / std / skew of each lane's gold diff at this minute.",
        _df_body(bundle["distributions"].round(2)),
    )

    def _hist_body():
        fig = go.Figure()
        for col in feature_cols:
            en, _ = lane_labels[col]
            fig.add_trace(go.Histogram(x=scaled_df[col], name=en, opacity=0.6))
        fig.update_layout(barmode="overlay", height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    render_card(
        "Gold Diff Distribution by Lane", "各分路经济差分布",
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
        st.plotly_chart(fig, use_container_width=True)

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
        _df_body(cm_df),
    )

    report_df = pd.DataFrame(metrics["classification_report"]).T.round(3)
    render_card(
        "Classification Report", "分类报告",
        "Precision / recall / F1 per class, held-out test set.",
        _df_body(report_df),
    )

    render_card(
        "Coefficient Significance (statsmodels)", "系数显著性（statsmodels）",
        "p-values and 95% CIs from the train-split fit — internal-team reference, not hidden.",
        _df_body(bundle["sm_table"].round(4)),
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
            x=lane_df["odds_ratio"], y=lane_df["lane"], orientation="h",
            error_x=dict(type="data", array=lane_df["odds_ratio_err"]),
            marker_color="#a81818",
        ))
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title=f"Odds ratio per {settings.model.gold_scale:.0f}g lead",
            height=380, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    render_card(
        "Lane Importance", "分路重要性",
        f"Odds multiplier per {settings.model.gold_scale:.0f}g lead, mean ± 95% CI over {n_splits}x{n_repeats} CV. "
        "Track this across patches to see which lane's gold lead most decides game outcome.",
        _importance_body,
    )

    render_card(
        "CV Coefficient Summary", "交叉验证系数汇总",
        "Full numeric stability summary underlying the chart above.",
        _df_body(lane_df.round(4)),
    )

    render_card(
        "Full-Data Model Coefficients", "全量数据模型系数",
        "Single fit on all available rows — the model of record also used by the Predictor tab.",
        _df_body(coef_df.round(4)),
    )


# ===========================================================================
# TAB 4: PREDICTOR
# ===========================================================================
def render_predictor_tab(settings: Settings, minute: int, team: str) -> None:
    feature_cols = list(settings.model.feature_cols)
    lane_labels = settings.model.lane_labels
    bounds = d.get_predictor_bounds(settings, minute, team)
    model = d.get_full_model(settings, minute, team)

    st.caption(
        f"Sliders are bounded to the 1st-99th percentile of observed gold diffs at minute {minute}, "
        f"in units of {settings.model.gold_scale:.0f}g."
    )

    gold_diffs = {}
    cols = st.columns(len(feature_cols))
    for col, feature in zip(cols, feature_cols):
        en, zh = lane_labels[feature]
        lo, hi = bounds.get(feature, (settings.model.predictor_slider_min / settings.model.gold_scale,
                                       settings.model.predictor_slider_max / settings.model.gold_scale))
        with col:
            gold_diffs[feature] = st.slider(
                f"{en} ({zh})", min_value=float(lo), max_value=float(hi), value=0.0, step=0.1,
                key=f"predictor_{feature}",
            )

    from src.model import predict_win_probability
    proba = predict_win_probability(model, feature_cols, gold_diffs)

    st.metric(f"P({team} wins)", f"{proba:.1%}")

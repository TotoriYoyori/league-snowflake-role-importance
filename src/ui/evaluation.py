import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from settings import Settings
from src import data
from src.ui.components import PILL_FAIL, PILL_OK, PILL_WARN, _df_body, render_card
from src.ui.theme import DEFAULT_MARGIN

# --------------- AUC THRESHOLDS (only this tab's pill reads them) ---------------
AUC_OK_THRESHOLD = 0.80
AUC_WARN_THRESHOLD = 0.65

# NOTE: the win/loss decision threshold itself (default 0.5) lives in
# Settings.win_loss_threshold — it's a modeling concern (affects the
# confusion matrix / classification report), not a display-only constant
# like the two above, so it doesn't belong here.


# --------------- TAB 2: MODEL EVALUATION ---------------
def render_evaluation_tab(settings: Settings, minute: int, team: str, min_game_duration: int) -> None:
    bundle = data.get_eval_bundle(settings, minute, team, min_game_duration)
    if "error" in bundle:
        st.error(f"Failed to load evaluation data: {bundle['error']}")
        return

    metrics = bundle["metrics"]
    auc = metrics["auc"]

    if auc >= AUC_OK_THRESHOLD:
        auc_level, auc_text = PILL_OK, f"AUC {auc:.3f} — OK"
    elif auc >= AUC_WARN_THRESHOLD:
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
            height=380, margin=DEFAULT_MARGIN,
        )
        st.plotly_chart(fig)

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
        f"Held-out test set predictions at the {settings.win_loss_threshold:.2f} probability threshold.",
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
        "p-values and 95% CIs from the train split — for statistical significance "
        "testing only. Differs from the Lane Importance tab, since this is based on training set.",
        _df_body(sm_table_labeled, hide_index=False),
    )

    render_card(
        "Sample Predictions", "预测样例",
        "First 10 held-out matches: actual vs. predicted outcome.",
        _df_body(bundle["preview_df"].head(10).round(3)),
    )

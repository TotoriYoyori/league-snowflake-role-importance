import plotly.graph_objects as go
import streamlit as st

from settings import Settings
from src import data
from src.ui.components import _df_body, render_card
from src.ui.theme import DEFAULT_MARGIN, PALETTE


# --------------- TAB 3: LANE IMPORTANCE ---------------
def render_importance_tab(
    settings: Settings,
    minute: int,
    team: str,
    n_splits: int,
    n_repeats: int,
    min_game_duration: int
) -> None:
    lane_df = data.get_lane_importance(settings, minute, team, n_splits, n_repeats, min_game_duration)
    error = data.load_error(lane_df)
    if error is not None:
        st.error(f"Failed to load lane importance data: {error}")
        return

    coef_df = data.get_full_model_coefficients(settings, minute, team, min_game_duration)
    error = data.load_error(coef_df)
    if error is not None:
        st.error(f"Failed to load model coefficients: {error}")
        return

    def _importance_body():
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=lane_df["odds_ratio"],
            y=lane_df["lane"],
            orientation="h",
            error_x=dict(type="data", array=lane_df["odds_ratio_err"]),
            marker=dict(color=PALETTE["dark_red"]),
        ))
        fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
        fig.update_layout(
            xaxis_title=f"Odds ratio per {settings.gold_scale:.0f}g lead",
            yaxis=dict(autorange="reversed"),
            height=380,
            margin=DEFAULT_MARGIN,
        )
        st.plotly_chart(fig)

    render_card(
        "Lane Importance", "分路重要性",
        f"Odds multiplier per {settings.gold_scale:.0f}g lead, mean ± 95% CI over {n_splits}x{n_repeats} CV, "
        "fit on all available data. Track this across patches to see which lane's gold lead most decides game outcome. "
        "For train-split p-value significance testing, see Model Evaluation.",
        _importance_body,
    )

    render_card(
        "CV Coefficient Summary", "交叉验证系数汇总",
        f"Full numeric stability summary underlying the chart above. Interpret as per ± {settings.gold_scale:.0f}g.",
        _df_body(lane_df.round(4)),
    )

    render_card(
        "Full-Data Model Coefficients", "全量数据模型系数",
        f"Single fit on all available rows. Interpret as per ± {settings.gold_scale:.0f}g.",
        _df_body(coef_df.round(4)),
    )

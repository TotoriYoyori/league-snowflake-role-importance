import streamlit as st

from settings import Settings
from src import data
from src.model import predict_win_probability
from src.ui.components import render_section_header

# --------------- SLIDER PRESETS ---------------
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


def _slider_key(
    feature: str,
    minute: int,
    team: str,
    min_game_duration: int
) -> str:
    """For generating session state key with Streamlit based on current slider value."""
    return f"predictor_{feature}_{minute}_{team}_{min_game_duration}"


def _prune_stale_slider_keys(minute: int, team: str, min_game_duration: int) -> None:
    """Evict compiling state keys if you play with the sliders too much"""
    current_suffix = f"_{minute}_{team}_{min_game_duration}"
    stale_keys = [
        key for key in st.session_state
        if key.startswith("predictor_") and not key.endswith(current_suffix)
    ]
    for key in stale_keys:
        del st.session_state[key]


def _apply_preset(
    preset_name: str,
    feature_cols: list[str],
    bounds: dict,
    step: float,
    gold_scale: float,
    minute: int,
    team: str,
    min_game_duration: int,
) -> None:
    preset = PREDICTOR_PRESETS[preset_name]
    for feature in feature_cols:
        frac = preset.get(feature, 0.0)
        lo_scaled, hi_scaled = bounds[feature]
        lo_raw = round((lo_scaled * gold_scale) / step) * step
        hi_raw = round((hi_scaled * gold_scale) / step) * step

        target = hi_raw * frac if frac >= 0 else lo_raw * abs(frac)
        target = round(target / step) * step
        st.session_state[_slider_key(feature, minute, team, min_game_duration)] = float(target)


# --------------- RENDERING ---------------
def render_predictor_tab(settings: Settings, minute: int, team: str, min_game_duration: int) -> None:
    _prune_stale_slider_keys(minute, team, min_game_duration)

    feature_cols = list(settings.feature_cols)
    lane_labels = settings.lane_labels
    bounds = data.get_predictor_bounds(settings, minute, team, min_game_duration)
    model = data.get_full_model(settings, minute, team, min_game_duration)
    if model is None:
        st.error("Failed to load model. Predictor unavailable.")
        return

    gold_scale = settings.gold_scale
    step = settings.predictor_slider_step

    render_section_header("Lane Gold Diff", "各分路经济差")
    st.caption(
        "Each slider is that lane's gold difference against its direct lane opponent: "
        "positive = ahead, negative = behind (at the selected minute)."
    )
    st.caption(
        "New here? Click a scenario below to auto-fill the sliders with a realistic "
        "example, then tweak individual lanes to explore your own what-if."
    )

    preset_cols = st.columns(len(PREDICTOR_PRESETS))
    for preset_col, preset_name in zip(preset_cols, PREDICTOR_PRESETS):
        with preset_col:
            if st.button(preset_name, key=f"preset_{preset_name}_{minute}_{team}_{min_game_duration}"):
                _apply_preset(preset_name, feature_cols, bounds, step, gold_scale, minute, team, min_game_duration)
                st.rerun()

    gold_diffs = {}
    cols = st.columns(len(feature_cols))
    for col, feature in zip(cols, feature_cols):
        en, zh = lane_labels[feature]
        lo_scaled, hi_scaled = bounds.get(
            feature,
            (settings.predictor_slider_min / gold_scale, settings.predictor_slider_max / gold_scale),
        )
        lo_raw = round((lo_scaled * gold_scale) / step) * step
        hi_raw = round((hi_scaled * gold_scale) / step) * step

        slider_key = _slider_key(feature, minute, team, min_game_duration)
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

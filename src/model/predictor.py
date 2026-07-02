import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.model.importance import fit_full_data_model


# --------------- SCORE A WHAT-IF SCENARIO ---------------
def predict_win_probability(
    model: LogisticRegression,
    feature_cols: list[str],
    gold_diffs: dict[str, float],
) -> float:
    """
    gold_diffs: {feature_col: value_in_scaled_units}, e.g.
    {"GOLD_DIFF_TOP": 0.5, "GOLD_DIFF_JUNGLE": -2.0, ...} for +500g Top,
    -2000g Jungle (already divided by gold_scale=1000 — caller's job to
    scale raw slider values the same way training data was scaled).

    Returns P(win) for the configured team, a single float in [0, 1].
    """
    X = pd.DataFrame([{
         col: gold_diffs.get(col, 0.0)
         for col in feature_cols
    }])

    return float(model.predict_proba(X[feature_cols])[0, 1])


# --------------- DATA-DRIVEN SLIDER BOUNDS ---------------
def predictor_slider_bounds(
    df: pd.DataFrame,
    feature_cols: list[str],
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
) -> dict[str, tuple[float, float]]:
    """
    Per-lane (min, max) slider bounds from the observed training data's
    [lower_pct, upper_pct] percentile range (default 1st-99th), in the same
    scaled units as the model was trained on.
    """
    bounds = {}
    for col in feature_cols:
        lo, hi = np.percentile(df[col], [lower_pct, upper_pct])
        bounds[col] = (float(lo), float(hi))
    return bounds

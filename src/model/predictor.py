import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# --------------- SCORE A WHAT-IF SCENARIO ---------------
def predict_win_probability(
    model: LogisticRegression,
    feature_cols: list[str],
    gold_diffs: dict[str, float],
) -> float:
    """gold_diffs: {feature_col: value_in_scaled_units}, e.g. {"GOLD_DIFF_TOP": 0.5, ...}. Returns P(win) in [0, 1]."""
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
    """Per-lane (min, max) slider bounds from the training data's [lower_pct, upper_pct] percentile range."""
    bounds = {}
    for col in feature_cols:
        lo, hi = np.percentile(df[col], [lower_pct, upper_pct])
        bounds[col] = (float(lo), float(hi))
    return bounds

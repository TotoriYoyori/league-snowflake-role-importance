import pandas as pd


# --------------- FEATURE DISTRIBUTIONS ---------------
def feature_distributions(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Mean/std/skew per lane gold-diff feature. ui.py renders this as a
    table alongside a histogram built from df[feature_cols] directly."""
    return df[feature_cols].agg(["mean", "std", "skew"]).T


# --------------- FEATURE CORRELATION ---------------
def feature_correlation(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Pairwise correlation between lane gold-diff features — checks
    multicollinearity risk. ui.py renders this as a styled table/heatmap."""
    return df[feature_cols].corr()


# --------------- CLASS BALANCE ---------------
def class_balance(
    df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """Win/loss count + proportion for the target column. ui.py uses the
    proportion to drive the class-balance pill (near-50/50 = OK)."""
    return (
        df[target_col]
        .value_counts()
        .rename("count")
        .to_frame()
        .assign(proportion=lambda d: d["count"] / d["count"].sum())
    )

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedKFold

from src.model.evaluation import fit_logistic_regression

# --------------- FULL-DATA FIT (MODEL OF RECORD) ---------------
def fit_full_data_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    **lr_kwargs,
) -> LogisticRegression:
    """Same fit logic as evaluation.fit_logistic_regression, called on the full dataset instead of a train split."""
    return fit_logistic_regression(
        df, feature_cols, target_col, **lr_kwargs
    )

# --------------- CV COEFFICIENT STABILITY (ALSO ON FULL DATA) ---------------
def coefficient_stability_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    n_splits: int = 10,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Refits across repeated k-folds of the full dataset for a sampling distribution of coefficients (more stable than a single fit)."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    X = df[feature_cols]
    y = df[target_col]

    fold_coefs = []
    for train_idx, _ in rkf.split(X):
        X_fold, y_fold = X.iloc[train_idx], y.iloc[train_idx]
        model = LogisticRegression(max_iter=1000).fit(X_fold, y_fold)
        fold_coefs.append(model.coef_[0])

    fold_coefs = np.array(fold_coefs)

    return (pd
        .DataFrame({
            "feature": feature_cols,
            "mean_coef": fold_coefs.mean(axis=0),
            "std_coef": fold_coefs.std(axis=0, ddof=1),
        })
        .assign(ci_lower_95=lambda df: df["mean_coef"] - 1.96 * df["std_coef"])
        .assign(ci_upper_95=lambda df: df["mean_coef"] + 1.96 * df["std_coef"])
        .assign(ci_lower_95_pct=np.percentile(fold_coefs, 2.5, axis=0))
        .assign(ci_upper_95_pct=np.percentile(fold_coefs, 97.5, axis=0))
        .sort_values("mean_coef", ascending=False)
        .reset_index(drop=True)
    )


# --------------- LANE IMPORTANCE (ODDS-RATIO TRANSFORM FOR DISPLAY) ---------------
def lane_importance(
    stability_summary: pd.DataFrame,
    decimals: int = 4,
) -> pd.DataFrame:
    """Transforms the CV coefficient summary into odds ratios (exp(coef))."""
    result = (stability_summary
        .assign(lane=lambda df: df["feature"].str.replace("GOLD_DIFF_", ""))
        .assign(odds_ratio=lambda df: np.exp(df["mean_coef"]))
        .assign(odds_ratio_lower=lambda df: np.exp(df["ci_lower_95"]))
        .assign(odds_ratio_upper=lambda df: np.exp(df["ci_upper_95"]))
        .assign(odds_ratio_err=lambda df: (df["odds_ratio_upper"] - df["odds_ratio_lower"]) / 2)
    )

    output_cols = [
        "lane", "mean_coef", "std_coef", "ci_lower_95", "ci_upper_95",
        "odds_ratio", "odds_ratio_lower", "odds_ratio_upper", "odds_ratio_err",
    ]
    return result[output_cols].round(decimals)

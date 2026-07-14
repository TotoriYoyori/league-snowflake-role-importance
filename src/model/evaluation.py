import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


# --------------- FIT (TRAIN SPLIT ONLY) ---------------
def fit_logistic_regression(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    **lr_kwargs,
) -> LogisticRegression:
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    model = LogisticRegression(max_iter=1000, **lr_kwargs)
    model.fit(X_train, y_train)

    return model


# --------------- COEFFICIENT REPORT (SKLEARN) ---------------
def report_model_coefficients(
    model: LogisticRegression,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Coefficients + odds ratios from the sklearn model. For p-values/CIs,
    use statsmodels_coefficient_table() instead."""
    return (pd
        .DataFrame({
            "feature": feature_cols,
            "coefficient": model.coef_[0],
            "odds_ratio": np.exp(model.coef_[0]),
        })
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )


# --------------- STATSMODELS COEFFICIENT TABLE, PRE-PARSED FOR UI ---------------
def statsmodels_coefficient_table(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Fits a statsmodels Logit purely for its inferential statistics.
    """
    X_sm = sm.add_constant(train_df[feature_cols])
    y_sm = train_df[target_col]
    sm_model = sm.Logit(y_sm, X_sm).fit(disp=0)

    conf_int = sm_model.conf_int(alpha=0.05)
    conf_int.columns = ["ci_lower_95", "ci_upper_95"]

    table = (pd
        .DataFrame({
            "feature": sm_model.params.index,
            "coef": sm_model.params.values,
            "std_err": sm_model.bse.values,
            "z": sm_model.tvalues.values,
            "p_value": sm_model.pvalues.values,
        })
        .join(conf_int.reset_index(drop=True))
        .reset_index(drop=True)
    )

    table = pd.concat([
        table[table["feature"] != "const"],
        table[table["feature"] == "const"],
    ]).reset_index(drop=True)

    return table


# --------------- HELD-OUT EVALUATION ---------------
def evaluate_on_test(
    model: LogisticRegression,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    threshold: float = 0.5,
) -> dict:
    """threshold is the win/loss decision cutoff on predicted probability —
    passed explicitly (rather than relying on sklearn's fixed-0.5
    model.predict()) so it stays in lockstep with preview_predictions()
    and with whatever the caller configures via Settings.win_loss_threshold."""
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_proba": y_proba,
    }


# --------------- PREVIEW PREDICTIONS (SAMPLE ROWS) ---------------
def preview_predictions(
    test_df: pd.DataFrame,
    y_proba: np.ndarray,
    target_col: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    return (test_df[["MATCH_ID", target_col]]
        .rename(columns={target_col: "actual"})
        .assign(predicted_proba=y_proba)
        .assign(predicted=(y_proba >= threshold).astype(int))
        .assign(correct=lambda df: df["actual"] == df["predicted"])
    )


# --------------- ROC CURVE DATA ---------------
def roc_curve_data(
    y_test: pd.Series,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    """Returns the fpr/tpr/threshold points; ui.py builds the Plotly ROC
    plot (needs the diagonal reference line + AUC annotation) on top."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})

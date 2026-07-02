# Public API for src/model/.
from src.model.prep import (
    pivot_diff_interval,
    scale_gold_diff,
    split_train_test,
    LANE_RENAME,
    FEATURE_COLS
)
from src.model.eda import (
    feature_distributions,
    feature_correlation,
    class_balance
)
from src.model.evaluation import (
    fit_logistic_regression,
    report_model_coefficients,
    statsmodels_coefficient_table,
    evaluate_on_test,
    preview_predictions,
    roc_curve_data,
)
from src.model.importance import (
    fit_full_data_model,
    coefficient_stability_cv,
    lane_importance,
)
from src.model.predictor import predict_win_probability, predictor_slider_bounds

__all__ = [
    "pivot_diff_interval", "scale_gold_diff", "split_train_test",
    "LANE_RENAME", "FEATURE_COLS",
    "feature_distributions", "feature_correlation", "class_balance",
    "fit_logistic_regression", "report_model_coefficients",
    "statsmodels_coefficient_table", "evaluate_on_test", "preview_predictions",
    "roc_curve_data",
    "fit_full_data_model", "coefficient_stability_cv", "lane_importance",
    "predict_win_probability", "predictor_slider_bounds",
]
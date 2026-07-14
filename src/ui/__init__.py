# Public API for src/ui/.
from src.ui.components import TAB_LABELS, render_header, render_section_header
from src.ui.eda import render_eda_tab
from src.ui.evaluation import render_evaluation_tab
from src.ui.importance import render_importance_tab
from src.ui.predictor import render_predictor_tab

__all__ = [
    "TAB_LABELS", "render_header", "render_section_header",
    "render_eda_tab", "render_evaluation_tab",
    "render_importance_tab", "render_predictor_tab",
]

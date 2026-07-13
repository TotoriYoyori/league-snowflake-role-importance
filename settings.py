from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    # ----- Data Preparations
    gold_scale: float = Field(default=1000.0, gt=0)
    test_size: float = Field(default=0.3, gt=0.0, lt=1.0)
    random_state: int = 42

    # ----- User Inputs
    minute_options: tuple[int, ...] = (5, 10, 15, 20, 25, 30)
    default_minute: int = 15
    team_options: tuple[str, ...] = ("Blue", "Red")
    default_team: str = "Blue"

    # Minimum match length to include, in minutes — user-adjustable
    min_game_duration_options: tuple[int, ...] = Field(default=(0, 5, 10, 15, 20))
    default_min_game_duration_minutes: int = Field(default=5, ge=0)

    predictor_slider_min: float = -6000.0
    predictor_slider_max: float = 6000.0
    predictor_slider_step: float = 100.0

    cv_n_splits_default: int = Field(default=10, gt=1)
    cv_n_splits_range: tuple[int, int] = (5, 20)
    cv_n_repeats_default: int = Field(default=5, ge=1)
    cv_n_repeats_range: tuple[int, int] = (1, 10)

    # ----- AUC threshold for the Model Evaluation tab's pill.
    auc_ok_threshold: float = Field(default=0.80, gt=0.5, le=1.0)
    auc_warn_threshold: float = Field(default=0.65, gt=0.5, le=1.0)

    # ----- Labeling
    raw_lane_names: tuple[str, ...] = ("Top", "Jungle", "Middle", "Bottom", "Support")
    lane_labels_zh: dict[str, str] = Field(
        default_factory=lambda: {
            "Top": "上单",
            "Jungle": "打野",
            "Middle": "中单",
            "Bottom": "下路",
            "Support": "辅助",
        }
    )

    model_config = ConfigDict(frozen=True)


    @property
    def feature_cols(self) -> tuple[str, ...]:
        """Pivoted column names, e.g. GOLD_DIFF_TOP, in raw_lane_names order."""
        return tuple(f"GOLD_DIFF_{lane.upper()}" for lane in self.raw_lane_names)

    @property
    def lane_rename_map(self) -> dict[str, str]:
        """Raw LANE value -> pivoted GOLD_DIFF_X column name, e.g. {'Top': 'GOLD_DIFF_TOP'}."""
        return dict(zip(self.raw_lane_names, self.feature_cols))

    @property
    def lane_labels(self) -> dict[str, tuple[str, str]]:
        """Pivoted column name -> (English, Chinese) display label."""
        return {
            f"GOLD_DIFF_{lane.upper()}": (lane, self.lane_labels_zh[lane])
            for lane in self.raw_lane_names
        }


# --------------- SINGLETON ---------------
SETTINGS = Settings()

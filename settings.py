import os

import streamlit as st
from pydantic import BaseModel, ConfigDict, Field


class SnowflakeSettings(BaseModel):
    """Base Snowflake schema specifications"""
    model_config = ConfigDict(frozen=True)

    database: str = "LEAGUE_RECORDS"
    warehouse: str = "COMPUTE_WH"
    schema_: str = Field(default="GOLD", alias="schema")
    connection_ttl: int | None = None  # None = forever


class CacheSettings(BaseModel):
    """Determine time to live duration for cached data since model refits are expensive."""
    model_config = ConfigDict(frozen=True)

    source_data_ttl: int = Field(default=600, gt=0)
    baseline_fit_ttl: int = Field(default=600, gt=0)
    full_fit_ttl: int = Field(default=600, gt=0)
    cv_stability_ttl: int = Field(default=600, gt=0)


class ModelSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # ----- Data Preparations
    gold_scale: float = Field(default=1000.0, gt=0)
    test_size: float = Field(default=0.3, gt=0.0, lt=1.0)
    random_state: int = 42
    min_game_duration: int = Field(default=300, ge=0)  # less than 300s match are excluded

    # ----- User Inputs
    minute_options: tuple[int, ...] = (5, 10, 15, 20, 25, 30)
    default_minute: int = 15
    team_options: tuple[str, ...] = ("Blue", "Red")
    default_team: str = "Blue"

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


class UISettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_title: str = "Role Importance"
    app_subtitle_en: str = "League of Legends Lane Quest Impact — Win Coefficient Tracker"
    app_subtitle_zh: str = "英雄联盟符文任务影响 — 分路胜率系数追踪"
    tab_labels: tuple[tuple[str, str], ...] = (
        ("EDA", "探索性分析"),
        ("Model Evaluation", "模型评估"),
        ("Lane Importance", "分路重要性"),
        ("Predictor", "预测器"),
    )

# --------------- ORCHESTRATING ALL ABOVE SETTINGS ---------------
class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    env: str
    is_local: bool
    snowflake: SnowflakeSettings
    cache: CacheSettings
    model: ModelSettings
    ui: UISettings
    sample_data_dir: str


# --------------- SNOWFLAKE ENVIRONMENT DETECTOR ---------------
def _running_in_snowflake() -> bool:
    """Reliable SiS detection: the session token only exists inside Snowflake."""
    return os.path.isfile("/snowflake/session/token")


@st.cache_resource
def get_settings() -> Settings:
    forced_env = os.environ.get("APP_ENV", "").strip().lower()
    if forced_env in ("local", "production"):
        is_local = forced_env == "local"
    else:
        is_local = not _running_in_snowflake()

    return Settings(
        env="local" if is_local else "production",
        is_local=is_local,
        snowflake=SnowflakeSettings(),
        cache=CacheSettings(),
        model=ModelSettings(),
        ui=UISettings(),
        sample_data_dir=os.environ.get(
            "LOCAL_DATA_DIR",
            os.path.join(os.path.dirname(__file__), "assets", "sample_data"),
        ),
    )

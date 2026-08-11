import pandas as pd
from sklearn.model_selection import train_test_split

# --------------- NAMING CONSTANTS ---------------
# Keys are uppercase to match SILVER.PLAYERS' cleaning view (UPPER(ROLE)), which is what
# GOLD.DIFF_INTERVALS.CHAMPION_ROLE actually stores (e.g. "TOP", not "Top").
CHAMPION_ROLE_RENAME = {
    "TOP": "GOLD_DIFF_TOP",
    "JUNGLE": "GOLD_DIFF_JUNGLE",
    "MIDDLE": "GOLD_DIFF_MIDDLE",
    "BOTTOM": "GOLD_DIFF_BOTTOM",
    "SUPPORT": "GOLD_DIFF_SUPPORT",
}
FEATURE_COLS = list(CHAMPION_ROLE_RENAME.values())


# --------------- PIVOT LONG -> WIDE ---------------
def pivot_diff_interval(
    df: pd.DataFrame,
    win_reference_team: str = "Blue",
) -> tuple[pd.DataFrame, int]:
    """One row per (match, lane) -> one row per match. n_dropped is the count excluded for incomplete lane data."""
    win_col = f"{win_reference_team.upper()}_WIN"

    pivoted_df = (df
        .pivot_table(
            index=["MATCH_ID", "WINNING_TEAM", "AVERAGE_RANK", "GAME_DATE", "GAME_DURATION"],
            columns="CHAMPION_ROLE",
            values="GOLD_DIFF",
        )
        .reset_index()
        .rename(columns=CHAMPION_ROLE_RENAME)
        .assign(**{win_col: lambda d: (d["WINNING_TEAM"] == win_reference_team.upper()).astype(int)})
    )
    before = len(pivoted_df)

    pivoted_df = (pivoted_df
        .dropna(subset=FEATURE_COLS)
        .reset_index(drop=True)
    )
    n_dropped = before - len(pivoted_df)

    return pivoted_df, n_dropped


# --------------- SCALE GOLD DIFF (COSMETIC, DOES NOT CHANGE KPIs) ---------------
def scale_gold_diff(
    df: pd.DataFrame,
    feature_cols: list[str],
    scale: float = 1000,
) -> pd.DataFrame:
    """Rescale features (raw gold units) to per-`scale`-gold units, for
    coefficient legibility. Purely cosmetic: does not change end model KPI."""
    return df.assign(**{
        col: df[col] / scale
        for col in feature_cols
    })


# --------------- TRAIN/TEST SPLIT ---------------
def split_train_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split so win-rate is preserved in both halves. Used only by the Model Evaluation tab."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return train_df, test_df

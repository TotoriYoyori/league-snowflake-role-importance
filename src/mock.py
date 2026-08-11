import os

import pandas as pd


# --------------- FILE CONSTANTS ---------------
SAMPLE_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "assets",
    "sample_data"
)
DIFF_INTERVAL_CSV = "sample_diff_intervals"
MATCH_SUMMARY_CSV = "sample_matchend_pivot_teamstats"


# --------------- MOCK SAMPLE DATA ---------------
def _read(sample_data_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(sample_data_dir, f"{name}.csv")
    return pd.read_csv(path)


def diff_interval_by_match(
    minute: int,
    team: str,
    min_game_duration: int,
) -> pd.DataFrame:
    diff = _read(SAMPLE_DATA_DIR, DIFF_INTERVAL_CSV)
    match = _read(SAMPLE_DATA_DIR, MATCH_SUMMARY_CSV)

    d = diff[(diff["MINUTE"] == minute) & (diff["TEAM"] == team.upper())]
    m = match[match["GAME_DURATION"] >= min_game_duration]

    return (d
        .merge(m, on="MATCH_ID", how="inner")
        [["MATCH_ID", "CHAMPION_ROLE", "GOLD_DIFF", "WINNING_TEAM", "AVERAGE_RANK", "GAME_DATE", "GAME_DURATION"]]
        .reset_index(drop=True)
    )

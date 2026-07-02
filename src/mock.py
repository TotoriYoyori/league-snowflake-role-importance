# mock.py reads data for Local Mode: CSVs are a 500-match random sample
# covering minutes 5-30, both teams, all 5 lanes.

import os

import pandas as pd

# --------------- FILE NAME CONSTANTS ---------------
DIFF_INTERVAL_CSV = "s500_diff_interval_state.csv"
MATCH_SUMMARY_CSV = "s500_match_team_stats_summary.csv"


# --------------- READ ASSETS/SAMPLE_DATA MOCK CSV ---------------
def diff_interval_by_match(
    sample_data_dir: str,
    minute: int = 15,
    team: str = "Blue",
    min_game_duration: int = 300,
) -> pd.DataFrame:
    """
    Mirrors query.py::DiffIntervalByMatch.build() + its execution, but reads
    from the two vendored CSVs and filters/joins in pandas instead of SQL.
    """
    diff = pd.read_csv(os.path.join(sample_data_dir, DIFF_INTERVAL_CSV))
    match = pd.read_csv(os.path.join(sample_data_dir, MATCH_SUMMARY_CSV))

    d = diff[(diff["MINUTE"] == minute) & (diff["TEAM"] == team)]
    m = match[match["GAME_DURATION"] >= min_game_duration]

    return (d
        .merge(m, on="MATCH_ID",how="inner")
        [["MATCH_ID", "LANE", "GOLD_DIFF", "WINNING_TEAM", "AVERAGE_RANK", "GAME_DATE", "GAME_DURATION"]]
        .reset_index(drop=True)
    )

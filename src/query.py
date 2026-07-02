from pydantic import BaseModel, ConfigDict

# --------------- QUERY DIRECTLY FROM SNOWFLAKE DATABASE ---------------
class DiffIntervalByMatch(BaseModel):
    """
    Per-lane gold diff at a fixed match minute, joined to match outcome.
    """

    model_config = ConfigDict(frozen=True)

    database: str
    schema_: str
    minute: int = 15
    team: str = "Blue"
    min_game_duration: int = 300

    def build(self) -> str:
        return f"""
            SELECT
                D.MATCH_ID,
                D.LANE,
                D.GOLD_DIFF,
                M.WINNING_TEAM,
                M.AVERAGE_RANK,
                M.GAME_DATE,
                M.GAME_DURATION
            FROM {self.database}.{self.schema_}.DIFF_INTERVAL_STATE AS D
            JOIN {self.database}.{self.schema_}.MATCH_TEAM_STATS_SUMMARY AS M
                ON D.MATCH_ID = M.MATCH_ID
            WHERE D.MINUTE = {self.minute}
              AND D.TEAM = '{self.team}'
              AND M.GAME_DURATION >= {self.min_game_duration}
        """

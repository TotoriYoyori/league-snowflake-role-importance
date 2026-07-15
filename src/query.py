from pydantic import BaseModel, ConfigDict, field_validator

from settings import settings


# --------------- QUERY DIRECTLY FROM SNOWFLAKE DATABASE ---------------
class DiffIntervalByMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    minute: int = 15
    team: str = "Blue"
    min_game_duration: int = 300

    @field_validator("team")
    @classmethod
    def _blue_or_red_team(cls, v: str) -> str:
        allowed = settings.team_options
        if v not in allowed:
            raise ValueError(f"team must be one of {allowed}, got {v!r}")

        return v

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
            FROM LEAGUE_RECORDS.GOLD.DIFF_INTERVAL_STATE AS D
            JOIN LEAGUE_RECORDS.GOLD.MATCH_TEAM_STATS_SUMMARY AS M
                ON D.MATCH_ID = M.MATCH_ID
            WHERE D.MINUTE = {self.minute}
              AND D.TEAM = '{self.team}'
              AND M.GAME_DURATION >= {self.min_game_duration}
        """

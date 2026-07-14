Role Importance
===============

> **Live demo:** [league-sf-role-importance.streamlit.app](https://league-sf-role-importance.streamlit.app/) 

> **Parent pipeline:** [github.com/TotoriYoyori/league-snowflake](https://github.com/TotoriYoyori/league-snowflake)

A logistic regression model that turns each lane's gold diff at a chosen match minute into a win
coefficient. Built for internal teams as audience. Same model
also powers a live predictor: type in 5 gold diffs, get a win probability back.

----
## How it's built:

- `src/query.py` (`DiffIntervalByMatch`): one frozen pydantic model with a `.build()`
  method, run only when deployed against Snowflake.
- `src/data.py`: owns all data procurement: caches every stage of the pipeline (raw fetch → pivot/scale
  → EDA → train/test fit → full-data fit → CV stability), reads mock CSVs locally (join done in pandas,
  mirroring `query.py`'s SQL), and wraps the live/mock fetch in a try/except so a failed load surfaces as
  a card-level error instead of crashing the app.
- `src/model/`: plain pandas/sklearn/statsmodels functions with no Streamlit or Snowflake dependency:
  pivoting, scaling, train/test split, the model fits, CV coefficient stability, and the odds-ratio
  transform. 
- `src/ui/`: renders the 4 tabs, one module per tab plus a shared `components.py` (header/card/pill
  chrome) and `theme.py` (color palette + CSS).
- `settings.py`: pydantic `Settings` built once at import. Validate fields (e.g. `n_splits > 1`) 
that users adjust from the sidebar.

----
## Project structure

```
LeagueSnowflakeRoleImportance/
├── streamlit_app.py     # entry point
├── settings.py          # validated config
├── src/
│   ├── query.py         # live Snowflake query
│   ├── data.py          # all data procurement for ui display, and caching
│   ├── model/           # pure ds prep/eda/evaluation/importance/predictor functions
│   └── ui/               # renders: theme (palette + CSS) and components (shared chrome)
├── assets/
│   └── sample_data/     # 1,000-match CSV sample, used whenever running locally
└── snowflake.yml         # deploy on Snowflake
```
----
## What can it answer?

| Question                                                               | Answered by |
|------------------------------------------------------------------------|-------------|
| Which lane's gold lead most decides the game right now?                | **Lane Importance** tab |
| Did last patch's nerf actually reduce a lane's impact?                 | **Lane Importance** tab |
| Is the model even any good at minute *X*?                              | **Model Evaluation** tab |
| We're up 500g mid, down 2000g top at minute 20, what's our win chance? | **Predictor** tab |
| What does the underlying data actually look like?                      | **EDA** tab |

Model, in short:

- **Features:** each lane's gold diff at a chosen minute (Top / Jungle / Middle / Bottom / Support), scaled
  to per-1,000g units (scale up small p, and also aid with general interpretation).
- **Target:** did the selected team win.
- Early-minute models tend to score lower on AUC than late-minute ones, there's just more game left to be  played. 
The Model Evaluation tab's pill flags when to trust a given minute's coefficients less.

----
## Data sources 
> See parent pipeline ([github.com/TotoriYoyori/league-snowflake](https://github.com/TotoriYoyori/league-snowflake)) 
for the schema of the source data.

| Source | Grain |
|--------|-------|
| `DIFF_INTERVAL_STATE` | 1 row per match, per lane, per minute |
| `MATCH_TEAM_STATS_SUMMARY` | 1 row per match |

Getting started
----------------

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Runs against the bundled 1,000-match CSV sample by default, this is also how the
[live demo](https://league-sf-role-importance.streamlit.app/) runs, since it has no Snowflake session to
detect. If you want to see the schema directly, the CSVs are under `assets/sample_data`.

This repo is attached as a subfolder under the parent [`league-snowflake`](https://github.com/TotoriYoyori/league-snowflake).
Running from the parent repo will automatically register this app to query live from the database.

Known limitations
------------------
- No persisted history: every session recomputes fresh off the same static data. Nothing snapshots a
  patch's coefficients for later comparison, so "patch over patch" tracking today means re-running this
  app on each patch's data and comparing manually (future additions)
- One fixed 70/30 train/test split for the Model Evaluation tab, not repeated or cross-validated (Lane
  Importance's CV stability check is separate and only covers the full-data fit).

Original source: [LoL Match Intervals: 2 Million In-Game Snapshots](https://www.kaggle.com/datasets/nathansmallcalder/league-of-legends-match-interval-snapshots-2026)

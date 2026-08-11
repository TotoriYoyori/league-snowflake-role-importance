# Role Importance

> **Live demo:** [league-sf-role-importance.streamlit.app](https://league-sf-role-importance.streamlit.app/) 

> **Parent pipeline:** [github.com/TotoriYoyori/league-snowflake](https://github.com/TotoriYoyori/league-snowflake)

A simple logistic regression Streamlit app. Observe how each lane's gold diff at a chosen match minute affect the total 
win probability for your LoL match. **Can run as a demo using sample data, or be connected to the parent pipeline
and use live, continuous data.**

----
## Project structure

```
LeagueSnowflakeRoleImportance/
├── streamlit_app.py     # entry point
├── settings.py          # validated config
├── src/
│   ├── query.py         # live Snowflake query
│   ├── data.py          # all data procurement for ui display, and caching
│   ├── mock.py          # local CSV mock data loading
│   ├── model/           # pure ds prep/eda/evaluation/importance/predictor functions
│   └── ui/              # renders: theme (palette + CSS) and components (shared chrome)
├── assets/
│   └── sample_data/     # CSV sample, used whenever running locally
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

- **Features:** each lane's gold diff at a chosen minute 
(`GOLD_DIFF_TOP` / `GOLD_DIFF_JUNGLE` / `GOLD_DIFF_MIDDLE` / `GOLD_DIFF_BOTTOM` / `GOLD_DIFF_SUPPORT`), scaled
to per-1,000g units (scale up small p, and also aid with general interpretation).
- **Target:** did the selected team win (0/1 on `{TEAM}_WIN`)
- Early-minute models tend to score lower on AUC than late-minute ones, there's just more game left to be  played. 
The Model Evaluation tab's pill flags when to trust a given minute's coefficients less.

----
## Known limitations

- No persisted history: every session recomputes fresh off the same static data. Nothing snapshots a
  patch's coefficients for later comparison, so "patch over patch" tracking today means re-running this
  app on each patch's data and comparing manually (future additions)
- One fixed 70/30 train/test split for the Model Evaluation tab, not repeated or cross-validated (Lane
  Importance's CV stability check is separate and only covers the full-data fit).

> Original source: [LoL Match Intervals: 2 Million In-Game Snapshots](https://www.kaggle.com/datasets/nathansmallcalder/league-of-legends-match-interval-snapshots-2026)

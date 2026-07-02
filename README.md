# Role Importance

## Does a lane quest actually make that lane matter more?

League of Legends' 2026 lane quest system aimed to increase individual lane identity and reward solo strength. 
My app checks whether that's true (using data instead of vibes!)

*How much does each lane's gold lead 
actually decide who wins?*

A logistic regression model turns lane gold diffs into win coefficients — one number per lane, 
comparable side by side, trackable patch over patch.

> **Info**: This app is a standalone deployment of a larger League of Legend ELT Snowflake Pipeline, visit this
> [link here](https://github.com/TotoriYoyori/league-snowflake) to see more ~

----
## What This App Can Do

* Pull per-lane gold diff at a chosen match minute.
* Fit a logistic regression: 5 lane gold diffs → probability of winning.
* Cross-validate the coefficients (10×5 repeated k-fold).
* Same model powers a live predictor: type in 5 gold diffs, get a win probability back.

> **Info:** Like the other apps in this pipeline, this one runs two ways from the same code —
> against the live warehouse inside Snowflake, or fully offline against a 1000-match CSV sample for demos.

----
## What can it answer?
I built this model specifically with the audience in mind being internal teams: data scientists, data engineers, 
balance teams, etc. That means all statistics and the steps to get there are shown. Less technical audiences such as 
PM and stakeholders can also view the results to a certain extent. 

| Question | Answered by |
|----------|-------------|
| Which lane's gold lead most decides the game right now? | **Lane Importance** tab |
| Did last patch's nerf actually reduce a lane's impact? | **Lane Importance** tab |
| Is the model even any good at minute *X*? | **Model Evaluation** tab |
| We're up 500g mid, down 2000g top at minute 20 — what's our win chance? | **Predictor** tab |
| What does the underlying data actually look like? | **EDA** tab |

----
## Model, in short

* **Feature:** each lane's gold diff at a chosen minute (Top / Jungle / Middle / Bottom / Support), 
scaled to per-1,000g units.
* **Target:** did the selected team win.

> Early-minute models score lower on AUC than late-minute ones — there's just more game left to be played. 
> The Model Evaluation tab's pill will tell you when to trust a given minute's coefficients less.

----
## Data sources
Local/demo mode runs against a 1,000-match random sample of both tables. 
If you want to see the schema, you can download and open the csv files under `assets/sample_data`.

For production version (Streamlit-in-Snowflake),
the following gold tables are sourced from my pipeline.

| Source | Grain |
|--------|-------|
| `DIFF_INTERVAL_STATE` | 1 row per match, per lane, per minute |
| `MATCH_TEAM_STATS_SUMMARY` | 1 row per match |
> This is [my ELT pipeline](https://github.com/TotoriYoyori/league-snowflake) to see the full works.
> 
----
Original source: [LoL Match Intervals: 2 Million In-Game Snapshots](https://www.kaggle.com/datasets/nathansmallcalder/league-of-legends-match-interval-snapshots-2026)

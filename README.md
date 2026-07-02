# Role Importance

## Does a lane quest actually make that lane matter more?

League of Legends' 2026 lane quest system was built to boost individual lane identity and reward solo strength. 
This app checks whether that's true, using data instead of vibes: 
**how much does each lane's gold lead 
actually decide who wins?**

A logistic regression model turns lane gold diffs into win coefficients — one number per lane, 
comparable side by side, trackable patch over patch.

> **Info**: This app is a standalone deployment of a larger League of Legend ELT Snowflake Pipeline, visit this
> [link here](https://github.com/TotoriYoyori/league-snowflake) to see more apps and features! 

----
## What This App Can Do

* Pull per-lane gold diff at a chosen match minute, joined to match outcome, from Gold.
* Fit a logistic regression: 5 lane gold diffs → probability of winning.
* Cross-validate the coefficients (10×5 repeated k-fold) for a stable, tracked-over-time number per lane.
* Same model powers a live predictor: type in 5 gold diffs, get a win probability back.

> **Info:** Like the other apps in this pipeline, this one runs two ways from the same code —
> against the live warehouse inside Snowflake, or fully offline against a 500-match CSV sample for demos.

----
## What can it answer?

| Question | Answered by |
|----------|-------------|
| Which lane's gold lead most decides the game right now? | **Lane Importance** tab |
| Did last patch's nerf actually reduce a lane's impact? | **Lane Importance** tab, compared run over run |
| Is the model even any good at minute *X*? | **Model Evaluation** tab |
| We're up 500g mid, down 2000g top at minute 20 — what's our win chance? | **Predictor** tab |
| What does the underlying data actually look like? | **EDA** tab |

----
## Model, in short

* **Feature:** each lane's gold diff at a chosen minute (Top / Jungle / Middle / Bottom / Support), 
scaled to per-1,000g units.
* **Target:** did the selected team win.
* **Held-out evaluation:** train/test split, reported on the **Model Evaluation** tab 
(AUC, confusion matrix, classification report, ROC).
* **Model of record:** once evaluation clears the bar, a single logistic regression is refit on 
*all* available matches — this is the model behind both **Lane Importance** and the **Predictor** tab.
* **Stability:** the tracked lane coefficients are a mean ± 95% CI over repeated k-fold cross-validation 
on the full dataset, not a single fit — a coefficient bounces around less across patches 
when it's the average of 50 refits.

> Early-minute models score lower on AUC than late-minute ones — there's just more game left to be played. 
> The Model Evaluation tab's pill will tell you when to trust a given minute's coefficients less.

----
## Data sources

Same Gold-layer tables as the rest of the pipeline:

| Source | Grain |
|--------|-------|
| `DIFF_INTERVAL_STATE` | 1 row per match, per lane, per minute |
| `MATCH_TEAM_STATS_SUMMARY` | 1 row per match |

Local/demo mode runs against a 500-match random sample of both tables, covering minutes 5–30 — 
small enough to ship in the repo, large enough that the model's AUC and coefficient ranking 
hold up against the full ~40K-match dataset.

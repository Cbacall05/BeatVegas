# Beat Vegas

This is my personal quest to out-model sportsbook lines for NFL games. Everything in this repo points at one goal: **Beat Vegas** by pairing clean data engineering with calibrated models and tooling I actually use on gameday.

---

## What This Project Covers

- Ingest schedules, odds, play-by-play, injuries, and rest/travel context.
- Engineer rolling team form, matchup edges, and contextual features.
- Train a stable of moneyline and totals models (LightGBM, XGBoost, logistic, random forest, linear).
- Calibrate the classification models with isotonic regression so probabilities line up with market closer data.
- Surface everything in a Streamlit dashboard with tabs for matchups, spreads, totals, and player props.
- Automate nightly data refreshes and model backfills so I can re-train with one command.

If you only want to run things, jump to **Getting Started**. The rest of this README dives into why each piece exists and how it helps nudge me closer to beating the books.

---

## Getting Started (Windows PowerShell examples)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# quick demo pipeline
python demo_pipeline.py --seasons 2021 2022 --validation-season 2023 --max-games 200

# head-to-head CLI prediction
python matchup_predictor.py CHI DET --seasons 2018 2019 2020 2021 2022 --validation-season 2023

# interactive dashboard
streamlit run apps/matchup_dashboard.py

# notebook tour (optional)
jupyter notebook notebooks/beat_vegas_demo.ipynb
```

---

## Data Sources & Storage

- **Schedules, odds, weekly team stats**: `nfl_data_py` (nflfastR). Cached under `data/pbp` and `data/external` so I can work offline.
- **Rest & travel deltas**: computed in `beat_vegas.schedule_enrichment` from kickoff dates and stadium coordinates. Stored at `data/external/features/rest_travel.parquet` for reuse.
- **Injury notes**: `nfl_data_py` plus manual overrides in `configs/manual_injuries.csv`. I can hand-tune questionable players or QB changes.
- **Player props inputs**: play-by-play usage, team totals, and roster lookups to build touchdown and passing-yard probabilities.

Nightly automation (`scripts/update_external_data.py`) keeps the caches warm, pulling updated seasons and recomputing rest/travel metrics.

---

## How Predictions Are Built

1. **Ingestion** (`beat_vegas.data_load`)
  - Validate requested seasons, pull schedules/odds/injuries, and attach rest/travel features on the way in.
  - Optional caching keeps downloads to a minimum.

2. **Feature Engineering** (`beat_vegas.features`)
  - Rolling averages for scoring, EPA, success rate, turnover margin, market spreads, and win rate.
  - Diff features compare home vs. away form, rest advantage, and travel mileage.

3. **Model Training** (`beat_vegas.models`)
  - Moneyline models: LightGBM, XGBoost, logistic regression, random forest.
  - Totals models: LightGBM, XGBoost, linear regression, random forest.
  - Classification metrics captured (log-loss, Brier, accuracy) plus isotonic calibration to make the win probabilities trustworthy.
  - Regression models record RMSE/MAE and a residual standard deviation to estimate total-point variance.

4. **Calibration Bias Reporting**
  - `pipeline.compute_bias_reports` bins model probabilities against market implied odds to see which teams/buckets drift. Bias summaries help decide when to trust or fade the models.

5. **Prediction Surfaces**
  - `matchup_predictor.py` creates ensemble projections for upcoming games, including raw vs. calibrated win rates, injury penalties, and total distributions.
  - `apps/matchup_dashboard.py` packages everything into tabs with charts, filters, and a debug view. Injury adjustments always start from raw model outputs to avoid double-calibration.

---

## Current Tooling Tour

- **Matchup Studio (Streamlit)**
  - Train models for any season window, preview upcoming games, and inspect ensemble edges.
  - Rest/travel chips highlight short weeks, cross-country travel, and net rest advantage.
  - Player props tab builds touchdown and passing-yard ladders from play-by-play usage and market totals.

- **Command Line Helpers**
  - `demo_pipeline.py`: quick end-to-end run with limited games for smoke testing.
  - `matchup_predictor.py`: CLI predictions for any home/away pairing. Saves CSVs if needed.
  - `scripts/backfill_ensemble.py`: full backfill that trains, calibrates, and writes a JSON summary of metrics and bias tables.
  - `scripts/update_external_data.py`: nightly refresh that downloads nflfastR artifacts and recomputes rest/travel parquet files.

- **Tests**
  - `tests/smoke_test.py` runs a synthetic pass through the model stack and asserts calibrated columns exist. It is intentionally lightweight, just guarding against accidental regressions.

---

## 2025 Week 7 Scorecard

Latest real-world checkpoint: Week 7 of the 2025 NFL season. I graded the model directly against final scores and closing market numbers.

| Metric | Model | Typical Vegas Benchmark | Notes |
| --- | --- | --- | --- |
| Straight-up winners | 71 % (10/14) | ~63 % | Clear edge versus Pick'em consensus. |
| Against the spread | 78.6 % (11/14) | ~52–55 % | Monster week; comfortably beat sharp averages. |
| Totals (O/U) | 78.6 % (11/14) | ~51–54 % | Nailed game scripts on both sides of the ball. |
| Mean absolute error (total points) | ~5.9 pts | 7–8 pts | Tighter than the market on expected scoring. |

**Highlights**
- Went 10-for-14 on outright winners and 11-for-14 on both spreads and totals.
- Calibration plots showed projected margins tracking actual outcomes with minimal drift.
- Misses came from chaotic blowouts (DAL-WAS, IND-LAC) rather than systematic bias.

**What it means**
- Ensemble spread/total differentials are pulling real signal, not just overfitting past seasons.
- Even holding 55 % ATS and 53 % totals would be profitable long-term; brushing ~79 % in a week shows the upside when the reads line up.

I'll keep logging weekly scorecards here so it is obvious when the edge is real and when it needs more tuning.

---

## Why The Models Look The Way They Do

- Multiple algorithms keep the ensemble from being too spiky. If LightGBM or XGBoost is missing (local env issue), logistic and random forest still run.
- Calibrated probabilities matter more than raw accuracy when you are comparing to sportsbook numbers. I use isotonic regression because it is monotonic and handles the weird shapes these probabilities can take.
- Rest and travel metrics (rest days, short weeks, travel buckets) stem from the idea that the line might not fully price fatigue. These features flow into both training and dashboard visuals.
- Injuries are messy, so I combine automated reports with my own CSV overrides. The dashboard shows both penalties and alerts so I can double-check before trusting a projection.

---

## Roadmap / Next Experiments

- Sharpen injury modeling: merge depth charts, practice reports, and betting splits to auto-scale penalties instead of hand tuning.
- Add walk-forward backtests that simulate betting strategies against historical closing lines to quantify bankroll impact.
- Expand player prop coverage (receiving yards, rushing attempts) once the touchdown and QB passing pipelines settle.
- Bring in live odds APIs for real-time syncing and alerts when the model edge crosses a threshold.
- Build a historical dashboard view so I can review week-over-week calibration drift and edge decay.

If you have ideas or spot gaps, open an issue or ping me. This is very much a living project.

---

## Troubleshooting Notes To Myself

- Stick with Python 3.12 for now; 3.13 still lacks wheels for some dependencies.
- If `lightgbm` or `xgboost` fail to import, the pipeline logs a warning and the ensemble downgrades gracefully.
- When Streamlit acts up, run `streamlit cache clear` or use the sidebar "Clear cached data and rerun" button (now powered by `st.rerun`).
- Cached datasets live under `data/`; delete them if inputs change drastically.

Happy modeling, and here’s to beating Vegas more often than not.
# Project Beat Vegas

Beat Vegas is a streamlined NFL matchup studio that learns from historical schedule data, projects upcoming games, and surfaces win probabilities, totals, and validation context through both CLI scripts and a sleek Streamlit dashboard.

## Quickstart

1. **Create a Python 3.12 virtual environment and install dependencies**

  ```powershell
  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```

2. **Run the baseline demo**

  ```powershell
  python demo_pipeline.py --seasons 2021 2022 --validation-season 2023 --max-games 200
  ```

   The script downloads schedules, weekly team metrics, and play-by-play data using `nfl_data_py`, engineers rolling features, trains LightGBM/Logistic/RandomForest baselines, and surfaces potential edges against sportsbook totals and moneylines.

3. **Get a quick head-to-head projection**

   ```powershell
   python matchup_predictor.py CHI DET --seasons 2018 2019 2020 2021 2022 --validation-season 2023
   ```

   Replace the abbreviations with the teams you care about. The script reports historical validation results (if those teams met in the validation season) and a hypothetical matchup projection based on the latest rolling averages for each team.

4. **Launch the Matchup Studio (interactive UI)**

  ```powershell
  streamlit run apps/matchup_dashboard.py
  ```

  This opens a new browser tab with a sleek control panel to train models, inspect upcoming matchups, visualize win probabilities, drill into total-point distributions, and dive into the Player Props lab for touchdown probabilities. Adjust seasons, rolling windows, or highlight specific teams without touching the CLI.

5. **Explore the notebook**

   Open `notebooks/beat_vegas_demo.ipynb` to step through the workflow interactively, generate distribution plots, and tweak features or model parameters.

## Feature Highlights

- **Historical training pipeline** – pulls schedules and odds from `nfl_data_py`, engineers rolling team strength features, and trains logistic/forest/LightGBM baselines for moneylines and totals.
- **Matchup predictor CLI** – targets any home/away pairing, prints validation season context, and produces forward-looking win and total projections.
- **Streamlit Matchup Studio** – interactive dashboard for selecting training seasons, tweaking rolling windows, spotlighting teams, and reviewing ensemble win probabilities in real time.
- **Total distribution explorer** – ensemble mean/variance estimates for totals with per-model breakdowns, market edge deltas, and quick density plots for each matchup.
- **Player touchdown lab** – per-game scorer probabilities built from recent usage, red-zone touches, market totals, and filtered against active team rosters so only current players surface.
- **Play-by-play cache ready** – parquet library under `data/pbp` makes it easy to plug in richer PBP-derived features when needed.

## Player Props Lab

- Jump into the **Player Props** tab inside the Streamlit dashboard to surface touchdown probabilities for every upcoming matchup.
- Select a game to compare both teams’ top scorers, inspect their recent usage (touches, targets, red-zone looks), and review a probability leaderboard.
- Probabilities come from a logistic model built on play-by-play usage trends blended with the market total and validated against active roster data, so only current contributors in rich scoring environments rise to the top.

## Architecture

```
beat_vegas/
  data_load.py    # Data ingestion, caching, schedule enrichment
  features.py     # Rolling/team strength features, game-level pivoting
  models.py       # Baseline models, metrics, mispricing detection
  player_models.py# Player-level touchdown dataset + logistic ensemble
  visuals.py      # ND blue + gold plotting helpers
  pipeline.py     # High-level orchestration helpers
configs/
  pipeline.yml    # Default seasons, feature toggles, thresholds
```

## Key Components

- **Data ingestion (`beat_vegas.data_load`)**
  - Validates season ranges, caches play-by-play parquet files, and harmonizes weekly team stats with schedule context (home/away, market odds).
- **Feature engineering (`beat_vegas.features`)**
  - Rolling averages for scoring, EPA, success rate, turnover margin, matchup differentials, and a composite team strength index.
- **Modeling (`beat_vegas.models`)**
  - LightGBM (preferred) plus logistic/linear + random forest baselines.
  - Classification metrics: Log-loss, Brier score, accuracy.
  - Regression metrics: RMSE, MAE.
  - Mispricing detection against market-implied probabilities and totals.
- **Visualization (`beat_vegas.visuals`)**
  - Win-probability histograms, total-point KDEs, model vs. market scatter, calibration curves, and error distributions styled with Notre Dame colors.

## Configuration

Edit `configs/pipeline.yml` to adjust seasons, feature flags, mispricing thresholds, or cache paths. You can also override these values via CLI arguments in `demo_pipeline.py`.

## Roadmap

- Integrate injury and depth-chart signals so model strength adjusts when key players are unavailable.
- Blend short- and long-term form (dual rolling windows, exponential decay) to stabilize early-season projections.
- Calibrate probabilities against market closing lines and add edge detection overlays.
- Automate odds ingestion (sportsbook APIs, consensus feeds) and persist forecasts to a database for daily tracking.
- Expand Streamlit dashboard with scenario sliders, CSV export, and historical backtesting views.

## Troubleshooting

- **Missing dependencies**: Ensure `lightgbm` and `nfl-data-py` are installed. The code will log warnings and gracefully skip LightGBM models if unavailable.
- **Python version**: Stick with Python 3.12. Newer releases (e.g., 3.13) currently lack prebuilt wheels for the NumPy versions that `nfl-data-py` depends on, which will cause installation failures on Windows.
- **Network limits**: If API calls fail, rerun the demo with a smaller season set or rely on cached parquet files in `data/pbp`.
- **Performance**: Use `--max-games` to limit the training dataset while prototyping.

> Tip: Reactivate the environment with `.\.venv\Scripts\Activate.ps1` before running any of the commands above.
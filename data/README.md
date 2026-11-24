# Competition Data Directory (`data/`)

Raw NFL Scouting Combine and Next Gen Stats tracking files are proprietary to the **NFL Big Data Bowl 2027** competition and are excluded from version control per competition rules.

## Expected Raw Files

Download the competition archive from the official Kaggle NFL Big Data Bowl 2027 Data tab and place the 9 CSV files in this `data/` directory if running the full pipeline from scratch (`python src/run_pipeline.py --from-raw`):

1. `players.csv` — Prospect metadata & NFL Draft selection info (2023–2025)
2. `combine_results.csv` — Official Combine measurements & stopwatch drills
3. `combine_tracking.csv` — 10 Hz optical tracking across Combine drills
4. `games.csv` — NFL game schedule & season metadata (`REG`, `PRE`, `POST`)
5. `player_play.csv` — Play-level pass-rush outcomes (`quick_pressure`, `time_to_pressure`, `player_get_off`, `lined_up_position`)
6. `player_career_successes.csv` — Career honors & milestone summaries
7. `game_tracking_2023.csv` — 10 Hz NFL game tracking (2023 season)
8. `game_tracking_2024.csv` — 10 Hz NFL game tracking (2024 season)
9. `game_tracking_2025.csv` — 10 Hz NFL game tracking (2025 season)

> **Note:** To inspect or reproduce all regression models, cross-validation benchmarks, cohort summaries, and the Kaggle submission notebook *without* downloading ~2.3 GB of raw tracking CSVs, use the pre-computed `processed/model_results.json` included in this repository (`python src/run_pipeline.py`).

# Capacity Planning Under Demand Uncertainty

Empirical study using 2024 New York City Yellow Taxi demand.

## Recovered implementation

This repository contains the recovered full-year research implementation from the `research-full-year-capacity` branch of the original research workspace. The implementation downloads the 12 monthly 2024 NYC TLC Yellow Taxi trip files, aggregates pickup demand hourly by pickup zone, selects the 20 highest-volume zones using the training period, evaluates Seasonal Naive, Ridge, Random Forest and XGBoost forecasts, and translates forecasts into capacity decisions under asymmetric shortage/excess costs.

The experiment uses a chronological 60% / 20% / 20% train-validation-test split. Capacity buffers tested are 0%, 10%, 20% and 30%, with shortage-to-excess cost ratios of 1:1, 3:1, 5:1 and 10:1.

## Run

```bash
pip install -r requirements.txt
python research_capacity/run_full_year.py
```

Outputs are written to `research_capacity/results/`.

## Important

The raw TLC trip data are intentionally not committed to this repository because the monthly parquet files are large. The script reads the official TLC-hosted files directly through DuckDB HTTPFS.

The executable research code is the recovered implementation; this repository is a standalone home for it and does not modify the original ML application repository.

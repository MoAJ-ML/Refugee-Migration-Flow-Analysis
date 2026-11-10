# UNHCR Displacement Stock Analysis

This educational project describes the 2000–2024 global aggregate in `download/persons_of_concern.csv` and explores statistical scenarios for 2025–2034. The data is attributed to the [UNHCR Refugee Data Finder](https://www.unhcr.org/refugee-statistics/) and governed by the included `download/UNHCR_RefugeeDataFinder_Copyright.pdf`. Record the export date and scope when obtaining a newer file.

The columns `Refugees` and `IDPs` describe populations at a point in time. Their annual differences are changes in recorded stock, **not** arrivals, returns, or migration flows. This script no longer labels those differences as net flows.

## Run and validate

```bash
pip install -r requirements.txt
MPLBACKEND=Agg python refugee_migration_analysis.py
python -m unittest discover -s tests
```

The script trains ARIMA on 2000–2019 and evaluates 2020–2024, then runs expanding-window one-year-ahead comparisons against a no-change naive baseline. After validation, it refits ARIMA on all available years before generating dated scenarios. Prophet is optional; its absence is reported and no Prophet metric is produced. Outputs include `rolling_origin_refugees.csv`, `rolling_origin_poc.csv`, `model_performance_metrics.csv`, `refugee_forecasts_2025_2034.csv`, and plots.

## Interpretation limits

These are statistical scenarios based on 25 annual global observations. Conflict, climate, policy, reporting changes and regional heterogeneity can dominate extrapolation. Holdout errors and a five-year rolling-origin sample do not establish a reliable directional policy forecast. Do not use the model to allocate services or predict individual movement. The old README decline percentages were historical; the generated CSVs were rerun on 10 November 2026 against input SHA-256 `8427b23e63757e70beb1467b3c0044048a1642dbd9eb8fe1cb1e3d0c39e117ff`.

For 2020–2024 expanding-window one-year predictions, refugee-stock MAE was 2.519 million (ARIMA) versus 2.380 million (no-change naive); total-persons-of-concern MAE was 11.965 million versus 13.726 million. The refugee baseline wins on this small sample, another reason not to present a directional policy conclusion.

No software license file is included. Suggested repository description: “Exploratory UNHCR displacement-stock analysis with naive and rolling-origin forecast checks.”

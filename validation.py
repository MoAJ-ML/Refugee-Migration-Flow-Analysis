"""Expanding-window, one-year-ahead comparison with a no-change baseline."""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def rolling_origin_forecasts(series, order, first_test_position=20):
    series = pd.Series(series).astype(float).sort_index()
    if len(series) <= first_test_position or first_test_position < 5:
        raise ValueError('Need at least five training years and one test year')
    rows = []
    for position in range(first_test_position, len(series)):
        train = series.iloc[:position]
        fitted = ARIMA(train, order=order).fit()
        rows.append({
            'Year': int(series.index[position].year),
            'Actual': float(series.iloc[position]),
            'ARIMA': float(np.asarray(fitted.forecast(1))[0]),
            'Naive': float(train.iloc[-1]),
        })
    return pd.DataFrame(rows)


def mean_absolute_errors(forecasts):
    return {name: float(np.mean(np.abs(forecasts['Actual'] - forecasts[name])))
            for name in ('ARIMA', 'Naive')}

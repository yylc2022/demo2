# models/arima.py
import numpy as np
import pandas as pd
import statsmodels.api as sm


def sliding_arima_forecast_writeback(
    df,
    price_col,
    window_size=100,
    step_size=1,
    p=1, d=0, q=1
):
    df = df.reset_index(drop=True).copy()

    price = df[price_col]
    logret = np.log(price).diff()

    df["arima_pred_logret"] = np.nan
    df["arima_true_logret"] = np.nan
    df["arima_pred_price"] = np.nan

    N = len(logret)
    if N <= window_size:
        raise ValueError("数据长度不足 window_size")

    i = 0
    while i + window_size < N:
        train = logret.iloc[i + 1: i + window_size + 1].dropna()

        try:
            model = sm.tsa.ARIMA(train, order=(p, d, q))
            fitted = model.fit()
            pred = fitted.forecast(steps=1).iloc[0]
        except Exception:
            pred = 0.0

        idx = i + window_size + 1
        if idx >= len(df):
            break

        prev_price = price.iloc[idx - 1]
        df.loc[idx, "arima_pred_logret"] = pred
        df.loc[idx, "arima_true_logret"] = logret.iloc[idx]
        df.loc[idx, "arima_pred_price"] = prev_price * np.exp(pred)

        i += step_size

    return df

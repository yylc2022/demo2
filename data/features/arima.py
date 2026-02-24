import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# =====================================================
# 滑动 ARIMA 预测（log return）
# 预测结果直接写回原始 DataFrame
# =====================================================
def sliding_arima_forecast_writeback(
    df,
    price_col,
    volume_col=None,
    window_size=200,
    step_size=1,
    p=1, d=0, q=1
):
    df = df.reset_index(drop=True).copy()

    price = df[price_col]
    logret = np.log(price).diff()

    # 预分配结果列（因果：前 window_size 行为 NaN）
    df["arima_pred_logret"] = np.nan
    df["arima_true_logret"] = np.nan
    df["arima_pred_price"] = np.nan

    N = len(logret)

    if N <= window_size:
        raise ValueError("数据长度不足 window_size")

    i = 0
    while i + window_size < N:
        train = logret.iloc[i + 1 : i + window_size + 1].dropna()

        try:
            model = sm.tsa.ARIMA(train, order=(p, d, q))
            fitted = model.fit()
            pred = fitted.forecast(steps=1).iloc[0]
        except Exception:
            pred = 0.0

        idx = i + window_size + 1  # 原始 df 中对应行

        if idx >= len(df):
            break

        prev_price = price.iloc[idx - 1]

        # 写回原表
        df.loc[idx, "arima_pred_logret"] = pred
        df.loc[idx, "arima_true_logret"] = logret.iloc[idx]
        df.loc[idx, "arima_pred_price"] = prev_price * np.exp(pred)

        i += step_size

    return df


# =====================================================
# 使用示例
# =====================================================
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    data_path = ROOT  / "raw" / "000300.SH-行情统计-20251203.xlsx"
    df_raw = pd.read_excel(data_path)

    df_out = sliding_arima_forecast_writeback(
        df=df_raw,
        price_col="收盘价",
        volume_col="成交量(万股)",
        window_size=50,
        step_size=1,
        p=1, d=0, q=1
    )

    df_out.to_excel(ROOT/"processed"/"000300_with_arima.xlsx", index=False)
    print(df_out.tail(5))

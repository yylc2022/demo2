# scripts/build_features.py
import pandas as pd

from models.arima import sliding_arima_forecast_writeback
from data.features.feature_engineering import generate_features
from pathlib import Path

def main():
    # 1️⃣ 读原始数据

    ROOT = Path(__file__).resolve().parents[1]
    df_raw = pd.read_excel(
        ROOT/"data/raw/000300.SH-行情统计-20251203.xlsx"
    )

    # 2️⃣ ARIMA 预测
    df_with_arima = sliding_arima_forecast_writeback(
        df=df_raw,
        price_col="收盘价",
        window_size=50,
        step_size=1,
        p=1, d=0, q=1
    )

    # 可选保存中间结果
    df_with_arima.to_csv(
       ROOT/ "data/processed/000300_with_arima.csv",
        index=False
    )

    # 3️⃣ 特征工程
    df_feat = generate_features(df_with_arima)

    df_feat.to_csv(ROOT/"data/processed/feat.csv", index=False)
    print("✅ Feature build finished")


if __name__ == "__main__":
    main()

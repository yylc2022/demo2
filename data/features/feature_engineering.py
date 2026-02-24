# features/feature_engineering.py
import pandas as pd
import numpy as np


def rolling_zscore(series, window=60):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / (std + 1e-6)


def generate_features(df):
    df = df.sort_values('交易日期').reset_index(drop=True)

    # ======================
    # Step 0: 基础收益
    # ======================
    df['logret'] = np.log(df['收盘价'] / df['收盘价'].shift(1))
    logret = df['logret']

    # ======================
    # 1. 价格「方向 + 节奏」因子
    # ======================
    for i in range(1, 16):
        df[f'logret_{i}'] = logret.shift(i - 1)

    # ======================
    # 2. 成交量（先 log）
    # ======================
    for i in range(1, 16):
        df[f'vol_{i}'] = np.log(1 + df['成交量(万股)'].shift(i - 1))

    # ======================
    # 3. 多尺度收益
    # ======================
    ret_1 = logret
    ret_20 = ret_1.rolling(20).sum()
    ret_60 = ret_1.rolling(60).sum()
    ret_120 = ret_1.rolling(120).sum()
    ret_acc = ret_20 - ret_60

    # ======================
    # 4. 波动率因子
    # ======================
    vol_5 = logret.rolling(5).std()
    vol_10 = logret.rolling(10).std()
    vol_20 = logret.rolling(20).std()
    vol_ratio = vol_5 / (vol_10 + 1e-6)

    # ======================
    # 5. K 线结构因子
    # ======================
    close = df['收盘价']
    open_ = df['开盘点位']
    high = df['最高点位']
    low = df['最低点位']

    body = (close - open_) / close
    upper = (high - close) / close
    lower = (open_ - low) / close

    # ======================
    # 6. 成交量配合
    # ======================
    volume = np.log(1+df['成交量(万股)'])
    vol_z = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()

    # ======================
    # 7. 相对位置
    # ======================
    sma20 = close.rolling(20).mean()
    pos_20 = (close - sma20) / sma20

    # ======================
    # 8. 二阶量价因子
    # ======================

    ret_prod_1=logret*logret.shift(1)
    W=30
    ret_prod_W=logret*logret.rolling(W).sum()

    vol_prod_1=vol_z*vol_z.shift(1)
    vol_prod_W=vol_z*vol_z.rolling(W).sum()

    # # ======================
    # # ARIMA 因子（直接读取）
    # # ======================
    # arima_pred = df['arima_pred_logret']
    # arima_err = df['arima_true_logret'] - df['arima_pred_logret']

    # ======================
    # Step 1: 构造标签
    # ======================
    H =1
    future_ret_H = logret.rolling(H).sum().shift(-H)
    label = (future_ret_H > 0).astype(int)

    # ======================
    # Step 2: 组织特征
    # ======================
    df_feat = pd.DataFrame({
        '交易日期': df['交易日期'],

        # logret
        'logret1': df['logret_1'],
        'logret2': df['logret_2'],
        'logret3': df['logret_3'],
        'logret4': df['logret_4'],
        'logret5': df['logret_5'],
        'logret6': df['logret_6'],
        'logret7': df['logret_7'],
        'logret8': df['logret_8'],
        'logret9': df['logret_9'],
        'logret10': df['logret_10'],
        'logret11': df['logret_11'],
        'logret12': df['logret_12'],
        'logret13': df['logret_13'],
        'logret14': df['logret_14'],

        # volume (log)
        'volume1': df['vol_1'],
        'volume2': df['vol_2'],
        'volume3': df['vol_3'],
        'volume4': df['vol_4'],
        'volume5': df['vol_5'],
        'volume6': df['vol_6'],
        'volume7': df['vol_7'],
        'volume8': df['vol_8'],
        'volume9': df['vol_9'],
        'volume10': df['vol_10'],
        'volume11': df['vol_11'],
        'volume12': df['vol_12'],
        'volume13': df['vol_13'],
        'volume14': df['vol_14'],

        # return scale
        'ret_1': ret_1,
        'ret_20': ret_20,
        'ret_60': ret_60,
        'ret_120': ret_120,
        'ret_acc': ret_acc,

        # volatility
        'vol_5': vol_5,
        'vol_10': vol_10,
        'vol_20': vol_20,
        'vol_ratio': vol_ratio,

        # kline
        'body': body,
        'upper': upper,
        'lower': lower,

        # volume + position
        'vol_z': vol_z,
        'pos_20': pos_20,

        # 二阶量价因子
        'ret_prod_1': ret_prod_1,
        'ret_prod_W': ret_prod_W,
        'vol_prod_1': vol_prod_1,
        'vol_prod_W': vol_prod_W,

        #OCHL 因子
        'open': df['开盘点位'],
        'close': df['收盘价'],
        'high': df['最高点位'],
        'low': df['最低点位'],

    })

    df_feat['label'] = label
    df_feat['future_ret_H'] = future_ret_H
    df_feat['logret'] = logret  # 仅用于分析，不喂模型

    # ======================
    # Step 2.5: rolling z-score 归一化
    # ======================
    zscore_window = 50

    zscore_cols = [
        'logret1',
        'logret2','logret3','logret4','logret5','logret6','logret7',
        'logret8','logret9','logret10','logret11','logret12','logret13','logret14',

        'volume1','volume2','volume3','volume4','volume5',
        'volume6','volume7','volume8','volume9','volume10','volume11',
        'volume12','volume13','volume14',

        'ret_1','ret_20','ret_60','ret_120','ret_acc',
        'vol_5','vol_10','vol_20','vol_ratio',
        'body','upper','lower',
        'pos_20',
        'ret_prod_1',
        'ret_prod_W',
        'vol_prod_1',
        'vol_prod_W'
    ]

    for col in zscore_cols:
        df_feat[col] = rolling_zscore(df_feat[col], zscore_window)

    # ======================
    # Step 3: 丢弃 NaN
    # ======================
    # 在 generate_features 内部修改：
    # df_feat = df_feat.dropna().reset_index(drop=True) # 删掉这行
    df_feat = df_feat.dropna(subset=zscore_cols).reset_index(drop=True)  # 只根据特征列删空值

    # 可选保存
    df_feat.to_csv('feat1.csv', index=False)

    return df_feat

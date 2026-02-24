import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys

# 基础路径配置
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.tcn import TCN
from data.features.feature_engineering import generate_features

# --- 配置参数 (必须与训练代码严格一致) ---
LOOKBACK = 50
FEATURE_COLS = [
    'logret1', 'logret2', 'logret3', 'logret4', 'logret5', 'logret6', 'logret7', 'logret8', 'logret9', 'logret10',
    'logret11', 'logret12', 'logret13', 'logret14',
    'volume1', 'volume2', 'volume3', 'volume4', 'volume5', 'volume6', 'volume7', 'volume8', 'volume9', 'volume10',
    'volume11', 'volume12', 'volume13', 'volume14',
    'ret_1', 'ret_20', 'ret_60', 'ret_120', 'ret_acc',
    'vol_5', 'vol_10', 'vol_20', 'vol_ratio',
    'body', 'upper', 'lower',
    'vol_z', 'pos_20'
]
MODEL_PATH = ROOT / "outputs" / "models" / "tcn_model2.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    model = TCN(len(FEATURE_COLS), num_channels=[16, 16], kernel_size=2, dropout=0.2).to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def run_historical_inference(file_path, model):
    # 1. 读取并生成特征
    df_raw = pd.read_excel(file_path) if file_path.suffix == '.xlsx' else pd.read_csv(file_path)
    df_feat = generate_features(df_raw)

    # 2. 准备原始预测容器
    # 注意：这里我们先计算出原始的预测序列
    raw_preds = np.full(len(df_feat), np.nan)

    # 3. 构造滑动窗口张量 (高效写法)
    feature_matrix = df_feat[FEATURE_COLS].values

    # windows[0] 是 [0:50] 的特征，模型预测的是索引为 50 的收益
    windows = [feature_matrix[i - LOOKBACK: i] for i in range(LOOKBACK, len(df_feat))]
    x_batch = torch.tensor(np.array(windows), dtype=torch.float32).to(DEVICE)

    # 4. 批量推理
    all_preds = []
    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(x_batch), batch_size):
            batch = x_batch[i: i + batch_size]
            out = model(batch).cpu().numpy().flatten()
            all_preds.extend(out)

    # 5. 对齐逻辑修改：
    # 原始预测值 all_preds[0] 是在第 49 天收盘后做出的，针对第 50 天的预测。
    # 我们先将其填入第 49 天 (t)
    raw_preds[LOOKBACK - 1: len(df_feat) - 1] = all_preds

    # 【关键步】将预测值整体向后移动一行，使 t 天做出的预测显示在 t+1 天那一行
    df_feat['model_prediction'] = pd.Series(raw_preds).shift(1).values

    # 6. 计算信号和策略收益
    # 此时 model_prediction 和 logret 已经在同一行对齐（都是针对当天的）
    df_feat['signal'] = np.sign(df_feat['model_prediction'])

    # 策略收益 = 当天开盘前的信号 * 当天的真实 logret
    df_feat['strategy_ret'] = df_feat['signal'] * df_feat['logret']

    return df_feat


# --- 执行推理 ---
if __name__ == "__main__":
    model = load_model()
    data_file = ROOT / "data" / "raw" / "000852.SH-行情统计-20251203.xlsx"

    result_df = run_historical_inference(data_file, model)

    # 保存结果
    output_path = ROOT / "outputs" / "historical_inference_full.csv"
    result_df.to_csv(output_path, index=False)

    print(f"✅ 历史全量推理完成！结果已保存至: {output_path}")
    print(result_df[['交易日期', 'model_prediction', 'logret', 'strategy_ret']].tail(10))
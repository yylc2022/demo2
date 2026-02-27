import pandas as pd
import numpy as np
import torch
from pathlib import Path
import sys
import os
import save_files_parquet
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
RAW_DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"


def load_model():
    """加载TCN模型"""
    model = TCN(len(FEATURE_COLS), num_channels=[16, 16], kernel_size=2, dropout=0.2).to(DEVICE)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)
        print(f"成功加载模型: {MODEL_PATH}")
    except FileNotFoundError:
        print(f"错误: 模型文件未找到 at {MODEL_PATH}")
        sys.exit(1)
    model.eval()
    return model


def run_inference_for_file(file_path, model):
    """对单个文件运行推理并返��带有预测的DataFrame"""
    try:
        # 处理 CSV 文件
        df_raw = pd.read_csv(file_path)

        # 解决 '交易日期' 既是 index 又是 column 的歧义问题
        # 如果 index 名字也是 '交易日期'，先重置索引
        if df_raw.index.name == '交易日期':
            df_raw = df_raw.reset_index(drop=True)
            
        # 确保数据中只有唯一的 '交易日期' 列
        # 如果列中有重复的 '交易日期'，我们只保留第一个
        if list(df_raw.columns).count('交易日期') > 1:
             # 找到所有名为 '交易日期' 的列的位置
             cols = list(df_raw.columns)
             first_idx = cols.index('交易日期')
             # 重命名其他重复列或只保留第一个
             df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]

        # 1. 生成特征 (此过程会排序、去头、并生成归一化后的 'ret_1' 等特征)
        df_feat = generate_features(df_raw)
        if 'gsid' in df_raw.columns:
            df_feat['gsid'] = df_raw['gsid']

        # 2. 计算原始未归一化的对数收益率 (用于 Ground Truth - 放在 CSV 'actual' 列)
        # 注意: generate_features 已经按时间排序并重置索引

        # 确保 '交易日期' 为 datetime 类型
        df_feat['交易日期'] = pd.to_datetime(df_feat['交易日期'])

    except Exception as e:
        print(f"特征生成失败 {file_path.name}: {e}")
        return None

    if len(df_feat) <= LOOKBACK:
        print(f"数据不足 {file_path.name}: 需要 > {LOOKBACK} 天, 只有 {len(df_feat)} 天。")
        return None

    feature_matrix = df_feat[FEATURE_COLS].values

    # 3. 准备滑��窗口
    # 目标：对 index = LOOKBACK 到 index = len(df_feat)-1 的数据进行推理（预测）
    # 对应的输入窗口是 [index-LOOKBACK : index] (即 t-lookback 到 t-1)
    # 预测结果对应的是 index 行的收益率

    valid_indices = range(LOOKBACK, len(df_feat))

    # windows list of (T, F)
    windows = [feature_matrix[i - LOOKBACK: i-1] for i in valid_indices]

    if not windows:
        return None

    # stack -> (B, T, F)
    x_batch = torch.tensor(np.array(windows), dtype=torch.float32).to(DEVICE)

    # TCN forward 期望 (B, T, F) 并会在内部转置为 (B, F, T)

    all_preds = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(x_batch), batch_size):
            batch = x_batch[i: i + batch_size]
            out = model(batch).cpu().numpy().flatten()
            all_preds.extend(out)

    # 4. 组装结果
    # 预测值 all_preds[k] 是基于 valid_indices[k]-LOOKBACK 到 valid_indices[k]-1 的数据
    # 它预测的是 valid_indices[k] 这一天的收益
    # 真实值即 df_feat['raw_logret'] 在 valid_indices[k] 的值

    results = pd.DataFrame({
        'date': df_feat['交易日期'].iloc[LOOKBACK:].values,
        'prediction': all_preds,
        'actual': df_feat['logret'].iloc[LOOKBACK:].values,
        'gsid': df_feat['gsid'].iloc[LOOKBACK:].values if 'gsid' in df_feat.columns else 0
    })

    return results


def main():
    """主程序：推理 raw 文件夹里的所有指数"""
    model = load_model()
    all_files = list(RAW_DATA_DIR.glob('*.csv'))

    if not all_files:
        print(f"在 {RAW_DATA_DIR} 中没有找到 .csv 文件")
        return

    # 用于存储最终 parquet 格式的 df
    parquet_dfs = []
    # 用于存储宽表格式的 df
    pred_dfs_wide = []
    actual_dfs_wide = []

    for file_path in all_files:
        variety = file_path.stem
        print(f"正在处理: {variety} ({file_path.name}) ...")
        sys.stdout.flush()

        res = run_inference_for_file(file_path, model)
        if res is not None:
            # --- 准备 Parquet 长表数据 ---
            parquet_df = pd.DataFrame({
                'gsid': res['gsid'].astype(np.int32),
                'gudt': (res['date'].dt.strftime('%Y%m%d').astype(np.int64)) * 1000000000 + 190000000,
                'tradeDate': res['date'].dt.strftime('%Y%m%d').astype(np.int32),
                'icode': variety,
                'predLogreturn': res['prediction'].astype(np.float64)
            })
            parquet_df['icode'] = parquet_df['icode'].astype('S40')
            parquet_dfs.append(parquet_df)

            # --- 准备原始宽表 CSV 数据 ---
            p_df_wide = res[['date', 'prediction']].rename(columns={'prediction': variety}).set_index('date')
            pred_dfs_wide.append(p_df_wide)

            a_df_wide = res[['date', 'actual']].rename(columns={'actual': variety}).set_index('date')
            actual_dfs_wide.append(a_df_wide)

            print(f"  {variety} 处理完成, 样本数: {len(res)}")
        else:
            print(f"  {variety} 处理失败")
        sys.stdout.flush()

    if not parquet_dfs:
        print("没有生成任何推理结果。")
        return

    # --- 保存 Parquet 文件 ---
    final_parquet = pd.concat(parquet_dfs, ignore_index=True)

    # 重命名并转换类型
    final_parquet = final_parquet.rename(columns={
        'gsid': '0',
        'gudt': '1',
        'tradeDate': '9',
        'icode': '13',
        'predLogreturn': '3000010'
    })
    final_parquet = final_parquet.astype(str)

    print(final_parquet.dtypes)
    parquet_save_path = OUTPUT_DIR / "inference_predictions.parquet"

    #final_parquet.to_parquet(parquet_save_path, index=False)
    save_files_parquet.save(final_parquet,parquet_save_path)
    print(f"Parquet 预测文件已保存至: {parquet_save_path}")

    # --- 保存原始 CSV 文件 ---
    final_preds_wide = pd.concat(pred_dfs_wide, axis=1, sort=True)
    final_actuals_wide = pd.concat(actual_dfs_wide, axis=1, sort=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_save_path = OUTPUT_DIR / "inference_predictions.csv"
    actual_save_path = OUTPUT_DIR / "inference_actual_returns.csv"

    final_preds_wide.to_csv(pred_save_path)
    final_actuals_wide.to_csv(actual_save_path)

    print(f"\n推理完成！")
    print(f"模型预测值 (CSV) 已保存至: {pred_save_path}")
    print(f"真实收益率 (CSV) 已保存至: {actual_save_path}")


if __name__ == "__main__":
    main()


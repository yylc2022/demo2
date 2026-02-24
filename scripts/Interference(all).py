# python
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.tcn import TCN
from data.features.feature_engineering import generate_features

from datasets.timeseries_dataset import TimeSeriesDataset
from torch.utils.data import DataLoader

RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "outputs" / "models" / "tcn_model2.pt"
OUT_CSV = ROOT / "outputs" / "inference_predictions.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# 与训练一致的配置
LOOKBACK = 50
feature_cols = [
     'logret1','logret2','logret3','logret4','logret5','logret6','logret7','logret8','logret9','logret10','logret11','logret12','logret13','logret14',
     'volume1','volume2','volume3','volume4','volume5','volume6','volume7','volume8','volume9','volume10','volume11','volume12','volume13','volume14',
     'ret_1','ret_20','ret_60','ret_120','ret_acc',
    'vol_5','vol_10','vol_20','vol_ratio',
    'body','upper','lower',
    'vol_z','pos_20',
    # 'ret_prod_1','ret_prod_W',
    # 'vol_prod_1','vol_prod_W',
]
label_col='future_ret_H'  # 需要定义 label_col 供 TimeSeriesDataset 使用，即使推理时只用到 X
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 准备模型结构（与训练保存时一致）
model = TCN(len(feature_cols), num_channels=[16,16], kernel_size=2, dropout=0.2).to(DEVICE)
# 加载权重（兼容 state_dict 或整模型）
try:
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    else:
        try:
            model.load_state_dict(state)
        except Exception:
            # 可能是整模型
            model = state.to(DEVICE)
except Exception as e:
    raise RuntimeError(f"加载模型失败: {e}")

model.eval()

def detect_time_index(df: pd.DataFrame):
    # 优先找到 datetime dtype 的列
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return df.set_index(c)
    # 查找包含 date / 时间 关键词的列
    for c in df.columns:
        low = c.lower()
        if 'date' in low or '时间' in low or 'trade_date' in low:
            try:
                ser = pd.to_datetime(df[c])
                return df.set_index(ser)
            except Exception:
                pass
    # 尝试把第一列解析为日期
    first = df.columns[0]
    try:
        ser = pd.to_datetime(df[first])
        return df.set_index(ser)
    except Exception:
        # 无法识别，返回原始 df（索引为行号）
        return df

all_preds = []  # 要合并的 (series,name)
# 添加：实际收益率输出路径 & 容器（将每个指数的实际收益率一起收集）
OUT_ACT_CSV = ROOT / "outputs" / "inference_actual_returns.csv"
all_actuals = []

for file in sorted(RAW_DIR.iterdir()):
    if not file.suffix.lower() in {'.xlsx', '.xls', '.csv'}:
        continue
    try:
        if file.suffix.lower() in {'.xlsx', '.xls'}:
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
    except Exception as e:
        print(f"Error reading {file.name}: {e}")
        continue

    # ----------------------------------------------------
    # FIX: 解决 "交易日期 is both index and column" 问题
    # ----------------------------------------------------
    # 如果���引名和列名冲突，先消除索引名
    if df.index.name in df.columns:
        df.index.name = None

    # 强制重置索引，确保全是 RangeIndex，避免多级索引问题
    df = df.reset_index(drop=True)

    # 去除可能的重复列名
    df = df.loc[:, ~df.columns.duplicated()]
    # ----------------------------------------------------

    # 特征工程
    try:
        df = generate_features(df)
    except Exception as e:
        print(f"特征生成失败 {file.name}: {e}")
        # 即使特征生成失败也尝试通过原始日期索引输出空预测、并尽可能提取实际收益
        df = df.copy()

    # 在创建 dataset 前先处理 NaN
    # 注意：generate_features 会生成 rolling 导致前面 NaN，也会生成 label shift 导致后面 NaN
    # 这里为了保证 TimeSeriesDataset 正常工作，我们只 dropna feature_cols
    # 至于 label_col，如果用于验证（IC等）则需要 dropna，如果纯预测，可以保留
    # 但由于 TimeSeriesDataset 内部会读取 label_col，所以必须保证 label_col 也没问题 或者 填充
    # 这里遵循 train.py 的逻辑： dropna()

    # 临时修复：如果 label_col 不存在（比如 feature_engineering 失败），加一个 dummy prediction logic
    if label_col not in df.columns:
        df[label_col] = 0.0

    df = df.dropna().reset_index(drop=True)

    if len(df) <= LOOKBACK:
        print(f"{file.name} 数据长度 <= LOOKBACK，跳过")
        # 同样需要输出该指数的空列（索引将基于日期检测）
        df_timeidx = detect_time_index(df)
        index = df_timeidx.index if isinstance(df_timeidx, pd.DataFrame) else df.index
        col_name = file.stem
        all_preds.append(pd.Series(dtype=float, name=col_name, index=index))
        # 实际收益：
        if 'ret_1' in df.columns:
            ser_actual = pd.Series(pd.to_numeric(df['ret_1'], errors='coerce').values, index=index, name=col_name)
        else:
            ser_actual = pd.Series(dtype=float, name=col_name, index=index)
        all_actuals.append(ser_actual)
        continue

    # ==========================================
    # 使用 DataLoader 进行批量推理 (优化性能)
    # ==========================================
    # 创建 dataset
    try:
        dataset = TimeSeriesDataset(df, feature_cols, label_col, LOOKBACK)
    except Exception as e:
        print(f"Dataset creation failed for {file.name}: {e}")
        continue

    inference_loader = DataLoader(
        dataset,
        batch_size=1024,  # 推理可以使用较大的 batch size
        shuffle=False,
        num_workers=0
    )

    preds_list = []
    with torch.no_grad():
        for X_batch, _ in inference_loader:
            X_batch = X_batch.to(DEVICE)
            # 模型输出
            batch_pred = model(X_batch).squeeze()
            if batch_pred.ndim == 0: # single element batch
                batch_pred = batch_pred.unsqueeze(0)
            preds_list.append(batch_pred.cpu().numpy())

    if not preds_list:
        pred = np.array([])
    else:
        pred = np.concatenate(preds_list)

    # ------------------------------------------
    # 对齐逻辑 (Alignment)
    # ------------------------------------------
    # Dataset 产生的第 k 个样本 (X[k], y[k])
    # TimeSeriesDataset 中：
    # idx 对应的 window 是 [idx : idx + LOOKBACK] (切片不包含 end) -> indices [idx, ..., idx + LOOKBACK - 1]
    # 对应的 label 是 df.iloc[idx + LOOKBACK - 1][label_col] (即窗口最后一个时间点的 label)
    # 而 label_col = 'future_ret_H' (t+1 return).
    # 所以 Model(X[k]) 预测的是 df.iloc[idx + LOOKBACK - 1] 这一行的 label_col

    # 也就是说：pred[k] 应该对齐到 df.iloc[idx + LOOKBACK - 1] 这一行？
    # 不完全是。train.py 中 label 是 shift(-H) 之后的。
    # 让我们直接看 idx=0。
    # X[0] = df.loc[0 : LOOKBACK-1]. End time is `LOOKBACK-1`.
    # Prediction is made at `LOOKBACK-1`.
    # Target (y[0]) is `future_ret_H` at `LOOKBACK-1`.
    # `future_ret_H` at `t` is `logret` at `t+1`.
    # So `y[0]` is `logret` at `LOOKBACK`.

    # 我们想要把 `pred[0]` 放在 `logret` 发生的时间点，也就是 `LOOKBACK` 这一行。
    # `df` 的第 `LOOKBACK` 行 (index=LOOKBACK) 正是 `logret` 发生的时间。

    # 所以： `df.iloc[LOOKBACK:]` 正好对应 `pred` 序列 (长度 len(df) - LOOKBACK + 1 ? Update: TimeSeriesDataset len is n-w+1)
    # TimeSeriesDataset len: len(df) - window_size + 1
    # df.iloc[window_size-1:] len: len(df) - (window_size - 1)
    # 让我们仔细检查 TimeSeriesDataset 的 __len__
    # return len(self.df) - self.window_size + 1
    # 比如 len=5, window=2. -> len = 5-2+1=4. indices: 0,1,2,3.
    # idx=0: [0,1]. y at 1.
    # idx=1: [1,2]. y at 2.
    # idx=2: [2,3]. y at 3.
    # idx=3: [3,4]. y at 4.
    # Predictions correspond to indices [window_size-1, ..., len-1] in ORIGINAL df.
    # df.iloc[window_size-1 :] covers exactly these indices.

    # 用户要求：t天的预测值放入到t+1天
    # pred[k] 是在时间 t (index = k + LOOKBACK - 1) 结束时做出的预测。
    # 我们希望这个预测出现在 t+1 (index = k + LOOKBACK) 的行上。

    # 原始对其：
    # df_sub = df.iloc[LOOKBACK-1:].copy() (Starts at t)
    # pred[0] corresponds to t. Place at t.

    # 新要求对齐：
    # pred[0] corresponds to t. Place at t+1.
    # 所以我们需要让 df_sub 从 t+1 开始，并且 pred 也要对应平移。

    # 如果我们将 pred 整体向后移动一位（在时间轴上），它就对应 t+1。
    # 或者，我们取 df 的时间轴时，直接取 t+1 的时间轴。

    # 方法：
    # 1. 取出 df 从 LOOKBACK 开始的切片 (这是 t+1, t+2, ...)
    # 2. 将 pred[0] (t的预测) 放在 df[LOOKBACK] (t+1行)
    # 3. pred 的长度可能比 df[LOOKBACK:] 少 1 (因为最后一天没有 t+1) 或者多 (取决于 loop range)

    # TimeSeriesDataset 生成 N 个样本。
    # 最后一个样本窗口结束于 len(df)-1 (最后一行). 时间 t_end.
    # 它的预测应该放在 t_end + 1. 但是 df 没有 t_end + 1 行。
    # 除非我们 extend df 或者只保留能对齐的部分。
    # 通常推断是对已有数据的回测推断，所以如果 t_end + 1 不在数据里，就丢弃或忽略。

    # 让我们看看 pred 的长度和 df 的关系。
    # Dataset len = L_orig - W + 1.
    # indices: 0, 1, ..., L_orig - W.
    # loops: 0 to L_orig - W.
    # Last window ends at (L_orig - W) + W - 1 = L_orig - 1. (Last row of df). OK.
    # This prediction is for L_orig. (Outside df).

    # df.iloc[LOOKBACK:] starts from index LOOKBACK. (Which is index W).
    # row 0 of df.iloc[LOOKBACK:] is index W.
    # pred[0] is from window index 0...W-1. Prediction at W-1.
    # User wants this prediction at W (t+1 relative to W-1).
    # So pred[0] should be aligned with df index W.

    # So `df_sub = df.iloc[LOOKBACK:]` is the correct timeframe to place `pred`.
    # Length check:
    # pred len = L - W + 1.
    # df.iloc[LOOKBACK:] len = L - W.
    # pred has 1 more element (the prediction for tomorrow, beyond dataset).
    # We should trim the last element of pred if we want to match existing df rows.

    df_sub = df.iloc[LOOKBACK:].copy().reset_index(drop=True)

    # 裁剪 pred 以匹配 df_sub (丢弃对未来的一步预测，或者保留它但不放入此df)
    # 这里我们使其长度一致以便合并
    current_pred = pred[:-1] # 丢弃最后一个预测（针对未来的）

    if len(current_pred) != len(df_sub):
         # Double check
         min_len = min(len(current_pred), len(df_sub))
         current_pred = current_pred[:min_len]
         df_sub = df_sub.iloc[:min_len]

    df_sub['pred'] = current_pred

    # 检测时间索引并得到 Series
    df_timeidx = detect_time_index(df_sub)
    if isinstance(df_timeidx, pd.DataFrame) and df_timeidx.index.equals(df_sub.index):
        index = df_sub.index
    else:
        index = df_timeidx.index

    # 预测值 series
    ser_pred = pd.Series(df_sub['pred'].values, index=index)
    col_name = file.stem
    ser_pred.name = col_name
    all_preds.append(ser_pred)

    # 实际收益率 (Actual Return)
    # 在 df_sub (也就是 Row T) 里面：
    # pred 是 Prediction for T (made at T-1).
    # ret_1 是 Return at T.
    # 用户要求：预测值与真实收益对齐，且 t 放到 t+1.
    # 这里的 Row T 就包含了 "Return at T" 和 "Prediction for T".
    # 所以直接取出 ret_1 即可。它就是 "t+1" 的真实收益 (相对于 generating prediction 的 t 而言)

    if 'ret_1' in df_sub.columns:
        actual = pd.to_numeric(df_sub['ret_1'], errors='coerce')
        ser_actual = pd.Series(actual.values, index=index, name=col_name)
    elif 'logret' in df_sub.columns:
        actual = pd.to_numeric(df_sub['logret'], errors='coerce')
        ser_actual = pd.Series(actual.values, index=index, name=col_name)
    else:
        # 回退
        price_col = None
        candidates = ['close', 'Close', 'close_price', 'closeprice', 'adj_close', 'adjclose', '收盘价', '收盘']
        for c in candidates:
            if c in df_sub.columns:
                price_col = c
                break
        if price_col is not None:
            price = pd.to_numeric(df_sub[price_col], errors='coerce')
            actual = price.pct_change()
            ser_actual = pd.Series(actual.values, index=index, name=col_name)
        else:
            ser_actual = pd.Series(dtype=float, index=index, name=col_name)

    all_actuals.append(ser_actual)


# 合并所有 series（使用 outer join）
if not all_preds:
    raise RuntimeError("没有可用的推理结果。")

pred_df = pd.concat(all_preds, axis=1, join='outer').sort_index()
# 保存为 CSV，索引为时间
pred_df.to_csv(OUT_CSV, index=True)

print(f"推理完成，保存到 {OUT_CSV}")

# 合并并保存实际收益率
if all_actuals:
    actual_df = pd.concat(all_actuals, axis=1, join='outer').sort_index()
    actual_df.to_csv(OUT_ACT_CSV, index=True)
    print(f"实际收益率保存到 {OUT_ACT_CSV}")
else:
    print("没有可用的实际收益率结果，未生成文件。")

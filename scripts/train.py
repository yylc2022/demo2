from pathlib import Path
import pandas as pd
from datasets.timeseries_dataset import TimeSeriesDataset
from data.features.feature_engineering import generate_features
from torch.utils.data import DataLoader
import torch
from trainers import trainer,trainer_regression,trainer_classification
from models.tcn import TCN
from models.lstm import LSTMModel

ROOT = Path(__file__).resolve().parents[1]
data_path = ROOT / "data" / "raw" / "000852.SH-行情统计-20251203.xlsx"


BATCH_SIZE=32
EPOCHS = 100
LEARNING_RATE = 0.001
LOOKBACK = 50



df = pd.read_excel(data_path)
#df = pd.read_csv(ROOT / "data" / "processed" / "feat.csv")
df=generate_features(df)
df = df.dropna().reset_index(drop=True)
print(df.head())
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
label_col='future_ret_H'


dataset = TimeSeriesDataset(df, feature_cols, label_col, LOOKBACK)


N = len(dataset)

# 训练 / 验证比例（你可以改成 0.8）
train_ratio = 0.8
split_idx = int(N * train_ratio)


# Subset → PyTorch 提供的子集工具
# 严格按时间顺序切分（金融序列不能随机打乱）
# 训练集：从最早的数据到 split_idx
# 验证集：剩下的后面的数据

# 按时间顺序切
train_dataset = torch.utils.data.Subset(
    dataset,
    range(0, split_idx)
)

val_dataset = torch.utils.data.Subset(
    dataset,
    range(split_idx, N)
)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples:   {len(val_dataset)}")


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,      # 训练集可以打乱
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,     # ❗验证集绝对不能打乱
)

print('样本量:',len(dataset))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TCN(len(feature_cols), num_channels=[16,16], kernel_size=2, dropout=0.2)
#model= LSTMModel(input_dim=len(feature_cols), hidden_dim=3, num_layers=2, dropout=0.2)
trainer1 = trainer.Trainer(model, train_loader, val_loader,lr=LEARNING_RATE,  device=DEVICE,save_dir=ROOT/"outputs" / "models",save_name='tcn_model2.pt')
#trainer1=trainer_regression.TrainerRegression(model, train_loader, val_loader,lr=LEARNING_RATE,  device=DEVICE)
#trainer1=trainer_classification.TrainerClassification(model, train_loader, val_loader,lr=LEARNING_RATE,  device=DEVICE)
val_probs, val_labels = trainer1.train(EPOCHS)

# 加载验证集上 IC 最高的最优模型
trainer1.load_best_model()


# -------------------------
# 7. 预测残差 + 合成最终预测
# -------------------------
model.eval()
with torch.no_grad():
    X_all = torch.tensor([df[feature_cols].values[i-LOOKBACK:i]
                          for i in range(LOOKBACK, len(df))], dtype=torch.float32).to(DEVICE)

    residual_pred = model(X_all).cpu().numpy().flatten()

df = df.iloc[LOOKBACK:].copy()
df['residual_pred'] = residual_pred
df['final_pred_logret'] =df['residual_pred']

print(df[['final_pred_logret','residual_pred','logret']].head())

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']   # 中文
plt.rcParams['axes.unicode_minus'] = False

# -------------------------
# 8. 构造真实收益 & 策略收益
# -------------------------
import numpy as np
# 真实 log return
df['true_logret'] = df['logret']



# TCN 残差策略
H=1
# 决策信号（t 时刻）
signal = np.sign(df['final_pred_logret'])

# 持有期真实收益（t+1 ~ t+H）
future_ret = df['logret'].rolling(H).sum().shift(-H)

# 策略收益在 t+H 记账
df['ret_final'] = signal * future_ret
df['ret_final'] = df['ret_final'].shift(H)

# Buy & Hold
df['ret_bh'] = df['logret'].shift(-H)
#
# # 累计收益（log return 累加）
# df['cum_arima'] = df['ret_arima'][:3500].cumsum()
# df['cum_final'] = df['ret_final'][:3500].cumsum()
# df['cum_bh'] = df['ret_bh'][:3500].cumsum()
# 累计收益（log return 累加）
df['cum_final'] = df['ret_final'][:N].cumsum()
df['cum_bh'] = df['ret_bh'][:N].cumsum()

plt.figure(figsize=(12,5))
plt.plot(df['cum_bh'], label='Bench', linestyle='--',linewidth=2)
#plt.plot(df['cum_arima'], label='ARIMA Only')
plt.plot(df['cum_final'], label='ARIMA + TCN Residual',linewidth=3)

plt.title('Cumulative Log Return Comparison')
plt.xlabel('Time')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(ROOT/'outputs'/'plots'/'test_cum_return_comparison.png')
plt.show()
plt.close()

df['cum_final'] = df['ret_final'][int(N*train_ratio):].cumsum()
df['cum_bh'] = df['ret_bh'][int(N*train_ratio):].cumsum()

plt.figure(figsize=(12,5))
plt.plot(df['cum_bh'], label='Bench', linestyle='--',linewidth=2)
plt.plot(df['cum_final'], label='TCN ',linewidth=3)
#plt.plot(df['cum_final'], label='ARIMA + TCN ',linewidth=3)

plt.title('Cumulative Log Return Comparison')
plt.xlabel('Time')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(ROOT/'outputs'/'plots'/'valid_cum_return_comparison.png')
plt.show()
plt.close()
# =========================
# Val 方向准确率 Directional Accuracy
# =========================
model.eval()

val_preds = []
val_trues = []

with torch.no_grad():
    for X, y in val_loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(X).squeeze()

        val_preds.append(pred.cpu())
        val_trues.append(y.cpu())

val_preds = torch.cat(val_preds)
val_trues = torch.cat(val_trues)

# 方向（+1 / -1 / 0）
pred_sign = torch.sign(val_preds)
true_sign = torch.sign(val_trues)

# 去掉真实收益为 0 的样本（可选，但更干净）
mask = true_sign != 0

direction_acc = (pred_sign[mask] == true_sign[mask]).float().mean().item()

print(f"✅ Val Direction Accuracy: {direction_acc:.4f}")

# 保存最终用于回测分析的 df
out_df_path = ROOT / "outputs" / "train_analyzed.csv"
df.to_csv(out_df_path, index=False) # index已经被reset了，所以不需要保存索引，除非有特定时间索引在index上
print(f"✅ Saved analyzed dataframe to {out_df_path}")

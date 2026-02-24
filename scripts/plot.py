import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_performance(csv_path, output_dir):
    # 1. 加载推理结果
    df = pd.read_csv(csv_path)
    df['交易日期'] = pd.to_datetime(df['交易日期'])

    # 2. 数据清洗
    # 过滤掉没有预测和收益的行（通常是开头和结尾）
    df = df.dropna(subset=['strategy_ret', 'logret', 'model_prediction']).copy()

    # 3. 计算收益指标 (基于 Log Return)
    # 因为对数收益率具有可加性：累计收益 = sum(log_ret)
    df['cum_strategy'] = df['strategy_ret'].cumsum()
    df['cum_benchmark'] = df['logret'].cumsum()

    # 转换为净值曲线 (从1开始)
    df['nv_strategy'] = np.exp(df['cum_strategy'])
    df['nv_benchmark'] = np.exp(df['cum_benchmark'])

    # 计算超额对数收益 (Alpha)
    df['excess_log_ret'] = df['cum_strategy'] - df['cum_benchmark']

    # 4. 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # --- 主图：累计收益曲线 ---
    ax1.plot(df['交易日期'], df['nv_benchmark'], label='基准 (Buy & Hold)',
             color='gray', alpha=0.5, linestyle='--')
    ax1.plot(df['交易日期'], df['nv_strategy'], label='TCN 择时策略',
             color='#d62728', linewidth=2)

    ax1.set_title(f'策略累计净值曲线 (预测对齐模式: t+1)', fontsize=16)
    ax1.set_ylabel('累计净值')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # --- 副图：超额收益 (Alpha) ---
    ax2.fill_between(df['交易日期'], df['excess_log_ret'], 0,
                     where=(df['excess_log_ret'] >= 0), color='red', alpha=0.3, label='正超额')
    ax2.fill_between(df['交易日期'], df['excess_log_ret'], 0,
                     where=(df['excess_log_ret'] < 0), color='green', alpha=0.3, label='负超额')

    ax2.set_ylabel('超额收益 (Log)')
    ax2.set_xlabel('日期')
    ax2.axhline(0, color='black', lw=1)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    out_path = Path(output_dir) / "strategy_backtest.png"
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"✅ 图表已保存至: {out_path}")


if __name__ == "__main__":
    # 配置路径
    ROOT = Path(__file__).resolve().parents[1]
    csv_file = ROOT / "outputs" / "historical_inference_full.csv"
    save_dir = ROOT / "outputs"

    plot_performance(csv_file, save_dir)
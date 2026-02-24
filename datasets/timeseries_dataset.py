# time_series.py
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.

    Args:
        df (pd.DataFrame): 原始时间序列数据，必须包含特征列和label列
        feature_cols (list[str]): 用作模型输入的特征列名
        label_col (str): 用作模型输出的label列名
        window_size (int): 时间序列滑动窗口长度
        transform (callable, optional): 数据预处理函数
    """

    def __init__(self, df, feature_cols, label_col, window_size=20, transform=None):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.window_size = window_size
        self.transform = transform

        # 将数据转换为numpy数组
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.labels = self.df[label_col].values.astype(np.float32)

    def __len__(self):
        # 数据长度为总行数减去窗口长度
        return len(self.df) - self.window_size + 1

    def __getitem__(self, idx):
        # 滑动窗口获取输入特征
        x = self.features[idx:idx + self.window_size]
        y = self.labels[idx + self.window_size - 1]  # 预测窗口最后一天的label

        if self.transform:
            x = self.transform(x)

        # 转为torch.tensor
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


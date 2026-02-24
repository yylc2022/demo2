import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    多层 LSTM 时间序列预测模型
    输入: (batch, seq_len, input_dim)
    输出: (batch, 1) 或 (batch, num_classes)
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            output_dim: int = 1,
            dropout: float = 0.2,
            bidirectional: bool = False
    ):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 输出层
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_dim, device=x.device)

        # LSTM 前向传播
        out, _ = self.lstm(x, (h0, c0))  # out: (batch, seq_len, hidden_dim*num_directions)

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # (batch, hidden_dim*num_directions)

        out = self.fc(out)  # (batch, output_dim)
        return out

# ======================
# Step 5: TCN（Temporal Convolutional Network）
# ======================


import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """裁掉 padding，保证因果性"""

    # 作用：裁掉卷积后的多余 padding，保证因果卷积 不会看到未来信息
    # TCN 卷积为了扩展感受野，会在左边或右边加 padding
    # 对于因果卷积（只看过去）：
    # padding 会让卷积输出比原序列长
    # Chomp1d 就把多余的末尾部分切掉

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


# TCN 的基本构建模块，类似 ResNet 的 残差卷积块
# 一般包含：两层 1D 卷积、 ReLU 激活、Dropout、残差连接（residual）
class TemporalBlock(nn.Module):

    #in_ch → 输入通道数（特征维度）
    # out_ch → 输出通道数（卷积核数量）
    # kernel_size → 卷积核大小
    # dilation → 膨胀系数（dilated conv 扩大感受野）
    # dropout → Dropout 概率
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()

        #计算卷积 padding，保证 *输出长度 = 输入长度 + padding - (kernel_size-1)dilation，之后用 Chomp1d 切掉 padding，保证因果性
        padding = (kernel_size - 1) * dilation

        #         Conv1d：
        # 输入 [B, in_ch, T] → 输出 [B, out_ch, T + padding]
        # Chomp1d：切掉 padding，得到 [B, out_ch, T]
        # ReLU：非线性
        # Dropout：防止过拟合

        self.conv1 = nn.Conv1d(
            in_ch, out_ch,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)


        # 同第一层，但输入输出通道都是 out_ch
        # 两层卷积可以捕捉更复杂的时间依赖

        self.conv2 = nn.Conv1d(
            out_ch, out_ch,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)


        # 如果输入通道数 in_ch != out_ch，用 1x1 卷积做线性变换匹配残差
        # 如果相等，直接用 x 做残差

        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1)
            if in_ch != out_ch else None
        )


    # 先经过两层卷积 + ReLU + Dropout
    # 再加上残差连接 x
    # 最后再经过一次 ReLU
    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TCN(nn.Module):

#     num_features → 输入特征数量（比如 13 个特征）
# num_channels → 每个 TemporalBlock 的卷积输出通道
# kernel_size → 卷积核大小
# dropout → Dropout 概率
    def __init__(
        self,
        num_features,
        num_channels=[16,16],
        kernel_size=2,
        dropout=0.8
    ):
        super().__init__()

        #         循环构建多个 TemporalBlock
        # 每层 dilation 翻倍 → 感受野指数增长
        # 例：dilation=1,2,4,8
        # 可以快速看到几十天甚至几百天的过去数据
        # nn.Sequential 把所有模块串起来

        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_features if i == 0 else num_channels[i-1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch, out_ch,
                    kernel_size,
                    dilation,
                    dropout
                )
            )

        self.network = nn.Sequential(*layers)

        # 每个样本最后的时间点输出 [B, out_ch] → 预测单个值 [B, 1]
        # 对应 label（是否开波段）
        self.fc = nn.Linear(num_channels[-1], 1)

    # 输入 x [batch_size, time_steps, features]
    # transpose(1,2) → [B, features, time_steps]
    # Conv1d 在 PyTorch 里需要 [B, C, T]
    # y = self.network(x) → 经过所有 TemporalBlock
    # out = y[:, :, -1] → 只取最后一个时间点的输出
    # 因为我们预测的是 当前时点 label
    # self.fc(out) → 得到预测值 [B]

    def forward(self, x):
        # x: [B, T, F] → [B, F, T]
        x = x.transpose(1, 2)
        y = self.network(x)
        # 只取最后一个时间点
        out = y[:, :, -1]
        logit = self.fc(out).squeeze(-1)
        return logit

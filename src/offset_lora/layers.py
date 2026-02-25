import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class OffsetLinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.05, eta=1.0, gamma=1.0):
        super(OffsetLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.eta = eta
        self.gamma = gamma

        # 标准 LoRA 参数 (可训练)
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # Offset 锚点参数 (冻结，不可训练)
        self.register_buffer('lora_A0', torch.zeros((r, in_features)))
        self.register_buffer('lora_B0', torch.zeros((out_features, r)))

        self.initialized = False

    def reset_parameters(self, weight_data=None):
        """
        对应论文 3.3.2 节：近似正交初始化策略
        """
        # 1. 初始化可训练参数 A 和 B (使用正交初始化或高斯)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 2. 设置 A0 和 B0 (偏移补偿项)
        # 根据论文，为了实现初始 ΔW = 0，令 A0 = A_init, B0 = B_init
        # 同时确保 A0, B0 满足近似正交条件以优化曲率
        with torch.no_grad():
            # 这里简单演示：使用正交矩阵填充
            nn.init.orthogonal_(self.lora_A0, gain=self.eta)
            nn.init.orthogonal_(self.lora_B0, gain=self.gamma)
            self.lora_A.copy_(self.lora_A0)
            self.lora_B.copy_(self.lora_B0)

        self.initialized = True

    def forward(self, x, base_layer_output):
        # 核心公式实现: ΔW = (B @ A - B0 @ A0)
        # 根据论文 3.1 节，我们需要计算补偿后的增量

        result = base_layer_output
        if self.r > 0:
            after_dropout = self.lora_dropout(x)

            # 计算当前路径: x @ A.T @ B.T
            current_path = (after_dropout @ self.lora_A.t() @ self.lora_B.t())

            # 计算偏移路径: x @ A0.T @ B0.T
            offset_path = (after_dropout @ self.lora_A0.t() @ self.lora_B0.t())

            # 最终增量 = (BA - B0A0) * scaling
            result += (current_path - offset_path) * self.scaling

        return result
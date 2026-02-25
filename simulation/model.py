import torch
import torch.nn as nn


class OffsetLoRAModel(nn.Module):
    def __init__(self, d, r, mode='standard', gamma_val=None, lora_alpha=1.0):
        super().__init__()
        self.mode = mode
        self.r = r
        # 增加 LoRA 缩放因子，默认 1.0 保持你现在的实验逻辑不变
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # 默认缩放因子（对应论文中的 η 和 γ）
        if gamma_val is None:
            gamma_val = 1.0 / (r ** 0.5)

        # 定义 A0, B0 缓冲区
        if mode == 'standard':
            A0_data = torch.randn(r, d) * (1.0 / d ** 0.5)
            B0_data = torch.zeros(d, r)
        elif mode in ['non_zero_li', 'offset_gaussian']:
            A0_data = torch.randn(r, d) * (1.0 / d ** 0.5)
            B0_data = torch.randn(d, r) * (1.0 / d ** 0.5)
        elif mode == 'offset_orthogonal':
            # QR 分解实现正交初始化 (对应 Ch4 理论)
            q_a, _ = torch.linalg.qr(torch.randn(d, r))
            q_b, _ = torch.linalg.qr(torch.randn(d, r))
            A0_data = q_a.T * gamma_val
            B0_data = q_b * gamma_val

        self.register_buffer('A0', A0_data.float())
        self.register_buffer('B0', B0_data.float())

        # 初始化可训练参数，确保初始时刻 dW = 0 (除 non_zero_li 外)
        self.A = nn.Parameter(self.A0.clone())
        self.B = nn.Parameter(self.B0.clone())

    def forward(self, x):
        if self.mode == 'non_zero_li':
            # 模拟没有偏移补偿的非零初始化情况
            dW = (self.B @ self.A) * self.scaling
        else:
            # Offset-LoRA 核心公式: ΔW = (BA - B0A0) * scaling
            # 这一步体现了“偏移补偿”的核心思想
            dW = (self.B @ self.A - self.B0 @ self.A0) * self.scaling
        return x @ dW
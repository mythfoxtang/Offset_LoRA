import torch
import torch.nn as nn


def kronecker_product(A, B):
    """
    计算两个矩阵的 Kronecker 积 A ⊗ B
    对应论文第 5 章中分析 Hessian 矩阵 H = (xx^T) ⊗ (gg^T) 的理论工具
    """
    res = torch.ger(A.view(-1), B.view(-1))
    res = res.reshape(*(A.shape + B.shape))
    res = res.permute(0, 2, 1, 3)
    return res.reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])


def get_isometry_init(out_features, in_features, r, gain=1.0):
    """
    实现等距映射初始化 (Isometric Initialization)
    确保 B0^T B0 = γ^2 I 和 A0 A0^T = η^2 I
    用于消除论文中提到的“梯度冷启动”和“病态曲率”
    """
    # 初始化 A0 (r x k)
    A0 = torch.empty(r, in_features)
    nn.init.orthogonal_(A0, gain=gain)

    # 初始化 B0 (d x r)
    B0 = torch.empty(out_features, r)
    nn.init.orthogonal_(B0, gain=gain)

    return A0, B0


def compute_condition_number(matrix):
    """
    计算矩阵的条件数 κ(H) = λ_max / λ_min
    用于验证 Offset-LoRA 提升训练稳定性的数值指标
    """
    singular_values = torch.linalg.svdvals(matrix)
    if singular_values[-1] == 0:
        return float('inf')
    return (singular_values[0] / singular_values[-1]).item()


def get_hessian_trace(model):
    """
    计算模型参数梯度的迹 (Trace)，用于监控损失平面的平滑度
    """
    total_trace = 0
    for p in model.parameters():
        if p.grad is not None:
            total_trace += torch.sum(p.grad ** 2)
    return total_trace.item()
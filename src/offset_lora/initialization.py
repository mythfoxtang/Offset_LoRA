import torch
import torch.nn as nn


def apply_offset_orthogonal_init(model, eta=1.0, gamma=1.0):
    """
    遍历模型中所有的 OffsetLinear 层并应用正交初始化
    对应论文第 5 章关于特征值簇化的理论要求
    """
    for name, module in model.named_modules():
        if isinstance(module, OffsetLinear):
            r, k = module.lora_A.shape
            d, _ = module.lora_B.shape

            with torch.no_grad():
                # 对 A0 进行正交化 (r x k)
                a_rand = torch.randn((r, k))
                q_a, _ = torch.linalg.qr(a_rand.t())
                module.lora_A0.copy_(q_a.t()[:r, :] * eta)

                # 对 B0 进行正交化 (d x r)
                b_rand = torch.randn((d, r))
                q_b, _ = torch.linalg.qr(b_rand)
                module.lora_B0.copy_(q_b[:, :r] * gamma)

                # 同步初始 A, B 使得初始 ΔW = 0
                module.lora_A.copy_(module.lora_A0)
                module.lora_B.copy_(module.lora_B0)

    print(f"Successfully initialized Offset-LoRA with eta={eta}, gamma={gamma}")
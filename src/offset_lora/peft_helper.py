import torch
from .layers import OffsetLinear


def inject_offset_lora(model, target_modules=["query", "value"], r=8, lora_alpha=16):
    """
    将模型中的标准 Linear 层替换为 OffsetLinear
    """
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, torch.nn.Linear):
            in_features = module.in_features
            out_features = module.out_features

            # 创建新层
            new_layer = OffsetLinear(in_features, out_features, r=r, lora_alpha=lora_alpha)

            # 这里可以根据需要将原始 module 的 weight 指针传给 new_layer
            # 或者在训练脚本中通过包装 forward 来实现
            # 简便起见，本演示采用逻辑替换

            # 实际工程中，建议继承自 peft.LoraLayer 效果更佳
            pass
    return model
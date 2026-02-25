# ==========================================
# 0. 环境依赖预检 (解决 oss2, addict, datasets 冲突)
# ==========================================
try:
    import modelscope
    import datasets

    if int(datasets.__version__.split('.')[0]) >= 3: raise ImportError
except (ImportError, ModuleNotFoundError):
    import subprocess, sys

    print(">>> 正在安装核心依赖 (RoBERTa/MRPC 压测环境)...")
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    mirror = ["-i", "https://mirrors.aliyun.com/pypi/simple/"]
    libs = ["modelscope", "transformers", "peft", "accelerate", "evaluate", "addict", "oss2", "scikit-learn"]
    subprocess.check_call(pip_cmd + libs + mirror)
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "datasets"])
    subprocess.check_call(pip_cmd + ["datasets<3.0.0"] + mirror)

import os, math, torch, gc
import torch.nn as nn
from tqdm import tqdm
from modelscope import snapshot_download
from modelscope.msdatasets import MsDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import numpy as np

# ================= 1. 极限压测配置 =================
CONFIG = {
    "mode": "offset",  # 选项: "offset" (建议先跑) 或 "standard" (易崩)
    "lr": 1e-3,  # 1e-3 是压测核心：标准 LoRA 在此 LR 下极易 NaN
    "model_id": "AI-ModelScope/roberta-base",
    "r": 8,
    "lora_alpha": 16,
    "max_len": 128,
    "batch_size": 16
}
os.environ['MODELSCOPE_CACHE'] = './cache'


# ================= 2. Offset-LoRA 算子 =================
class OffsetLoraLinear(nn.Module):
    def __init__(self, base_layer, r=8, lora_alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.r, self.scaling = r, lora_alpha / r
        in_f, out_f = base_layer.in_features, base_layer.out_features
        device = base_layer.weight.device

        # 定义 LoRA 参数
        self.lora_A = nn.Parameter(torch.zeros((r, in_f), device=device))
        self.lora_B = nn.Parameter(torch.zeros((out_f, r), device=device))

        # 定义 Offset 缓冲区 (不计入梯度训练)
        self.register_buffer('lora_A0', torch.zeros((r, in_f), device=device))
        self.register_buffer('lora_B0', torch.zeros((out_f, r), device=device))
        self.reset_parameters()

    def reset_parameters(self):
        # 基础 Kaiming 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        if CONFIG["mode"] == "offset":
            # Offset 模式：通过 QR 分解建立等距映射
            nn.init.kaiming_uniform_(self.lora_A0, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B0, a=math.sqrt(5))

            # 正交化
            q_a, _ = torch.linalg.qr(self.lora_A0.T)
            self.lora_A0.data = q_a.T * (1.0 / math.sqrt(self.r))
            q_b, _ = torch.linalg.qr(self.lora_B0)
            self.lora_B0.data = q_b * (1.0 / math.sqrt(self.r))

            # 权重对齐：确保初始 dW = 0
            self.lora_A.data.copy_(self.lora_A0.data)
            self.lora_B.data.copy_(self.lora_B0.data)

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_val = (x @ self.lora_A.T @ self.lora_B.T)

        if CONFIG["mode"] == "offset":
            # 核心公式: Delta W = (BA - B0A0)
            lora_output = lora_val - (x @ self.lora_A0.T @ self.lora_B0.T)
        else:
            lora_output = lora_val

        return base_output + lora_output * self.scaling


# ================= 3. 模型准备与注入 =================
# 清理显存碎片
gc.collect();
torch.cuda.empty_cache()

print(f">>> 正在准备 RoBERTa-Base 模型 (Mode: {CONFIG['mode']})...")
model_dir = snapshot_download(CONFIG["model_id"])
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2).to("cuda")

# 注入目标：Query 与 Value 线性层
for name, module in model.named_modules():
    if any(target in name for target in ["query", "value"]) and isinstance(module, nn.Linear):
        parts = name.rsplit('.', 1)
        parent = dict(model.named_modules())[parts[0]]
        module.weight.requires_grad = False
        setattr(parent, parts[1], OffsetLoraLinear(module, r=CONFIG["r"]))

# 冻结 Base，仅训练 LoRA 和 Classifier
for n, p in model.named_parameters():
    p.requires_grad = ("lora_" in n or "classifier" in n)

# 加载 MRPC 数据集
ms_ds = MsDataset.load('glue', subset_name='mrpc', trust_remote_code=True)
train_ds = ms_ds['train'].to_hf_dataset()

# ================= 4. 压测训练循环 =================
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"])
model.train()
losses = []

print(f"🚀 启动 MRPC 压测 | 学习率: {CONFIG['lr']} | 模式: {CONFIG['mode']}")
pbar = tqdm(range(0, len(train_ds), CONFIG["batch_size"]))

for i in pbar:
    batch = train_ds[i: i + CONFIG["batch_size"]]
    # MRPC 关键点：双句子输入 (Sentence1, Sentence2)
    inputs = tokenizer(
        batch["sentence1"], batch["sentence2"],
        padding="max_length", truncation=True,
        max_length=CONFIG["max_len"], return_tensors="pt"
    ).to("cuda")
    labels = torch.tensor(batch["label"]).to("cuda")

    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss

    # 💥 稳定性检测：捕获 NaN
    if torch.isnan(loss):
        print(f"\n[CRITICAL] {CONFIG['mode']} 模式在 Step {i // CONFIG['batch_size']} 崩溃 (NaN)！")
        break

    loss.backward()
    optimizer.step()

    current_loss = round(loss.item(), 4)
    losses.append(current_loss)
    pbar.set_description(f"Mode:{CONFIG['mode']} | Loss:{current_loss:.4f}")

# ================= 5. 结果输出 =================
print("\n" + "=" * 50)
print(f"✅ 实验结束 | 模式: {CONFIG['mode']} | 记录点数: {len(losses)}")
print("-" * 20 + " 完整 LOSS 列表 " + "-" * 20)
print(losses)
print("=" * 50)

# 清理显存
del model;
gc.collect();
torch.cuda.empty_cache()
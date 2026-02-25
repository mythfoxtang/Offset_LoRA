# ==========================================
# 0. 环境依赖预检 (解决 oss2, addict, datasets 冲突)
# ==========================================
try:
    import modelscope
    import datasets
    import bitsandbytes

    if int(datasets.__version__.split('.')[0]) >= 3: raise ImportError
except (ImportError, ModuleNotFoundError):
    import subprocess, sys

    print(">>> 正在自动安装核心依赖 (bitsandbytes, peft, modelscope 等)...")
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    mirror = ["-i", "https://mirrors.aliyun.com/pypi/simple/"]
    # 增加 bitsandbytes 用于 4-bit 量化
    libs = ["modelscope", "transformers", "peft", "accelerate", "bitsandbytes", "evaluate", "addict", "oss2",
            "scikit-learn"]
    subprocess.check_call(pip_cmd + libs + mirror)
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "datasets"])
    subprocess.check_call(pip_cmd + ["datasets<3.0.0"] + mirror)

import os, math, torch, gc
import torch.nn as nn
from tqdm import tqdm
from modelscope import snapshot_download
from modelscope.msdatasets import MsDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# ================= 1. 核心配置 =================
CONFIG = {
    "mode": "offset",  # 选项: "offset" 或 "standard"
    "lr": 5e-4,  # 针对 Llama-3 的微调学习率
    "model_id": "LLM-Research/Meta-Llama-3-8B-Instruct",
    "r": 8,
    "lora_alpha": 16,
    "sample_size": 2000,  # 实验样本量
    "batch_size": 2,
    "grad_accum": 8,  # 梯度累积，等效 BatchSize = 16
    "max_length": 64
}
os.environ['MODELSCOPE_CACHE'] = './cache'


# ================= 2. Offset-LoRA 算子 (支持量化设备分配) =================
class OffsetLoraLinear(nn.Module):
    def __init__(self, base_layer, r=8, lora_alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.r, self.scaling = r, lora_alpha / r
        in_f, out_f = base_layer.in_features, base_layer.out_features

        # 自动获取基础层设备，确保在 Multi-GPU 或量化加载下参数位置正确
        target_device = base_layer.weight.device

        self.lora_A = nn.Parameter(torch.zeros((r, in_f), device=target_device))
        self.lora_B = nn.Parameter(torch.zeros((out_f, r), device=target_device))
        self.register_buffer('lora_A0', torch.zeros((r, in_f), device=target_device))
        self.register_buffer('lora_B0', torch.zeros((out_f, r), device=target_device))
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化 A 为 Kaiming，B 为 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        if CONFIG["mode"] == "offset":
            # 这里的初始化决定了 Offset 模式的稳定性
            nn.init.kaiming_uniform_(self.lora_A0, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B0, a=math.sqrt(5))

            # 执行正交投影 (QR 分解)
            q_a, _ = torch.linalg.qr(self.lora_A0.T)
            self.lora_A0.data = q_a.T * (1.0 / math.sqrt(self.r))
            q_b, _ = torch.linalg.qr(self.lora_B0)
            self.lora_B0.data = q_b * (1.0 / math.sqrt(self.r))

            # 使 A, B 初始状态对齐 A0, B0，确保初始 Delta W = 0
            self.lora_A.data.copy_(self.lora_A0.data)
            self.lora_B.data.copy_(self.lora_B0.data)

    def forward(self, x):
        # 兼容量化训练的半精度计算
        dtype = x.dtype
        base_output = self.base_layer(x)

        # 计算 BA
        lora_val = (x @ self.lora_A.to(dtype).T @ self.lora_B.to(dtype).T)

        if CONFIG["mode"] == "offset":
            # 核心公式: Delta W = BA - B0A0
            lora_output = lora_val - (x @ self.lora_A0.to(dtype).T @ self.lora_B0.to(dtype).T)
        else:
            lora_output = lora_val

        return base_output + lora_output * self.scaling


# ================= 3. 模型与数据准备 =================
print(f">>> 正在下载并加载 Llama-3-8B (4-bit 量化模式)...")
model_dir = snapshot_download(CONFIG["model_id"])
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit 量化配置 (NF4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_dir,
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id
model = prepare_model_for_kbit_training(model)

# ================= 4. 动态注入 LoRA 层 =================
print(f">>> 注入目标层: q_proj (Mode: {CONFIG['mode']})")
for name, module in model.named_modules():
    if "q_proj" in name and "Linear" in str(type(module)):
        parts = name.rsplit('.', 1)
        parent = dict(model.named_modules())[parts[0]]
        module.weight.requires_grad = False
        setattr(parent, parts[1], OffsetLoraLinear(module, r=CONFIG["r"]))

# 收集需要更新的参数：LoRA 分支 + 分类头 (score)
trainable_params = [p for n, p in model.named_parameters() if "lora_" in n or "score" in n]
for p in trainable_params: p.requires_grad = True

# 加载并预处理 SST-2 数据
ms_ds = MsDataset.load('glue', subset_name='sst2', trust_remote_code=True)
train_ds = ms_ds['train'].to_hf_dataset().shuffle(seed=42).select(range(CONFIG["sample_size"]))

# ================= 5. 训练循环 =================
optimizer = torch.optim.AdamW(trainable_params, lr=CONFIG["lr"])
model.train()

losses = []
step_count = 0
pbar = tqdm(range(0, len(train_ds), CONFIG["batch_size"]))

for i in pbar:
    batch_data = train_ds[i: i + CONFIG["batch_size"]]
    inputs = tokenizer(batch_data["sentence"], truncation=True, padding="max_length",
                       max_length=CONFIG["max_length"], return_tensors="pt").to(model.device)
    labels = torch.tensor(batch_data["label"]).to(model.device)

    outputs = model(**inputs, labels=labels)
    loss = outputs.loss / CONFIG["grad_accum"]
    loss.backward()

    # 梯度累积步
    if (i // CONFIG["batch_size"] + 1) % CONFIG["grad_accum"] == 0:
        optimizer.step()
        optimizer.zero_grad()
        step_count += 1

        current_loss = round(loss.item() * CONFIG["grad_accum"], 4)
        losses.append(current_loss)
        pbar.set_description(f"Step {step_count} Loss: {current_loss:.4f}")

# ================= 6. 结果输出 =================
print("\n" + "=" * 50)
print(f"✅ 实验完成！当前模式: {CONFIG['mode']}")
print(f"模式说明: {'偏移补偿 (Offset-LoRA)' if CONFIG['mode'] == 'offset' else '标准初始化 (Standard)'}")
print("-" * 20 + " 完整 LOSS 列表 " + "-" * 20)
print(losses)
print("=" * 50)

# 清理显存
del model;
gc.collect();
torch.cuda.empty_cache()
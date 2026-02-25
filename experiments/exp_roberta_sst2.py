# 1. 安装核心库和缺失的依赖（解决 oss2, addict 等）
try:
    import modelscope
    import datasets

    # 强制修正 datasets 版本冲突
    if int(datasets.__version__.split('.')[0]) >= 3: raise ImportError
except (ImportError, ModuleNotFoundError):
    import subprocess, sys

    print(">>> 正在安装核心依赖与修正 datasets 版本...")
    pip_cmd = [sys.executable, "-m", "pip", "install"]
    mirror = ["-i", "https://mirrors.aliyun.com/pypi/simple/"]
    subprocess.check_call(pip_cmd + ["modelscope", "transformers", "peft", "accelerate", "evaluate", "addict", "oss2",
                                     "scikit-learn"] + mirror)
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "datasets"])
    subprocess.check_call(pip_cmd + ["datasets<3.0.0"] + mirror)

import os, math, torch
import torch.nn as nn
from modelscope import snapshot_download
from modelscope.msdatasets import MsDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback

# ================= 1. 配置 =================
CONFIG = {
    "mode": "offset",  # 选项: "offset" 或 "standard"
    "lr": 1e-3,  # 极限压测学习率
    "model_id": "AI-ModelScope/roberta-large",
    "r": 8,
    "lora_alpha": 16
}

os.environ['MODELSCOPE_CACHE'] = './cache'


# ================= 2. Offset-LoRA 核心类 =================
class OffsetLoraLinear(nn.Module):
    def __init__(self, base_layer, r=8, lora_alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.scaling = lora_alpha / r
        in_f, out_f = base_layer.in_features, base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros((r, in_f)))
        self.lora_B = nn.Parameter(torch.zeros((out_f, r)))
        self.register_buffer('lora_A0', torch.zeros((r, in_f)))
        self.register_buffer('lora_B0', torch.zeros((out_f, r)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if CONFIG["mode"] == "offset":
            nn.init.kaiming_uniform_(self.lora_A0, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B0, a=math.sqrt(5))
            q_a, _ = torch.linalg.qr(self.lora_A0.T)
            self.lora_A0.data = q_a.T * (1.0 / math.sqrt(self.r))
            q_b, _ = torch.linalg.qr(self.lora_B0)
            self.lora_B0.data = q_b * (1.0 / math.sqrt(self.r))
            self.lora_A.data.copy_(self.lora_A0.data)
            self.lora_B.data.copy_(self.lora_B0.data)

    def forward(self, x):
        base_output = self.base_layer(x)
        if CONFIG["mode"] == "offset":
            lora_output = (x @ self.lora_A.T @ self.lora_B.T) - (x @ self.lora_A0.T @ self.lora_B0.T)
        else:
            lora_output = (x @ self.lora_A.T @ self.lora_B.T)
        return base_output + lora_output * self.scaling


# ================= 3. 数据加载与模型注入 =================
print(">>> 正在初始化环境与模型...")
model_dir = snapshot_download(CONFIG["model_id"])
tokenizer = AutoTokenizer.from_pretrained(model_dir)
ms_ds = MsDataset.load('glue', subset_name='sst2', trust_remote_code=True)
train_ds = ms_ds['train'].to_hf_dataset()
val_ds = ms_ds['validation'].to_hf_dataset()


def tokenize_fn(ex):
    return tokenizer(ex["sentence"], truncation=True, padding="max_length", max_length=128)


tokenized_train = train_ds.map(tokenize_fn, batched=True).remove_columns(['sentence', 'idx'])
tokenized_val = val_ds.map(tokenize_fn, batched=True).remove_columns(['sentence', 'idx'])

model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)

for name, module in model.named_modules():
    if "query" in name and isinstance(module, torch.nn.Linear):
        parent_name = name.rsplit('.', 1)[0]
        child_name = name.rsplit('.', 1)[1]
        parent = dict(model.named_modules())[parent_name]
        module.weight.requires_grad = False
        setattr(parent, child_name, OffsetLoraLinear(module, r=CONFIG["r"]))


# ================= 4. 训练监控与参数设置 =================
class LossLogger(TrainerCallback):
    def __init__(self): self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs: self.losses.append(logs["loss"])


logger = LossLogger()
train_args = TrainingArguments(
    output_dir="./output_roberta_sst2",
    learning_rate=CONFIG["lr"],
    per_device_train_batch_size=32,
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    report_to="none",
    save_strategy="no", eval_strategy="no"
)

trainer = Trainer(model=model, args=train_args, train_dataset=tokenized_train, callbacks=[logger])

print(f"🚀 启动实验 | 模式: {CONFIG['mode']} | 学习率: {CONFIG['lr']}")
trainer.train()

print("\n" + "=" * 50)
print(f"实验完成！模式: {CONFIG['mode']} | Loss 序列:\n{logger.losses}")
print("=" * 50)
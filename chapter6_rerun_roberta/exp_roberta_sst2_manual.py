import argparse
import gc
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from modelscope import snapshot_download
from modelscope.msdatasets import MsDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from metrics import safe_tag, save_json, summarize_losses


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OffsetLoraLinear(nn.Module):
    def __init__(self, base_layer, mode, r=8, lora_alpha=16):
        super().__init__()
        self.base_layer = base_layer
        self.mode = mode
        self.r = r
        self.scaling = lora_alpha / r
        in_f, out_f = base_layer.in_features, base_layer.out_features
        device = base_layer.weight.device
        self.lora_A = nn.Parameter(torch.zeros((r, in_f), device=device))
        self.lora_B = nn.Parameter(torch.zeros((out_f, r), device=device))
        self.register_buffer("lora_A0", torch.zeros((r, in_f), device=device))
        self.register_buffer("lora_B0", torch.zeros((out_f, r), device=device))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        if self.mode == "offset":
            nn.init.kaiming_uniform_(self.lora_A0, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B0, a=math.sqrt(5))
            q_a, _ = torch.linalg.qr(self.lora_A0.T)
            q_b, _ = torch.linalg.qr(self.lora_B0)
            self.lora_A0.data = q_a.T * (1.0 / math.sqrt(self.r))
            self.lora_B0.data = q_b * (1.0 / math.sqrt(self.r))
            self.lora_A.data.copy_(self.lora_A0.data)
            self.lora_B.data.copy_(self.lora_B0.data)

    def forward(self, x):
        base_output = self.base_layer(x)
        lora_val = x @ self.lora_A.T @ self.lora_B.T
        if self.mode == "offset":
            lora_val = lora_val - (x @ self.lora_A0.T @ self.lora_B0.T)
        return base_output + lora_val * self.scaling


def build_model(model_id, mode, rank, alpha):
    model_dir = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2).to("cuda")

    for name, module in model.named_modules():
        if "query" in name and isinstance(module, nn.Linear):
            parts = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parts[0]]
            module.weight.requires_grad = False
            setattr(parent, parts[1], OffsetLoraLinear(module, mode=mode, r=rank, lora_alpha=alpha))

    for name, param in model.named_parameters():
        param.requires_grad = ("lora_" in name or "classifier" in name)

    return model_dir, tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offset", "standard"], required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--model-id", default="AI-ModelScope/roberta-large")
    parser.add_argument("--output-root", default="results/chapter6_rerun_roberta")
    args = parser.parse_args()

    set_seed(args.seed)
    gc.collect()
    torch.cuda.empty_cache()

    model_dir, tokenizer, model = build_model(args.model_id, args.mode, args.rank, args.alpha)
    ms_ds = MsDataset.load("glue", subset_name="sst2", trust_remote_code=True)
    train_ds = ms_ds["train"].to_hf_dataset().shuffle(seed=args.seed)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    losses = []
    model.train()
    pbar = tqdm(range(0, len(train_ds), args.batch_size), desc=f"sst2 {args.mode} lr={args.lr} seed={args.seed}")

    for batch_idx, start in enumerate(pbar):
        if args.max_steps and batch_idx >= args.max_steps:
            break
        batch = train_ds[start : start + args.batch_size]
        inputs = tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=args.max_len,
            return_tensors="pt",
        ).to("cuda")
        labels = torch.tensor(batch["label"]).to("cuda")

        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        if torch.isnan(loss):
            break
        loss.backward()
        optimizer.step()
        value = round(float(loss.item()), 6)
        losses.append(value)
        pbar.set_postfix(loss=value)

    payload = {
        "task": "roberta_sst2",
        "mode": args.mode,
        "lr": args.lr,
        "seed": args.seed,
        "model_id": args.model_id,
        "model_dir": model_dir,
        "rank": args.rank,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "max_steps": args.max_steps,
        "losses": losses,
        "metrics": summarize_losses(losses) if losses else {},
    }
    out_name = f"roberta_sst2_{args.mode}_lr{safe_tag(args.lr)}_seed{args.seed}.json"
    save_json(Path(args.output_root) / "raw" / out_name, payload)
    print(f"saved: {Path(args.output_root) / 'raw' / out_name}")


if __name__ == "__main__":
    main()


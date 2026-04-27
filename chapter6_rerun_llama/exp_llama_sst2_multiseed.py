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
from peft import prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

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
        dtype = x.dtype
        base_output = self.base_layer(x)
        lora_val = x @ self.lora_A.to(dtype).T @ self.lora_B.to(dtype).T
        if self.mode == "offset":
            lora_val = lora_val - (x @ self.lora_A0.to(dtype).T @ self.lora_B0.to(dtype).T)
        return base_output + lora_val * self.scaling


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offset", "standard"], required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--model-id", default="LLM-Research/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-root", default="results/chapter6_rerun_llama")
    args = parser.parse_args()

    set_seed(args.seed)
    gc.collect()
    torch.cuda.empty_cache()

    model_dir = snapshot_download(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    for name, module in model.named_modules():
        if "q_proj" in name and "Linear" in str(type(module)):
            parts = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parts[0]]
            module.weight.requires_grad = False
            setattr(parent, parts[1], OffsetLoraLinear(module, mode=args.mode, r=args.rank, lora_alpha=args.alpha))

    trainable = [p for n, p in model.named_parameters() if "lora_" in n or "score" in n]
    for param in trainable:
        param.requires_grad = True
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    ms_ds = MsDataset.load("glue", subset_name="sst2", trust_remote_code=True)
    train_ds = ms_ds["train"].to_hf_dataset().shuffle(seed=args.seed).select(range(args.sample_size))

    losses = []
    nan_step = None
    opt_steps = 0
    model.train()
    pbar = tqdm(range(0, len(train_ds), args.batch_size), desc=f"llama {args.mode} lr={args.lr} seed={args.seed}")

    for micro_idx, start in enumerate(pbar):
        batch = train_ds[start : start + args.batch_size]
        inputs = tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors="pt",
        ).to(model.device)
        labels = torch.tensor(batch["label"]).to(model.device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss / args.grad_accum
        if torch.isnan(loss):
            nan_step = opt_steps
            break
        loss.backward()

        if (micro_idx + 1) % args.grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            opt_steps += 1
            value = round(float(loss.item() * args.grad_accum), 6)
            losses.append(value)
            pbar.set_postfix(loss=value, opt_steps=opt_steps)
            if args.max_steps and opt_steps >= args.max_steps:
                break

    payload = {
        "task": "llama_sst2",
        "mode": args.mode,
        "lr": args.lr,
        "seed": args.seed,
        "model_id": args.model_id,
        "model_dir": model_dir,
        "rank": args.rank,
        "alpha": args.alpha,
        "sample_size": args.sample_size,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "optimizer_steps": opt_steps,
        "nan_step": nan_step,
        "losses": losses,
        "metrics": summarize_losses(losses) if losses else {},
    }
    out_name = f"llama_sst2_{args.mode}_lr{safe_tag(args.lr)}_seed{args.seed}.json"
    save_json(Path(args.output_root) / "raw" / out_name, payload)
    print(f"saved: {Path(args.output_root) / 'raw' / out_name}")


if __name__ == "__main__":
    main()


# 第六章需要重跑时的代码模板说明

这里只放思路，不在这里直接执行。

## 模板目标

重跑时不要只打印 loss，要把 loss 真正保存为 `json`。

## RoBERTa / MRPC 模板

在原脚本末尾增加：

```python
import json
from pathlib import Path

out = {
    "tag": "roberta_mrpc_standard_1e3",
    "lr": CONFIG["lr"],
    "mode": CONFIG["mode"],
    "losses": losses,
}
Path("results").mkdir(exist_ok=True)
Path("results/roberta_mrpc_standard_1e3.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
```

## Llama / SST 模板

同样处理，把 `losses` 保存到：

```python
results/llama_sst_offset_5e4.json
results/llama_sst_standard_5e4.json
```

## 推荐重跑窗口

- RoBERTa `1e-3`
  先跑到能拿到前 `30` 到 `50` 步即可
- RoBERTa `5e-4`
  先跑前 `60` 步
- Llama `5e-4`
  先跑前 `30` 到 `40` 步

## 明天如果必须重跑

优先顺序：
1. `RoBERTa 1e-3`
2. `Llama 5e-4`
3. `MRPC 5e-4`

不建议第一时间重跑全部。

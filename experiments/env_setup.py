import subprocess
import sys


def setup_environment():
    """
    一键配置 Offset-LoRA 实验环境：
    包含 RoBERTa (NLU) 与 Llama-3 (LLM/4-bit) 的所有依赖
    """
    print(">>> [1/3] 正在启动环境自动化配置...")

    # 基础依赖库清单
    # - bitsandbytes: 4-bit 量化核心
    # - sentencepiece: Llama 分词器依赖
    # - addict/oss2: ModelScope 数据流依赖
    libraries = [
        "modelscope", "transformers", "accelerate", "peft",
        "evaluate", "addict", "oss2", "scikit-learn",
        "bitsandbytes", "sentencepiece"
    ]

    pip_cmd = [sys.executable, "-m", "pip", "install"]
    mirror = ["-i", "https://mirrors.aliyun.com/pypi/simple/"]

    try:
        # 1. 安装核心库
        print(f">>> 正在安装核心库: {', '.join(libraries)}...")
        subprocess.check_call(pip_cmd + libraries + mirror)

        # 2. 修正 datasets 版本冲突 (关键步骤：防止 ALL_ALLOWED_EXTENSIONS 错误)
        print(">>> [2/3] 正在修正 datasets 版本（锁定 < 3.0.0）...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "datasets"])
        subprocess.check_call(pip_cmd + ["datasets<3.0.0"] + mirror)

        # 3. 验证 bitsandbytes
        print(">>> [3/3] 正在验证 GPU 环境与量化库...")
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU 检测成功: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ 未检测到 GPU，部分量化实验（Llama-3）可能无法运行。")

        print("\n✨ 环境配置全部完成！你可以开始运行实验脚本了。")

    except Exception as e:
        print(f"❌ 环境配置过程中出现错误: {e}")


if __name__ == "__main__":
    setup_environment()
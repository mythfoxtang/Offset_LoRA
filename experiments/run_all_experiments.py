import subprocess
import sys
import time
import os

# ================= 自动化配置 =================
# 定义需要运行的脚本列表
EXPERIMENT_SCRIPTS = [
    "experiments/exp_6_1_roberta_sst2.py",
    "experiments/exp_6_1_roberta_mrpc.py",
    "experiments/exp_6_1_llama_sst2.py"
]

MODES = ["standard", "offset"]  # 对比模式


def run_cmd(command):
    """执行系统命令并实时打印输出"""
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8'
    )
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
    return process.poll()


def main():
    print("=" * 60)
    print("🚀 Offset-LoRA 论文 6.1 节实验一键自动化运行工具")
    print("=" * 60)

    # 1. 环境预检查与安装
    print("\n[Step 1/2] 正在配置环境依赖...")
    # 引用你之前写的 env_setup.py 或直接在此安装
    setup_cmd = (
        f"{sys.executable} -m pip install modelscope transformers accelerate peft evaluate "
        f"addict oss2 scikit-learn bitsandbytes sentencepiece -i https://mirrors.aliyun.com/pypi/simple/ && "
        f"{sys.executable} -m pip uninstall -y datasets && "
        f"{sys.executable} -m pip install 'datasets<3.0.0' -i https://mirrors.aliyun.com/pypi/simple/"
    )
    run_cmd(setup_cmd)

    # 2. 循环运行实验
    print("\n[Step 2/2] 开始执行对比实验...")
    start_time = time.time()

    for script in EXPERIMENT_SCRIPTS:
        if not os.path.exists(script):
            print(f"⚠️ 跳过: 找不到脚本 {script}")
            continue

        for mode in MODES:
            print(f"\n" + "#" * 40)
            print(f"正在运行: {script} | 模式: {mode}")
            print("#" * 40)

            # 通过环境变量或修改临时文件来切换模式
            # 这里建议在脚本中读取环境变量 os.getenv('LORA_MODE')
            env = os.environ.copy()
            env["LORA_MODE"] = mode

            # 执行脚本
            exit_code = run_cmd(f"{sys.executable} {script}")

            if exit_code != 0:
                print(f"❌ 实验中断: {script} 在 {mode} 模式下运行失败。")
            else:
                print(f"✅ 完成: {script} ({mode})")

    end_time = time.time()
    total_min = (end_time - start_time) / 60
    print("\n" + "=" * 60)
    print(f"🎉 所有实验运行完毕！总耗时: {total_min:.2f} 分钟")
    print("请检查控制台输出的 Loss 序列进行绘图。")
    print("=" * 60)


if __name__ == "__main__":
    main()
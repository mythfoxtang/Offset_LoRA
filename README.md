# Offset-LoRA: Gradient-based Initialization Optimization for LLM Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

This repository contains the official implementation of the paper: **"Offset-LoRA: Gradient-based Initialization Optimization for Large Language Model Adaptation."**

## 📖 Overview

Standard LoRA (Low-Rank Adaptation) suffers from two critical geometric bottlenecks at initialization ($B=0$):

1. **Gradient Cold-Start**: Initial gradients in specific subspaces cancel out due to symmetry, leading to slow convergence in early training stages.
2. **Curvature Vacuum**: The initial sub-Hessian matrix $H_{AA} = 0$, resulting in an extremely flat and ill-conditioned loss landscape.

**Offset-LoRA** eliminates these issues by introducing an **Orthogonal Offset Compensation** mechanism:

$$\Delta W = BA - B_0A_0$$

By ensuring a non-zero, well-conditioned initial state without altering the pre-trained weights, Offset-LoRA achieves superior stability and faster convergence.

---

## ✨ Key Features

* 🚀 **Warmup-Free Training**: Eliminates the need for complex learning rate schedulers; supports high learning rates (e.g., $1 \times 10^{-3}$) from step zero.
* 📉 **Curvature Optimization**: Leverages **Eigenvalue Clustering Theory** to ensure initial Hessian eigenvalues are concentrated, minimizing the condition number $\kappa \to 1$.
* 🛡️ **Gradient Locking Effect**: Maintains dynamical stability during dimension expansion through isometric mapping properties.
* 🧩 **Plug-and-Play**: Fully compatible with the HuggingFace `transformers` and `peft` ecosystems.
## 🛠️ Installation

git clone https://github.com/YourUsername/Offset-LoRA.git
cd Offset-LoRA
pip install -r requirements.txt

---

## 🚀 Quick Start

### Reproducing Numerical Simulations (Chapter 5)
To generate the Hessian spectral distribution plots:

python simulation/hessian_analysis.py --save_plot

### LLM Fine-tuning (Chapter 6)
To run RoBERTa-large on the GLUE/MRPC task:

bash scripts/run_roberta_mrpc.sh --lr 1e-3 --method offset-lora

---

## 📊 Performance

### Convergence Speed
Offset-LoRA (Blue) exhibits immediate loss decay compared to the "lagging" start of Standard LoRA (Red), especially in the first 100 iterations.

### Hessian Spectral Distribution
* Standard LoRA: Eigenvalues are heavily spiked at zero ("Curvature Vacuum").
* Offset-LoRA: Eigenvalues are highly clustered at gamma^2, creating a "Circular Bowl" loss landscape that is ideal for first-order optimizers.

---

## 📜 Citation

If you find this work useful in your research, please cite:

@article{tang2026offsetlora,
  title={Offset-LoRA: Gradient-based Initialization Optimization for Large Language Model Adaptation},
  author={Tang, Zhaoyi},
  school={School of Mathematical Sciences, Fudan University},
  year={2026}
}

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

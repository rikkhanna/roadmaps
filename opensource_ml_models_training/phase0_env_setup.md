# Phase 0: Environment Setup (M2 Native)

## 0.1 Install Miniforge (arm64 only)

```bash
# Download native arm64 installer
brew install miniforge
# OR manual:
curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o miniforge.sh
bash miniforge.sh -b && rm miniforge.sh
source ~/.zshrc
```

## 0.2 Create Isolated Environment

```bash
conda create -n ml_poc python=3.10 -y
conda activate ml_poc

# Confirm arm64 Python (NOT x86_64)
python -c "import platform; print(platform.machine())"  # Should print: arm64
```

## 0.3 Install All Dependencies

```bash
# PyTorch — stable build with MPS support
pip install torch torchvision torchaudio

# Apple MLX — native Silicon ML framework (best for LLMs)
pip install mlx mlx-lm

# HuggingFace ecosystem
pip install transformers datasets peft accelerate sentencepiece

# Classical ML & data
pip install scikit-learn xgboost lightgbm catboost pandas numpy scipy

# Synthetic data generation
pip install sdv faker

# Experiment tracking & model registry
pip install mlflow

# Serving & UI
pip install gradio fastapi uvicorn streamlit

# Vision synthetic data
pip install diffusers pillow

# Utilities
pip install tqdm rich ipykernel jupyterlab
```

## 0.4 Verify MPS Acceleration

```python
# verify_mps.py
import torch
import platform

print(f"Platform   : {platform.machine()}")           # arm64
print(f"PyTorch    : {torch.__version__}")
print(f"MPS avail  : {torch.backends.mps.is_available()}")
print(f"MPS built  : {torch.backends.mps.is_built()}")

# Quick tensor smoke-test on MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.rand(1000, 1000, device=device)
    y = torch.rand(1000, 1000, device=device)
    z = x @ y  # Matrix multiply on GPU
    print(f"MPS matmul : OK — shape {z.shape}, device {z.device}")
else:
    print("WARNING: MPS not available — training will use CPU only")
```

```bash
python verify_mps.py
```

## 0.5 Verify MLX

```python
# verify_mlx.py
import mlx.core as mx

a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
print("MLX dot product:", mx.sum(a * b).item())  # 32.0
```

## 0.6 Shell Config (~/.zshrc)

```bash
# Append to ~/.zshrc
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Fallback unsupported ops to CPU
export TOKENIZERS_PARALLELISM=false   # Prevent HuggingFace fork deadlocks
export MLFLOW_TRACKING_URI=./mlruns   # Store experiments locally
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # Fix multiprocessing on macOS

source ~/.zshrc  # Apply changes
```

## 0.7 requirements.txt (Freeze for Reproducibility)

```text
torch>=2.2.0
torchvision>=0.17.0
torchaudio>=2.2.0
mlx>=0.12.0
mlx-lm>=0.12.0
transformers>=4.40.0
datasets>=2.19.0
peft>=0.10.0
accelerate>=0.29.0
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.3.0
pandas>=2.2.0
numpy>=1.26.0
sdv>=1.11.0
faker>=24.0.0
mlflow>=2.12.0
gradio>=4.28.0
fastapi>=0.111.0
uvicorn>=0.29.0
diffusers>=0.27.0
pillow>=10.3.0
tqdm>=4.66.0
rich>=13.7.0
```

```bash
pip install -r requirements.txt
```

# Phase 2: Model Selection & Download

## 2.1 Memory Budget — Choose the Right Model

| Model | Params | Format | VRAM Needed | M2 8GB? | M2 16GB? | M2 32GB? |
|-------|--------|--------|------------|---------|----------|----------|
| XGBoost / LightGBM | — | Native | <1 GB | ✅ | ✅ | ✅ |
| ResNet50 (fine-tune) | 25M | PyTorch fp32 | 2 GB | ✅ | ✅ | ✅ |
| YOLOv8n | 3M | PyTorch | <1 GB | ✅ | ✅ | ✅ |
| Phi-3-Mini | 3.8B | 4-bit MLX | ~2.5 GB | ✅ | ✅ | ✅ |
| Qwen2-1.5B | 1.5B | 4-bit MLX | ~1 GB | ✅ | ✅ | ✅ |
| Mistral-7B-Instruct | 7B | 4-bit MLX | ~4.5 GB | ⚠️ tight | ✅ | ✅ |
| Llama-3-8B-Instruct | 8B | 4-bit MLX | ~5 GB | ❌ | ✅ | ✅ |
| Llama-3-13B | 13B | 4-bit MLX | ~8 GB | ❌ | ⚠️ tight | ✅ |

> **Rule:** Total model memory + 3 GB (macOS) must be ≤ your total RAM.

---

## 2.2 Download Classical ML (No download needed)

```bash
# Already installed in Phase 0 — verify:
python -c "import xgboost, lightgbm, sklearn; print('Classical ML: OK')"
```

---

## 2.3 Download Vision Model (HuggingFace)

```python
# download_resnet.py
from torchvision.models import resnet50, ResNet50_Weights
import torch

# Downloads ~100MB, cached to ~/.cache/torch
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
print(f"ResNet50 params: {sum(p.numel() for p in model.parameters()):,}")
torch.save(model.state_dict(), "models/resnet50_pretrained.pth")
print("Saved to models/resnet50_pretrained.pth")
```

```bash
# OR download YOLOv8 nano (smallest, fastest)
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"  # Downloads ~6MB
```

---

## 2.4 Download LLM via MLX (Recommended for M2)

```bash
# Small/Fast — Phi-3 Mini (3.8B, 4-bit) — ~2.5GB
python -m mlx_lm.convert \
    --hf-path "microsoft/Phi-3-mini-4k-instruct" \
    --mlx-path "models/phi3-mini-4bit" \
    -q  # quantize to 4-bit

# Capable — Llama-3-8B (16GB RAM needed)
python -m mlx_lm.convert \
    --hf-path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --mlx-path "models/llama3-8b-4bit" \
    -q

# Or pull pre-quantized from community (faster, no GPU conversion needed)
huggingface-cli download mlx-community/Meta-Llama-3-8B-Instruct-4bit \
    --local-dir models/llama3-8b-4bit
```

---

## 2.5 Verify Model Loading

```python
# verify_models.py
import torch
from torchvision.models import resnet50
import mlx.core as mx
from mlx_lm import load, generate

# --- Vision ---
model = resnet50()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device).eval()
dummy = torch.rand(1, 3, 224, 224, device=device)
with torch.no_grad():
    out = model(dummy)
print(f"ResNet50 forward pass: {out.shape}")  # torch.Size([1, 1000])

# --- LLM (MLX) ---
mlx_model, tokenizer = load("models/phi3-mini-4bit")
response = generate(mlx_model, tokenizer, prompt="What is a neural network?", max_tokens=100)
print(f"\nPhi-3 response:\n{response}")
```

---

## 2.6 Project Directory Structure

```
ml_poc/
├── data/
│   ├── synthetic_tabular.csv
│   ├── faker_tabular.csv
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── images/
│       ├── defective/
│       └── normal/
├── models/
│   ├── resnet50_pretrained.pth
│   └── phi3-mini-4bit/
├── src/
│   ├── generate_tabular.py
│   ├── generate_nlp_data.py
│   ├── train_classical.py
│   ├── train_vision.py
│   ├── train_llm.py
│   ├── evaluate.py
│   └── serve.py
├── mlruns/             # MLflow experiments
├── requirements.txt
└── README.md
```

# Phase 3: Training & Fine-Tuning

## 3.1 Classical ML — XGBoost on Tabular Data

```python
# src/train_classical.py
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path

# --- Load synthetic data ---
df = pd.read_csv("data/synthetic_tabular.csv")
print(f"Dataset shape: {df.shape}")

# --- Encode categoricals ---
for col in ["device", "country"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

FEATURES = ["age", "session_length", "items_viewed", "device", "country"]
TARGET = "purchase_made"

X = df[FEATURES].values
y = df[TARGET].astype(int).values

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train XGBoost ---
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    tree_method="hist",    # Fast histogram method; no GPU needed
    random_state=42,
    early_stopping_rounds=20,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

# --- Evaluate ---
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# --- Cross-validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- Save ---
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/xgboost_purchase.pkl")
print("Model saved to models/xgboost_purchase.pkl")
```

---

## 3.2 Deep Learning — CNN Image Classifier on MPS

```python
# src/train_vision.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import time

# --- Device setup (CRITICAL for M2) ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

# --- Data transforms ---
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Datasets ---
full_dataset = datasets.ImageFolder("data/images", transform=TRAIN_TRANSFORMS)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_ds.dataset.transform = VAL_TRANSFORMS

# Keep num_workers=0 on macOS to avoid multiprocessing issues
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

print(f"Classes: {full_dataset.classes}")
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

# --- Model: ResNet50 transfer learning ---
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze backbone, train only the head
for param in model.parameters():
    param.requires_grad = False

num_classes = len(full_dataset.classes)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes),
)

model.to(device)

# --- Training setup ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

EPOCHS = 15
best_val_acc = 0.0
Path("models/checkpoints").mkdir(parents=True, exist_ok=True)

def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total

# --- Training loop ---
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss,   val_acc   = run_epoch(val_loader,   training=False)
    scheduler.step()

    elapsed = time.time() - t0
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} | "
          f"Time: {elapsed:.1f}s")

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc,
        }, "models/checkpoints/best_model.pth")
        print(f"  ✅ New best saved (val_acc={val_acc:.3f})")

print(f"\nBest Val Accuracy: {best_val_acc:.3f}")
```

---

## 3.3 LLM Fine-Tuning — LoRA via Apple MLX

### Step A: Format data

```python
# src/format_mlx_data.py
# Ensures data/train.jsonl and data/valid.jsonl exist (from Phase 1)
# MLX expects: {"text": "<full conversation string>"}
import json
from pathlib import Path

for split in ["train", "valid"]:
    path = Path(f"data/{split}.jsonl")
    lines = path.read_text().strip().split("\n")
    records = [json.loads(l) for l in lines if l.strip()]
    print(f"{split}: {len(records)} records, first keys: {list(records[0].keys())}")
    # Verify format — must have 'text' key
    assert all("text" in r for r in records), "Missing 'text' key in some records!"
print("Data format: OK")
```

### Step B: Run LoRA fine-tuning

```bash
# Fine-tune Phi-3-Mini (fits on 8GB M2)
python -m mlx_lm.lora \
    --model models/phi3-mini-4bit \
    --train \
    --data data/ \
    --iters 500 \
    --batch-size 4 \
    --lora-layers 8 \
    --grad-checkpoint \
    --save-every 100 \
    --adapter-path models/phi3-lora-adapters

# Fine-tune Llama-3-8B (requires 16GB M2)
python -m mlx_lm.lora \
    --model models/llama3-8b-4bit \
    --train \
    --data data/ \
    --iters 1000 \
    --batch-size 4 \
    --lora-layers 16 \
    --grad-checkpoint \
    --save-every 200 \
    --adapter-path models/llama3-lora-adapters
```

### Step C: Test fine-tuned model

```python
# src/test_finetuned_llm.py
from mlx_lm import load, generate

# Load base model + LoRA adapter
model, tokenizer = load(
    "models/phi3-mini-4bit",
    adapter_path="models/phi3-lora-adapters",
)

TEST_QUESTIONS = [
    "How do I detect customer churn using gradient boosting?",
    "What features should I use for fraud detection?",
    "Explain precision vs recall for anomaly detection.",
]

for q in TEST_QUESTIONS:
    prompt = f"<|user|>\n{q}\n<|assistant|>\n"
    response = generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=False)
    print(f"\nQ: {q}")
    print(f"A: {response}")
    print("-" * 60)
```

### Step D: Fuse adapters into model (for deployment)

```bash
python -m mlx_lm.fuse \
    --model models/phi3-mini-4bit \
    --adapter-path models/phi3-lora-adapters \
    --save-path models/phi3-finetuned-final \
    --de-quantize   # Optional: fuse into fp16 for faster inference
```

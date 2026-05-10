# Phase 4: Evaluation & Experiment Tracking

## 4.1 Start MLflow Tracking Server

```bash
# In a separate terminal tab — keep it running while training
mlflow ui --host 0.0.0.0 --port 5000
# Open: http://localhost:5000
```

---

## 4.2 Tabular Model Evaluation with MLflow

```python
# src/evaluate_classical.py
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

mlflow.set_experiment("tabular_purchase_prediction")

# --- Load data & model ---
df = pd.read_csv("data/synthetic_tabular.csv")
for col in ["device", "country"]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

FEATURES = ["age", "session_length", "items_viewed", "device", "country"]
X = df[FEATURES].values
y = df["purchase_made"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = joblib.load("models/xgboost_purchase.pkl")

# --- Run metrics in MLflow ---
with mlflow.start_run(run_name="xgboost_v1"):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "f1":        f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "accuracy":  (y_pred == y_test).mean(),
    }

    mlflow.log_params({
        "model_type": "XGBoost",
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "learning_rate": model.learning_rate,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "data_source": "sdv_synthetic",
    })
    mlflow.log_metrics(metrics)

    # Log model artifact
    mlflow.sklearn.log_model(model, "model", registered_model_name="purchase_predictor")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Purchase", "Purchase"]).plot(ax=ax)
    ax.set_title(f"Confusion Matrix (AUC={metrics['roc_auc']:.3f})")
    plt.tight_layout()
    plt.savefig("/tmp/confusion_matrix.png", dpi=150)
    mlflow.log_artifact("/tmp/confusion_matrix.png")

    # Feature importance
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    importances = model.feature_importances_
    ax2.barh(FEATURES, importances)
    ax2.set_title("Feature Importances")
    plt.tight_layout()
    plt.savefig("/tmp/feature_importance.png", dpi=150)
    mlflow.log_artifact("/tmp/feature_importance.png")

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No Purchase", "Purchase"]))
```

---

## 4.3 Vision Model Evaluation with MLflow

```python
# src/evaluate_vision.py
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
mlflow.set_experiment("vision_circuit_board_classifier")

# --- Load model checkpoint ---
checkpoint = torch.load("models/checkpoints/best_model.pth", map_location=device)

model = models.resnet50(weights=None)
num_classes = 2
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

# --- Val loader ---
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset = datasets.ImageFolder("data/images", transform=val_transforms)
_, val_ds = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
val_loader = DataLoader(val_ds, batch_size=16, num_workers=0)

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

with mlflow.start_run(run_name="resnet50_transfer_v1"):
    mlflow.log_params({
        "model": "ResNet50",
        "epochs_trained": checkpoint["epoch"],
        "best_val_acc": checkpoint["val_acc"],
        "data_source": "stable_diffusion_synthetic",
    })
    acc = (all_preds == all_labels).mean()
    mlflow.log_metric("val_accuracy", acc)
    mlflow.pytorch.log_model(model, "model")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=dataset.classes).plot(ax=ax)
    plt.savefig("/tmp/vision_cm.png", dpi=150)
    mlflow.log_artifact("/tmp/vision_cm.png")

    print(classification_report(all_labels, all_preds, target_names=dataset.classes))
```

---

## 4.4 LLM Evaluation — Overfitting Check & LLM-as-Judge

```python
# src/evaluate_llm.py
"""
Uses an Ollama model as an automated judge to score fine-tuned model responses.
Run: ollama serve (in background), then python src/evaluate_llm.py
"""
import json, requests
from pathlib import Path
from mlx_lm import load, generate

# Load fine-tuned model
model, tokenizer = load("models/phi3-mini-4bit", adapter_path="models/phi3-lora-adapters")

# Load validation samples
val_lines = Path("data/valid.jsonl").read_text().strip().split("\n")
val_records = [json.loads(l) for l in val_lines[:20]]  # Evaluate on 20 samples

JUDGE_PROMPT = """Rate the following answer from 1-5 (5=perfect) on accuracy and usefulness.
Question: {question}
Answer: {answer}
Respond with ONLY a JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""

def judge_response(question: str, answer: str) -> dict:
    payload = {
        "model": "llama3",
        "prompt": JUDGE_PROMPT.format(question=question, answer=answer),
        "stream": False,
        "options": {"temperature": 0.1},
    }
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
    text = r.json()["response"]
    start, end = text.find("{"), text.rfind("}") + 1
    return json.loads(text[start:end])

scores = []
for record in val_records:
    text = record["text"]
    # Extract question from <|user|> block
    question = text.split("<|user|>")[1].split("<|assistant|>")[0].strip()
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    
    answer = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
    result = judge_response(question, answer)
    scores.append(result["score"])
    print(f"Score: {result['score']}/5 — {result['reason']}")

avg_score = sum(scores) / len(scores)
print(f"\n=== Average LLM-as-Judge Score: {avg_score:.2f}/5.00 ===")

# POC threshold: if avg < 3.0, data quality is too low
if avg_score >= 3.5:
    print("✅ POC quality threshold met.")
else:
    print("⚠️  Below threshold. Improve synthetic data variety and re-train.")
```

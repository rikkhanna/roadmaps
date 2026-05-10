# Phase 3.5: Hyperparameter Tuning (Optuna)

> Run **after** a baseline training run (Phase 3) to find optimal hyperparameters automatically.

```bash
pip install optuna optuna-integration[mlflow]
```

---

## 3.5.1 XGBoost HPO with Optuna + MLflow

```python
# src/tune_xgboost.py
import optuna
import mlflow
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_experiment("xgboost_hpo")

# --- Load data ---
df = pd.read_csv("data/synthetic_tabular.csv")
for col in ["device", "country"]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

FEATURES = ["age", "session_length", "items_viewed", "device", "country"]
X = df[FEATURES].values
y = df["purchase_made"].astype(int).values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-5, 1.0, log=True),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    # Log each trial to MLflow
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_metric("cv_roc_auc_mean", scores.mean())
        mlflow.log_metric("cv_roc_auc_std", scores.std())

    return scores.mean()

# --- Run study ---
with mlflow.start_run(run_name="optuna_xgboost_hpo"):
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=50, timeout=300)  # 5 min budget

    best = study.best_params
    print(f"\n=== Best Trial ===")
    print(f"  ROC-AUC : {study.best_value:.4f}")
    print(f"  Params  : {best}")

    # Train final model with best params
    final_model = XGBClassifier(**best, use_label_encoder=False, eval_metric="logloss")
    final_model.fit(X, y)
    joblib.dump(final_model, "models/xgboost_tuned.pkl")

    mlflow.log_params(best)
    mlflow.log_metric("best_cv_roc_auc", study.best_value)
    mlflow.sklearn.log_model(final_model, "tuned_model", registered_model_name="purchase_predictor_tuned")

# Optuna visualization
fig = optuna.visualization.plot_param_importances(study)
fig.write_html("reports/optuna_param_importance.html")
print("Saved: reports/optuna_param_importance.html")
```

---

## 3.5.2 PyTorch Vision HPO with Optuna

```python
# src/tune_vision.py
import optuna
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import mlflow

optuna.logging.set_verbosity(optuna.logging.WARNING)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
mlflow.set_experiment("resnet_hpo")

def get_loaders(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder("data/images", transform=transform)
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0),
        len(ds.classes),
    )

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1) == labels).sum().item()
        total += len(labels)
    return correct / total

def eval_model(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total

def objective(trial):
    lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout      = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32])

    train_loader, val_loader, num_classes = get_loaders(batch_size)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(5):  # Short run per trial
        train_one_epoch(model, train_loader, optimizer, criterion)
        val_acc = eval_model(model, val_loader)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        best_val_acc = max(best_val_acc, val_acc)
        if device.type == "mps":
            torch.mps.empty_cache()

    return best_val_acc

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3),
)
study.optimize(objective, n_trials=15, timeout=600)

print(f"\nBest Val Acc: {study.best_value:.4f}")
print(f"Best Params:  {study.best_params}")
```

---

## 3.5.3 MLX LLM — Learning Rate Grid Search

```bash
# Quick grid search for LoRA learning rate (run each, compare MLflow)
for LR in 1e-4 5e-5 1e-5; do
  echo "=== Training with lr=$LR ==="
  python -m mlx_lm.lora \
      --model models/phi3-mini-4bit \
      --train \
      --data data/ \
      --iters 300 \
      --batch-size 4 \
      --lora-layers 8 \
      --learning-rate $LR \
      --adapter-path "models/phi3-lora-lr${LR}"
done

# Compare loss curves — check mlruns/ or printed validation loss
# Use the adapter with lowest final validation loss
```

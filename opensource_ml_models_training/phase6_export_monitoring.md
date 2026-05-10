# Phase 6: Model Export & Production Monitoring

---

## 6.1 Export to ONNX (Cross-Platform)

> ONNX is the universal interchange format — run your model in any framework or runtime.

```bash
pip install onnx onnxruntime
```

```python
# src/export_onnx.py
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from torchvision import models

device = torch.device("cpu")  # Export from CPU, not MPS (ONNX exporter limitation)

# --- Load trained model ---
model = models.resnet50(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 128), nn.ReLU(), nn.Linear(128, 2))
ckpt = torch.load("models/checkpoints/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# --- Export ---
dummy_input = torch.rand(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "models/resnet50_circuit.onnx",
    opset_version=17,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}},  # Support variable batch size
    do_constant_folding=True,
)

# --- Validate ONNX model ---
onnx_model = onnx.load("models/resnet50_circuit.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model valid ✅")
print(f"Model size: {onnx_model.ByteSize() / 1e6:.1f} MB")

# --- Inference with ONNX Runtime ---
sess = ort.InferenceSession("models/resnet50_circuit.onnx")
result = sess.run(["logits"], {"image": dummy_input.numpy()})
print(f"ONNX inference output shape: {result[0].shape}")

# XGBoost → ONNX
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

xgb_pipeline = joblib.load("models/full_pipeline.pkl")
initial_type = [("float_input", FloatTensorType([None, 5]))]
xgb_onnx = convert_sklearn(xgb_pipeline, initial_types=initial_type)
with open("models/xgboost_pipeline.onnx", "wb") as f:
    f.write(xgb_onnx.SerializeToString())
print("XGBoost pipeline exported to ONNX ✅")
```

---

## 6.2 Export to Apple CoreML (Native M2/iOS Deployment)

> CoreML runs natively on the Neural Engine of M-series chips — fastest possible inference on Apple devices.

```bash
pip install coremltools
```

```python
# src/export_coreml.py
import coremltools as ct
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

# --- Load model ---
model = models.resnet50(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 128), nn.ReLU(), nn.Linear(128, 2))
ckpt = torch.load("models/checkpoints/best_model.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# --- Trace with TorchScript (required for CoreML conversion) ---
dummy = torch.rand(1, 3, 224, 224)
traced = torch.jit.trace(model, dummy)

# --- Convert to CoreML ---
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(
        name="image",
        shape=dummy.shape,
        scale=1 / 255.0,
        bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    )],
    classifier_config=ct.ClassifierConfig(["Normal", "Defective"]),
    compute_units=ct.ComputeUnit.ALL,  # Uses Neural Engine + GPU + CPU
    minimum_deployment_target=ct.target.macOS13,
)

# Add metadata
mlmodel.short_description = "Circuit board defect classifier (POC)"
mlmodel.author = "ML POC"
mlmodel.version = "1.0"

mlmodel.save("models/CircuitBoardClassifier.mlpackage")
print("CoreML model saved ✅")

# --- Verify inference on M2 ---
test_input = {"image": (np.random.rand(1, 3, 224, 224) * 255).astype(np.uint8)}
pred = mlmodel.predict(test_input)
print(f"CoreML prediction: {pred}")

# XGBoost → CoreML via ONNX bridge
import onnxmltools
# (First export XGBoost to ONNX via step 6.1, then convert)
# mlmodel_xgb = ct.convert("models/xgboost_pipeline.onnx")
# mlmodel_xgb.save("models/PurchasePredictor.mlpackage")
```

```bash
# Verify the .mlpackage in Xcode or Quick Look
open models/CircuitBoardClassifier.mlpackage
```

---

## 6.3 Drift Detection & Model Monitoring (Evidently AI)

> After your POC starts receiving real (or new synthetic) data, models can silently degrade.
> Evidently AI detects this automatically.

```bash
pip install evidently
```

```python
# src/monitor_drift.py
"""
Simulate production monitoring:
- Reference dataset = training data
- Current dataset   = new batch of data (simulated)
Run weekly or after each new batch of predictions.
"""
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from pathlib import Path

# --- Reference data (what model was trained on) ---
reference = pd.read_csv("data/synthetic_tabular.csv")

# --- Simulate "production" data arriving 3 months later ---
# In real POC, replace this with actual new data coming into your system
np.random.seed(99)
current = reference.copy()
# Simulate drift: older users, longer sessions (distribution shift)
current["age"]            += np.random.normal(5, 3, len(current))   # Users aged up
current["session_length"] += np.random.normal(60, 30, len(current)) # Sessions longer
current["age"] = current["age"].clip(18, 90)

# --- Columns to monitor ---
FEATURE_COLS = ["age", "session_length", "items_viewed"]
TARGET_COL   = "purchase_made"

# --- Data Drift Report ---
drift_report = Report(metrics=[
    DatasetDriftMetric(),
    ColumnDriftMetric(column_name="age"),
    ColumnDriftMetric(column_name="session_length"),
    ColumnDriftMetric(column_name="items_viewed"),
])

drift_report.run(
    reference_data=reference[FEATURE_COLS + [TARGET_COL]],
    current_data=current[FEATURE_COLS + [TARGET_COL]],
)

Path("reports").mkdir(exist_ok=True)
drift_report.save_html("reports/drift_report.html")
print("Drift report saved → reports/drift_report.html")

# --- Programmatic check (for CI/CD or alerting) ---
drift_result = drift_report.as_dict()
dataset_drift = drift_result["metrics"][0]["result"]["dataset_drift"]
drifted_cols  = drift_result["metrics"][0]["result"]["number_of_drifted_columns"]
total_cols    = drift_result["metrics"][0]["result"]["number_of_columns"]

print(f"\n=== Drift Summary ===")
print(f"  Dataset drift detected : {dataset_drift}")
print(f"  Drifted columns        : {drifted_cols}/{total_cols}")

if dataset_drift:
    print("\n⚠️  ALERT: Significant data drift detected!")
    print("   Action: Collect new labels → retrain model → re-evaluate.")
else:
    print("\n✅ No significant drift. Model is still valid.")
```

```bash
python src/monitor_drift.py
open reports/drift_report.html
```

---

## 6.4 When to Retrain — Decision Checklist

| Signal | Threshold | Action |
|--------|-----------|--------|
| Dataset drift (Evidently) | >30% columns drifted | Retrain with new data |
| Prediction confidence drops | Avg prob < 0.55 | Investigate & retrain |
| Classification accuracy drops | >5% decline vs baseline | Retrain |
| New categories appear | Unknown device/country | Update encoder & retrain |
| Time elapsed | >30 days on fresh POC | Validate with new synthetic batch |

---

## 6.5 Full End-to-End Run Script

```bash
#!/bin/bash
# run_poc.sh — Run entire POC pipeline in order
set -e

echo "=== Phase 0: Environment ===" && python verify_mps.py
echo "=== Phase 1: Generate Data ===" && python src/generate_tabular.py && python src/generate_ctgan.py
echo "=== Phase 1.5: Validate Data ===" && python src/validate_data.py
echo "=== Phase 3: Train ===" && python src/train_classical.py && python src/train_vision.py
echo "=== Phase 3.5: Tune ===" && python src/tune_xgboost.py
echo "=== Phase 4: Evaluate ===" && python src/evaluate_classical.py && python src/evaluate_vision.py
echo "=== Phase 6: Export ===" && python src/export_onnx.py && python src/export_coreml.py
echo "=== Phase 6: Monitor ===" && python src/monitor_drift.py
echo "=== Phase 5: Serve ===" && python src/serve_gradio.py
```

```bash
chmod +x run_poc.sh
./run_poc.sh
```

# Phase 1.5: Data Validation & Preprocessing Pipeline

> Run this **after** Phase 1 (data generation) and **before** Phase 3 (training).
> Garbage in → garbage model. Always validate synthetic data quality first.

---

## 1.5.1 Synthetic Data Quality Report (ydata-profiling)

```bash
pip install ydata-profiling
```

```python
# src/validate_data.py
import pandas as pd
from ydata_profiling import ProfileReport
from pathlib import Path

df = pd.read_csv("data/synthetic_tabular.csv")

# Auto-generate a full HTML report with distributions, correlations, missing values
profile = ProfileReport(
    df,
    title="Synthetic Data Quality Report",
    explorative=True,
    correlations={"auto": {"calculate": True}},
)
Path("reports").mkdir(exist_ok=True)
profile.to_file("reports/data_quality_report.html")
print("Report saved → reports/data_quality_report.html")

# Quick sanity checks (fail fast)
assert df.isnull().sum().sum() == 0, "❌ Null values found!"
assert df["purchase_made"].nunique() == 2, "❌ Target column issue!"
assert len(df) >= 1000, "❌ Dataset too small for training!"
assert df["age"].between(18, 100).all(), "❌ Age out of valid range!"

print("✅ All data quality checks passed.")
print(df.describe().T[["mean", "std", "min", "max"]])
```

```bash
python src/validate_data.py
open reports/data_quality_report.html   # View in browser
```

---

## 1.5.2 Statistical Similarity Check (Synthetic vs. Schema)

```python
# src/check_distribution.py
"""Compare synthetic data distributions against expected schema."""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load synthetic data
synth = pd.read_csv("data/synthetic_tabular.csv")

# Define expected distributions (from domain knowledge)
EXPECTED = {
    "age":            {"mean": 35, "std": 12, "min": 18, "max": 70},
    "session_length": {"mean": 300, "std": 200, "min": 0,  "max": 1800},
    "items_viewed":   {"mean": 7,   "std": 5,  "min": 0,  "max": 50},
}

print("=== Distribution Checks ===")
for col, exp in EXPECTED.items():
    actual_mean = synth[col].mean()
    actual_std  = synth[col].std()
    # Allow 20% deviation from expected mean
    drift = abs(actual_mean - exp["mean"]) / exp["mean"]
    status = "✅" if drift < 0.2 else "⚠️ "
    print(f"{status} {col:20s} | Expected mean={exp['mean']:.1f}, Got mean={actual_mean:.1f}, drift={drift:.1%}")

# Purchase class balance — warn if heavily imbalanced
balance = synth["purchase_made"].value_counts(normalize=True)
print(f"\nClass balance: {balance.to_dict()}")
if balance.min() < 0.15:
    print("⚠️  Severe class imbalance detected — consider using class_weight='balanced'")
else:
    print("✅ Class balance acceptable")

# Correlation check — catch spurious correlations in synthetic data
numeric_cols = ["age", "session_length", "items_viewed"]
corr = synth[numeric_cols].corr()
print("\n=== Correlation Matrix ===")
print(corr.round(3))

# Kolmogorov-Smirnov test for normality on key features
print("\n=== Normality Tests (KS) ===")
for col in numeric_cols:
    stat, p = stats.kstest(synth[col], "norm", args=(synth[col].mean(), synth[col].std()))
    print(f"  {col:20s} | KS stat={stat:.3f}, p={p:.4f}")
```

---

## 1.5.3 Reproducible Preprocessing Pipeline

```python
# src/build_pipeline.py
"""
Build a sklearn Pipeline that bundles preprocessing + model together.
This ensures the same transforms are applied at both train and inference time.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("data/synthetic_tabular.csv")

NUMERIC_FEATURES     = ["age", "session_length", "items_viewed"]
CATEGORICAL_FEATURES = ["device", "country"]
TARGET               = "purchase_made"

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Define preprocessing for each column type ---
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Handle missing values
    ("scaler",  StandardScaler()),                   # Normalize
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer,     NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# --- Full pipeline: preprocess → model ---
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )),
])

full_pipeline.fit(X_train, y_train)
acc = full_pipeline.score(X_test, y_test)
print(f"Pipeline accuracy: {acc:.4f}")

# Save entire pipeline (preprocessor + model bundled together)
joblib.dump(full_pipeline, "models/full_pipeline.pkl")
print("Saved: models/full_pipeline.pkl")

# --- Inference test (raw input, no manual encoding needed) ---
sample = pd.DataFrame([{
    "age": 32, "session_length": 450, "items_viewed": 10,
    "device": "mobile", "country": "IN",
}])
prob = full_pipeline.predict_proba(sample)[0][1]
print(f"\nSample inference: P(purchase)={prob:.1%}")
```

---

## 1.5.4 GAN-Based Tabular Data (CTGAN) — High-Fidelity Alternative

> Relevant to your existing `gan.py`. CTGAN learns the *actual statistical structure* 
> of your seed data, producing more realistic synthetic rows than simple Gaussian methods.

```bash
pip install ctgan
```

```python
# src/generate_ctgan.py
import pandas as pd
from ctgan import CTGAN
from pathlib import Path

# Seed data — use a small real sample or schema-based bootstrap
seed_data = pd.DataFrame({
    "age":            [25, 34, 45, 22, 60, 31, 28, 52, 41, 38],
    "session_length": [120, 340, 50, 780, 200, 310, 95, 420, 180, 650],
    "items_viewed":   [3, 12, 1, 25, 8, 7, 2, 15, 6, 20],
    "cart_value":     [0, 120.5, 0, 340.0, 89.0, 55.0, 0, 210.0, 0, 480.0],
    "purchase_made":  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "device":         ["mobile","desktop","tablet","mobile","desktop"] * 2,
    "country":        ["IN","US","DE","IN","US"] * 2,
})

# Columns that are discrete/categorical
DISCRETE_COLUMNS = ["purchase_made", "device", "country"]

# Train CTGAN
model = CTGAN(epochs=300, verbose=True)
model.fit(seed_data, discrete_columns=DISCRETE_COLUMNS)

# Generate high-fidelity synthetic data
synthetic = model.sample(5000)

Path("data").mkdir(exist_ok=True)
synthetic.to_csv("data/ctgan_synthetic.csv", index=False)

print(f"Generated {len(synthetic)} rows via CTGAN")
print("\nOriginal class balance:")
print(seed_data["purchase_made"].value_counts(normalize=True))
print("\nSynthetic class balance:")
print(synthetic["purchase_made"].value_counts(normalize=True))
```

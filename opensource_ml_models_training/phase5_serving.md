# Phase 5: POC Serving & UI Demonstration

## 5.1 Gradio — All-in-One Demo UI

```python
# src/serve_gradio.py
"""
Interactive POC demo combining all 3 models.
Run: python src/serve_gradio.py
Open: http://localhost:7860
"""
import gradio as gr
import torch
import joblib
import numpy as np
import pandas as pd
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from mlx_lm import load as mlx_load, generate as mlx_generate

# ─── Load all models once at startup ──────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. XGBoost tabular model
xgb_model = joblib.load("models/xgboost_purchase.pkl")

# 2. ResNet50 vision model
resnet = models.resnet50(weights=None)
resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(resnet.fc.in_features, 128), nn.ReLU(), nn.Linear(128, 2))
ckpt = torch.load("models/checkpoints/best_model.pth", map_location=device)
resnet.load_state_dict(ckpt["model_state_dict"])
resnet.to(device).eval()

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 3. Fine-tuned LLM
llm_model, llm_tokenizer = mlx_load(
    "models/phi3-mini-4bit",
    adapter_path="models/phi3-lora-adapters",
)

CLASS_NAMES = ["Normal", "Defective"]
DEVICE_MAP = {"Mobile": 0, "Desktop": 1, "Tablet": 2}
COUNTRY_MAP = {"India": 0, "USA": 1, "Germany": 2}

# ─── Prediction functions ──────────────────────────────────────────────────────

def predict_purchase(age, session_length, items_viewed, device_type, country):
    features = np.array([[age, session_length, items_viewed,
                          DEVICE_MAP[device_type], COUNTRY_MAP[country]]])
    prob = xgb_model.predict_proba(features)[0][1]
    label = "🛒 Will Purchase" if prob > 0.5 else "❌ Won't Purchase"
    return f"{label}\n\nConfidence: {prob:.1%}"

def predict_image(image):
    if image is None:
        return "Please upload an image."
    img_tensor = IMG_TRANSFORM(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = resnet(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = probs.argmax()
    return f"**{CLASS_NAMES[idx]}** — Confidence: {probs[idx]:.1%}\n\n" + \
           "\n".join([f"{c}: {p:.1%}" for c, p in zip(CLASS_NAMES, probs)])

def ask_llm(question):
    if not question.strip():
        return "Please enter a question."
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    return mlx_generate(llm_model, llm_tokenizer, prompt=prompt, max_tokens=400, verbose=False)

# ─── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="M2 ML POC Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 M2 ML POC Demo\nThree open-source models trained locally on synthetic data.")

    with gr.Tab("📊 Purchase Predictor (XGBoost)"):
        with gr.Row():
            age = gr.Slider(18, 70, value=30, label="Age")
            session = gr.Slider(0, 1800, value=300, label="Session Length (s)")
        with gr.Row():
            items = gr.Slider(0, 50, value=8, step=1, label="Items Viewed")
            dev = gr.Dropdown(["Mobile", "Desktop", "Tablet"], value="Mobile", label="Device")
            country = gr.Dropdown(["India", "USA", "Germany"], value="India", label="Country")
        predict_btn = gr.Button("Predict", variant="primary")
        output_text = gr.Textbox(label="Prediction")
        predict_btn.click(predict_purchase, [age, session, items, dev, country], output_text)

    with gr.Tab("🔍 Circuit Board Classifier (ResNet50)"):
        with gr.Row():
            img_input = gr.Image(label="Upload Circuit Board Image")
            img_output = gr.Markdown(label="Result")
        gr.Button("Classify", variant="primary").click(predict_image, img_input, img_output)

    with gr.Tab("💬 ML Expert Assistant (Fine-tuned LLM)"):
        question_box = gr.Textbox(label="Ask an ML question", lines=3,
                                   placeholder="How do I handle class imbalance in fraud detection?")
        answer_box = gr.Textbox(label="Answer", lines=10)
        gr.Button("Ask", variant="primary").click(ask_llm, question_box, answer_box)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

```bash
python src/serve_gradio.py
# Open http://localhost:7860
```

---

## 5.2 FastAPI — REST API Endpoint

```python
# src/serve_api.py
"""
Production-style REST API for the tabular model.
Run: uvicorn src.serve_api:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib, numpy as np, time

app = FastAPI(title="ML POC API", version="1.0.0", description="Purchase Prediction POC")
model = joblib.load("models/xgboost_purchase.pkl")

DEVICE_MAP  = {"mobile": 0, "desktop": 1, "tablet": 2}
COUNTRY_MAP = {"IN": 0, "US": 1, "DE": 2}

class PredictRequest(BaseModel):
    age:            int   = Field(..., ge=18, le=100, example=32)
    session_length: int   = Field(..., ge=0,  le=7200, example=320)
    items_viewed:   int   = Field(..., ge=0,  le=500,  example=8)
    device:         Literal["mobile", "desktop", "tablet"] = "mobile"
    country:        Literal["IN", "US", "DE"] = "IN"

class PredictResponse(BaseModel):
    prediction:   int
    probability:  float
    label:        str
    latency_ms:   float

@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost_purchase_predictor"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    features = np.array([[
        req.age,
        req.session_length,
        req.items_viewed,
        DEVICE_MAP[req.device],
        COUNTRY_MAP[req.country],
    ]])
    prob = float(model.predict_proba(features)[0][1])
    pred = int(prob >= 0.5)
    latency = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        prediction=pred,
        probability=round(prob, 4),
        label="Purchase" if pred else "No Purchase",
        latency_ms=round(latency, 2),
    )
```

```bash
# Start server
uvicorn src.serve_api:app --host 0.0.0.0 --port 8000 --reload

# Test with curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 32, "session_length": 500, "items_viewed": 12, "device": "mobile", "country": "IN"}'

# Interactive docs
open http://localhost:8000/docs
```

---

## 5.3 Streamlit — Analytics Dashboard

```python
# src/serve_streamlit.py
# Run: streamlit run src/serve_streamlit.py
import streamlit as st
import pandas as pd
import joblib, numpy as np

st.set_page_config(page_title="ML POC Dashboard", page_icon="🧠", layout="wide")
st.title("🧠 ML POC — Purchase Prediction Dashboard")

model = joblib.load("models/xgboost_purchase.pkl")

# Sidebar controls
st.sidebar.header("Prediction Inputs")
age     = st.sidebar.slider("Age", 18, 70, 30)
session = st.sidebar.slider("Session Length (s)", 0, 1800, 300)
items   = st.sidebar.slider("Items Viewed", 0, 50, 8)
device  = st.sidebar.selectbox("Device", ["mobile", "desktop", "tablet"])
country = st.sidebar.selectbox("Country", ["IN", "US", "DE"])

feat = np.array([[age, session, items, {"mobile":0,"desktop":1,"tablet":2}[device], {"IN":0,"US":1,"DE":2}[country]]])
prob = float(model.predict_proba(feat)[0][1])

col1, col2, col3 = st.columns(3)
col1.metric("Purchase Probability", f"{prob:.1%}")
col2.metric("Prediction", "✅ Buy" if prob > 0.5 else "❌ No Buy")
col3.metric("Confidence", "High" if abs(prob - 0.5) > 0.3 else "Low")

# Batch analysis on synthetic data
st.subheader("Batch Analysis on Synthetic Dataset")
df = pd.read_csv("data/synthetic_tabular.csv")
st.write(f"Dataset: {len(df):,} rows")
st.bar_chart(df["purchase_made"].value_counts().rename({0: "No Purchase", 1: "Purchase"}))
st.dataframe(df.head(50), use_container_width=True)
```

```bash
streamlit run src/serve_streamlit.py
# Opens http://localhost:8501 automatically
```

---

## 5.4 Troubleshooting — M2 Specific Issues

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| `NotImplementedError: MPS backend` | Op not yet in MPS | `export PYTORCH_ENABLE_MPS_FALLBACK=1` |
| Memory pressure (yellow/red) | Batch size too large | Reduce batch size to 4 or 8 |
| `UserWarning: resource_tracker` | macOS multiprocessing bug | Set `num_workers=0` in DataLoader |
| Training stops after 1 epoch | MPS OOM crash | Add `torch.mps.empty_cache()` after each epoch |
| MLX `nan` loss | Learning rate too high | Reduce `--learning-rate` to `1e-5` |
| Ollama timeout | Model loading | Wait 30s after `ollama serve` before requests |

```python
# Add after each training epoch to free MPS memory
if device.type == "mps":
    torch.mps.empty_cache()
```

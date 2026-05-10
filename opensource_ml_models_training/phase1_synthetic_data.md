# Phase 1: Synthetic Data Generation

## 1.1 Tabular Data — SDV (Statistical Schema-Aware)

```python
# generate_tabular.py
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# --- Step 1: Define a real-world-like seed schema ---
seed_data = pd.DataFrame({
    "user_id":        range(1, 101),
    "age":            [25, 34, 45, 22, 60] * 20,
    "session_length": [120, 340, 50, 780, 200] * 20,   # seconds
    "items_viewed":   [3, 12, 1, 25, 8] * 20,
    "purchase_made":  [0, 1, 0, 1, 0] * 20,            # target label
    "device":         ["mobile", "desktop", "tablet", "mobile", "desktop"] * 20,
    "country":        ["IN", "US", "DE", "IN", "US"] * 20,
})

# --- Step 2: Auto-detect metadata ---
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(seed_data)
metadata.update_column("user_id", sdtype="id")
metadata.update_column("purchase_made", sdtype="categorical")
metadata.set_primary_key("user_id")

# --- Step 3: Fit and generate ---
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(seed_data)

synthetic_df = synthesizer.sample(num_rows=5000)
synthetic_df.to_csv("data/synthetic_tabular.csv", index=False)

print(f"Generated {len(synthetic_df)} rows")
print(synthetic_df.head())
print("\nClass balance:")
print(synthetic_df["purchase_made"].value_counts(normalize=True))
```

## 1.2 Tabular Data — Faker (Schema-Custom)

```python
# generate_with_faker.py
import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()
random.seed(42)
np.random.seed(42)

def generate_row():
    age = random.randint(18, 65)
    return {
        "user_id":         fake.uuid4(),
        "name":            fake.name(),
        "age":             age,
        "email":           fake.email(),
        "signup_date":     fake.date_between(start_date="-2y", end_date="today"),
        "country":         fake.country_code(),
        "session_length":  max(10, int(np.random.exponential(scale=300))),
        "items_viewed":    np.random.poisson(lam=7),
        "cart_value":      round(max(0, np.random.normal(loc=80, scale=40)), 2),
        "purchase_made":   1 if random.random() < 0.35 else 0,
        "device":          random.choice(["mobile", "desktop", "tablet"]),
    }

df = pd.DataFrame([generate_row() for _ in range(10_000)])
df.to_csv("data/faker_tabular.csv", index=False)
print(df.describe())
```

## 1.3 NLP Instruction Data — Via Ollama (Local LLM)

```bash
# Install Ollama first
brew install ollama
ollama pull llama3          # ~4.7GB
ollama serve &              # Start in background
```

```python
# generate_nlp_data.py
import json, requests, time
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
OUTPUT_FILE = Path("data/train.jsonl")
OUTPUT_FILE.parent.mkdir(exist_ok=True)

DOMAIN_TOPICS = [
    "customer churn prediction",
    "fraud detection in banking",
    "product recommendation systems",
    "anomaly detection in sensor data",
    "sentiment analysis of product reviews",
]

PROMPT_TEMPLATE = """You are an ML expert. Generate a realistic Q&A pair about: {topic}.

Respond ONLY in valid JSON with this exact structure:
{{"instruction": "<specific ML question>", "response": "<detailed expert answer>"}}

Make the question technical and the answer practical, covering implementation details."""

def generate_qa(topic: str) -> dict | None:
    payload = {
        "model": "llama3",
        "prompt": PROMPT_TEMPLATE.format(topic=topic),
        "stream": False,
        "options": {"temperature": 0.8, "num_predict": 512},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        text = r.json()["response"].strip()
        # Parse the JSON response from the model
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception as e:
        print(f"  Error: {e}")
        return None

# Generate 200 samples (40 per topic)
with open(OUTPUT_FILE, "w") as f:
    for topic in DOMAIN_TOPICS:
        print(f"\nGenerating for: {topic}")
        for i in range(40):
            result = generate_qa(topic)
            if result:
                # MLX/HuggingFace chat format
                record = {
                    "text": f"<|user|>\n{result['instruction']}\n<|assistant|>\n{result['response']}"
                }
                f.write(json.dumps(record) + "\n")
                print(f"  [{i+1}/40] OK", end="\r")
            time.sleep(0.2)

print(f"\nSaved to {OUTPUT_FILE}")
```

```bash
# Split into train/valid
python -c "
import json, random
from pathlib import Path

lines = Path('data/train.jsonl').read_text().strip().split('\n')
random.shuffle(lines)
split = int(len(lines) * 0.9)

Path('data/train.jsonl').write_text('\n'.join(lines[:split]))
Path('data/valid.jsonl').write_text('\n'.join(lines[split:]))
print(f'Train: {split}, Valid: {len(lines)-split}')
"
```

## 1.4 Computer Vision Synthetic Data — Stable Diffusion (MPS)

```python
# generate_vision_data.py
import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
from tqdm import tqdm

# Choose a lightweight model for speed
MODEL_ID = "runwayml/stable-diffusion-v1-5"
device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,   # Use float16 to halve memory usage
    safety_checker=None,
)
pipe = pipe.to(device)
pipe.enable_attention_slicing()  # Reduces peak memory on M2

# Define classes for a binary classification POC
CLASSES = {
    "defective": "macro photo of a defective circuit board with burn marks, high resolution",
    "normal":    "macro photo of a clean, pristine circuit board, high resolution",
}

for label, prompt in CLASSES.items():
    out_dir = Path(f"data/images/{label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating '{label}' images...")
    for i in tqdm(range(50)):
        image = pipe(
            prompt,
            negative_prompt="blurry, low quality, cartoon",
            num_inference_steps=25,   # Fewer steps = faster, good enough for POC
            guidance_scale=7.5,
            height=512, width=512,
        ).images[0]
        image.save(out_dir / f"{label}_{i:03d}.png")

print("Done. Structure: data/images/{defective,normal}/")
```

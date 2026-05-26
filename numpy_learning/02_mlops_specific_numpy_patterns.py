# pyrefly: ignore [missing-import]
import numpy as np

# Creating arrays — know all three
X = np.array([[1.0, 2.0], [3.0, 4.0]]) 

# Min-max normalization (manual, so you understand it)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)  # ε avoids div/0

# Reproducible train/val/test split
np.random.seed(42)
idx = np.random.permutation(len(X))
train_idx = idx[:int(0.7*len(X))]
val_idx   = idx[int(0.7*len(X)):int(0.85*len(X))]

# Save / load arrays (fast — use for processed features)
np.save('features.npy', X_norm)
X_loaded = np.load('features.npy')

# Check for NaN/Inf — always do this before training
np.isnan(X).any()   # True = you have NaN values
np.isinf(X).any()   # True = you have Inf values
np.nan_to_num(X, nan=0.0, posinf=1e6)  # safe replacement
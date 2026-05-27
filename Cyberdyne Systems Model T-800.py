import numpy as np
import pandas as pd
import torch as pt
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Configure

DATA_PATH = Path(r"dummy path")  # Replace with actual path 

LSOA_COL = "LSOA code"
MONTH_COL = "Month"

BATCH_SIZE = 1024
LR = 1e-3
EPOCHS = 50

device = "cuda" if pt.cuda.is_available() else "cpu"

# Load data

df = pd.read_parquet(DATA_PATH)

df[MONTH_COL] = pd.to_datetime(df[MONTH_COL])
df = df.sort_values([LSOA_COL, MONTH_COL]).reset_index(drop=True)

# Target is next month crime 

df["next_month"] = df.groupby(LSOA_COL)[MONTH_COL].shift(-1)
df["crime_next_month"] = df.groupby(LSOA_COL)["crime_count"].shift(-1)

# ensure consecutive months
df["expected_next"] = df[MONTH_COL] + pd.DateOffset(months=1)
df = df[df["next_month"] == df["expected_next"]].copy()

df["target_month"] = df["next_month"]

df = df.dropna(subset=["crime_next_month"])

# Features of the file

candidate_features = [
    "crime_count",
    "crime_1m_ago",
    "crime_3m_ago",
    "crime_6m_ago",
    "yearly_avg",
    "neighbor_crime_count",
    "neighbor_1m_ago",
    "neighbor_3m_ago",
    "neighbor_6m_ago",
]

feature_cols = [c for c in candidate_features if c in df.columns]

if len(feature_cols) == 0:
    raise ValueError("No feature columns found. Check candidate_features against df.columns.")

print("Using features:", feature_cols)

X = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
y_count = df["crime_next_month"].to_numpy(dtype=np.float32)

# Train / Validation / Test split based on time

target_months = pd.to_datetime(df["target_month"])

train_mask = target_months < pd.Timestamp("2025-01-01")
val_mask = (target_months >= "2025-01-01") & (target_months < "2025-07-01")
test_mask = target_months >= "2025-07-01"

X_train, y_train = X[train_mask], y_count[train_mask]
X_val, y_val = X[val_mask], y_count[val_mask]
X_test, y_test = X[test_mask], y_count[test_mask]

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# We standardize things to make training easier

mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True)
std = np.where(std == 0, 1.0, std)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Actual model

class CrimeRiskNetwork(pt.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = pt.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Train function using MSE on log-counts

def train_model():
    model = CrimeRiskNetwork(X_train.shape[1]).to(device)

    loss_fn = pt.nn.MSELoss()

    y_train_t = pt.tensor(np.log1p(y_train), dtype=pt.float32)
    y_val_t = pt.tensor(np.log1p(y_val), dtype=pt.float32)

    optimizer = pt.optim.AdamW(model.parameters(), lr=LR)

    train_ds = TensorDataset(
        pt.tensor(X_train, dtype=pt.float32),
        y_train_t
    )

    val_ds = TensorDataset(
        pt.tensor(X_val, dtype=pt.float32),
        y_val_t
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_val = float("inf")

    for epoch in range(EPOCHS):
        model.train()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).view(-1, 1)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        total = 0

        with pt.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).view(-1, 1)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                total += loss.item() * xb.size(0)

        val_loss = total / len(val_loader.dataset)

        if epoch % 10 == 0:
            print(f"epoch {epoch} | val loss {val_loss:.4f}")

        best_val = min(best_val, val_loss)

    return model, best_val

# Train model

print("\nTraining MSE-log model...")
model, val_loss = train_model()

print(f"\nFinal validation loss: {val_loss:.4f}")

# Test evaluation

model.eval()

X_test_t = pt.tensor(X_test, dtype=pt.float32).to(device)

with pt.no_grad():
    raw_preds = model(X_test_t).cpu().numpy().flatten()

# Convert predicted log-counts back to predicted crime counts
preds = np.expm1(raw_preds)

# Prevent negative predicted counts
preds = np.maximum(preds, 0)

mae = np.mean(np.abs(preds - y_test))
rmse = np.sqrt(np.mean((preds - y_test) ** 2))

print(f"\nTest MAE: {mae:.3f}")
print(f"Test RMSE: {rmse:.3f}")

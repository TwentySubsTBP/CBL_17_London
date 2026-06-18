import numpy as np
import pandas as pd
import torch as pt
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Configure
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = PROJECT_ROOT / "scripts" / "crime_data_with_force_stop_search.parquet"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "model_predictions_by_lsoa_month.csv"
EXPOSURE_PATH = PROJECT_ROOT / "scripts" / "estimated_exposure_lsoa_month_clean.parquet"

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

# Load exposure data
df_exposure = pd.read_parquet(EXPOSURE_PATH)

df_exposure = df_exposure.rename(columns={
    "LSOA21CD": LSOA_COL,
    "Month": MONTH_COL
})

# FIX: period-safe conversion
if isinstance(df_exposure[MONTH_COL].dtype, pd.PeriodDtype):
    df_exposure[MONTH_COL] = df_exposure[MONTH_COL].dt.to_timestamp()
else:
    df_exposure[MONTH_COL] = pd.to_datetime(df_exposure[MONTH_COL])

df = df.merge(
    df_exposure[[LSOA_COL, MONTH_COL, "downweight"]],
    on=[LSOA_COL, MONTH_COL],
    how="left"
)

df["downweight"] = df["downweight"].fillna(1.0)

# Target
df["next_month"] = df.groupby(LSOA_COL)[MONTH_COL].shift(-1)
df["crime_next_month"] = df.groupby(LSOA_COL)["crime_count"].shift(-1)

df["expected_next"] = df[MONTH_COL] + pd.DateOffset(months=1)
df = df[df["next_month"] == df["expected_next"]].copy()

df["target_month"] = df["next_month"]
df = df.dropna(subset=["crime_next_month"])

# Features
df["month_num"] = df[MONTH_COL].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
df["quarter"] = df[MONTH_COL].dt.quarter
df["year"] = df[MONTH_COL].dt.year

df["trend_1m"] = df["crime_count"] - df["crime_1m_ago"]
df["trend_3m"] = df["crime_count"] - df["crime_3m_ago"]
df["trend_6m"] = df["crime_count"] - df["crime_6m_ago"]

df["neighbor_trend_1m"] = df["neighbor_crime_count"] - df["neighbor_1m_ago"]
df["neighbor_trend_3m"] = df["neighbor_crime_count"] - df["neighbor_3m_ago"]
df["neighbor_trend_6m"] = df["neighbor_crime_count"] - df["neighbor_6m_ago"]

df["crime_ratio_1m"] = df["crime_count"] / (df["crime_1m_ago"] + 1.0)
df["crime_ratio_3m"] = df["crime_count"] / (df["crime_3m_ago"] + 1.0)
df["neighbor_ratio_1m"] = df["neighbor_crime_count"] / (df["neighbor_1m_ago"] + 1.0)

for c in ["crime_ratio_1m", "crime_ratio_3m", "neighbor_ratio_1m"]:
    df[c] = df[c].clip(0, 10)

# Features
candidate_features = [
    "crime_count", "crime_1m_ago", "crime_3m_ago", "crime_6m_ago",
    "yearly_avg",
    "neighbor_crime_count", "neighbor_1m_ago", "neighbor_3m_ago", "neighbor_6m_ago",
    "month_sin", "month_cos", "quarter", "year",
    "trend_1m", "trend_3m", "trend_6m",
    "neighbor_trend_1m", "neighbor_trend_3m", "neighbor_trend_6m",
    "crime_ratio_1m", "crime_ratio_3m", "neighbor_ratio_1m",
    "stop_search_count",
    "ss_object_controlled_drugs",
    "ss_object_offensive_weapons",
    "ss_object_stolen_goods",
    "ss_object_article_for_use_in_theft",
    "ss_outcome_arrest",
    "ss_outcome_a_no_further_action_disposal"
]

feature_cols = [c for c in candidate_features if c in df.columns]

X = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
y = df["crime_next_month"].to_numpy(dtype=np.float32)

# Split
target_months = pd.to_datetime(df["target_month"])

train_mask = target_months < pd.Timestamp("2025-01-01")
val_mask = (target_months >= pd.Timestamp("2025-01-01")) & (target_months < pd.Timestamp("2025-07-01"))
test_mask = target_months >= pd.Timestamp("2025-07-01")

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Normalize
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True)
std = np.where(std == 0, 1.0, std)

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# Model
class CrimeRiskNetwork(pt.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = pt.nn.Sequential(
            pt.nn.Linear(input_dim, 64),
            pt.nn.ReLU(),
            pt.nn.Linear(64, 32),
            pt.nn.ReLU(),
            pt.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

w_train = df.loc[train_mask, "downweight"].to_numpy(dtype=np.float32)

def train_model():
    model = CrimeRiskNetwork(X_train.shape[1]).to(device)

    loss_fn = pt.nn.PoissonNLLLoss(log_input=True)

    optimizer = pt.optim.AdamW(model.parameters(), lr=LR)

    train_ds = TensorDataset(
        pt.tensor(X_train, dtype=pt.float32),
        pt.tensor(y_train, dtype=pt.float32),
        pt.tensor(w_train, dtype=pt.float32)
    )

    val_ds = TensorDataset(
        pt.tensor(X_val, dtype=pt.float32),
        pt.tensor(y_val, dtype=pt.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        model.train()

        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device).view(-1,1), wb.to(device).view(-1,1)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss = (loss * wb).sum() / wb.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0

        with pt.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).view(-1,1)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                total += loss.item() * xb.size(0)

        print(f"epoch {epoch} | val loss {total / len(val_loader.dataset):.4f}")

    return model

print("\nTraining...")
model = train_model()

# Test
model.eval()
X_test_t = pt.tensor(X_test, dtype=pt.float32).to(device)

with pt.no_grad():
    preds = np.exp(model(X_test_t).cpu().numpy().flatten())

print("MAE:", np.mean(np.abs(preds - y_test)))
print("RMSE:", np.sqrt(np.mean((preds - y_test) ** 2)))

results = df.loc[test_mask, [LSOA_COL, MONTH_COL]].copy()
results["actual"] = y_test
results["predicted"] = preds

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
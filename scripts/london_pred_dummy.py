import pandas as pd
import numpy as np

print("Reading your feature engineering output...")
# 1. Load the parquet file your script generated
df = pd.read_parquet("crime_data.parquet")

# 2. Get the most recent month in your dataset to simulate "today's" dispatch scenario
latest_month = df["Month"].max()
print(f"Simulating predictions for the most recent month: {latest_month}")
df_latest = df[df["Month"] == latest_month].copy()

# 3. SMART DUMMY LOGIC: Base the probability on historical 'yearly_avg'
# We want areas with high historical crime to have higher hotspot probabilities.
max_avg = df_latest["yearly_avg"].max() if df_latest["yearly_avg"].max() > 0 else 1

# Normalize the yearly average between 0 and 1, and add a little random "CNN noise"
np.random.seed(42) # Ensures the dummy data doesn't change every time you run it
noise = np.random.uniform(-0.15, 0.15, size=len(df_latest))

df_latest["hotspot_probability"] = (df_latest["yearly_avg"] / max_avg) + noise

# Bound the probabilities strictly between 0.0 (0%) and 1.0 (100%)
df_latest["hotspot_probability"] = df_latest["hotspot_probability"].clip(0.0, 1.0)

# 4. Format columns to match what our dashboard expects
df_dashboard = df_latest[["LSOA code", "crime_count", "yearly_avg", "hotspot_probability"]].copy()
df_dashboard = df_dashboard.rename(columns={"LSOA code": "LSOA_code"})

# 5. Save it
df_dashboard.to_csv("london_predictions.csv", index=False)
print("🚀 Success! 'london_predictions.csv' is ready for your dashboard.")
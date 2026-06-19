from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_PATH = PROJECT_ROOT /"main"/"data"/"for training" / "model_predictions_by_lsoa_month.csv"
WORKFORCE_PATH = PROJECT_ROOT /"main"/"data"/"for training" / "police_workforce_selected_forces.csv"

OUTPUT_PATH = PROJECT_ROOT /"main"/"data"/ "outputs" / "police_force_resource_allocation.csv"
SUMMARY_PATH = PROJECT_ROOT /"main"/"data"/ "outputs" / "police_force_resource_allocation_summary.csv"

# Load files


pred = pd.read_csv(PREDICTIONS_PATH)
workforce = pd.read_csv(WORKFORCE_PATH)

print("Prediction file shape:", pred.shape)
print("Workforce file shape:", workforce.shape)


needed_prediction_cols = [
    "LSOA code",
    "Month",
    "target_month",
    "force_slug",
    "police_force_name",
    "actual_crime_count",
    "predicted_crime_count",
    "absolute_error",
]

missing_prediction_cols = [c for c in needed_prediction_cols if c not in pred.columns]

if missing_prediction_cols:
    raise ValueError(f"Missing columns in prediction file: {missing_prediction_cols}")

needed_workforce_cols = [
    "force_slug",
    "police_force_name",
    "police_officers",
    "total_paid_workforce",
    "officers_per_100k_population",
]

missing_workforce_cols = [c for c in needed_workforce_cols if c not in workforce.columns]

if missing_workforce_cols:
    raise ValueError(f"Missing columns in workforce file: {missing_workforce_cols}")


pred["Month"] = pd.to_datetime(pred["Month"])
pred["target_month"] = pd.to_datetime(pred["target_month"])

# Create predicted hotspot labels

# For each target month, the top 10% highest predicted LSOA-months are hotspots.

pred["predicted_hotspot_threshold"] = (
    pred.groupby("target_month")["predicted_crime_count"]
    .transform(lambda x: x.quantile(0.90))
)

pred["is_predicted_hotspot"] = (
    pred["predicted_crime_count"] >= pred["predicted_hotspot_threshold"]
).astype(int)


pred["actual_hotspot_threshold"] = (
    pred.groupby("target_month")["actual_crime_count"]
    .transform(lambda x: x.quantile(0.90))
)

pred["is_actual_hotspot"] = (
    pred["actual_crime_count"] >= pred["actual_hotspot_threshold"]
).astype(int)


pred["hotspot_true_positive"] = (
    (pred["is_predicted_hotspot"] == 1) &
    (pred["is_actual_hotspot"] == 1)
).astype(int)

pred["hotspot_false_positive"] = (
    (pred["is_predicted_hotspot"] == 1) &
    (pred["is_actual_hotspot"] == 0)
).astype(int)

pred["hotspot_false_negative"] = (
    (pred["is_predicted_hotspot"] == 0) &
    (pred["is_actual_hotspot"] == 1)
).astype(int)

total_tp = pred["hotspot_true_positive"].sum()
total_fp = pred["hotspot_false_positive"].sum()
total_fn = pred["hotspot_false_negative"].sum()

hotspot_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else np.nan
hotspot_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else np.nan

print()
print("Hotspot evaluation:")
print(f"True positives: {total_tp:,}")
print(f"False positives: {total_fp:,}")
print(f"False negatives: {total_fn:,}")
print(f"Hotspot precision: {hotspot_precision:.4f}")
print(f"Hotspot recall: {hotspot_recall:.4f}")

# Aggregate predicted demand by police force


force_demand = (
    pred.groupby(["force_slug", "police_force_name"])
    .agg(
        total_predicted_crime=("predicted_crime_count", "sum"),
        total_actual_crime=("actual_crime_count", "sum"),
        avg_predicted_crime=("predicted_crime_count", "mean"),
        avg_actual_crime=("actual_crime_count", "mean"),
        predicted_hotspot_lsoa_months=("is_predicted_hotspot", "sum"),
        actual_hotspot_lsoa_months=("is_actual_hotspot", "sum"),
        hotspot_true_positives=("hotspot_true_positive", "sum"),
        total_lsoa_months=("LSOA code", "count"),
        unique_lsoas=("LSOA code", "nunique"),
        avg_absolute_error=("absolute_error", "mean"),
    )
    .reset_index()
)

force_demand["hotspot_rate"] = (
    force_demand["predicted_hotspot_lsoa_months"] /
    force_demand["total_lsoa_months"]
)

force_demand["actual_hotspot_rate"] = (
    force_demand["actual_hotspot_lsoa_months"] /
    force_demand["total_lsoa_months"]
)

force_demand["force_hotspot_precision"] = (
    force_demand["hotspot_true_positives"] /
    force_demand["predicted_hotspot_lsoa_months"].replace(0, np.nan)
)

allocation = force_demand.merge(
    workforce[
        [
            "force_slug",
            "workforce_table_name",
            "police_officers",
            "total_paid_workforce",
            "officers_per_100k_population",
        ]
    ],
    on="force_slug",
    how="left",
)

missing_workforce = allocation["police_officers"].isna().sum()

if missing_workforce > 0:
    print()
    print("WARNING: Some forces did not match with workforce data:")
    print(allocation[allocation["police_officers"].isna()][["force_slug", "police_force_name"]])

# Calculate resource pressure indicators

allocation["predicted_crime_per_officer"] = (
    allocation["total_predicted_crime"] /
    allocation["police_officers"]
)

allocation["actual_crime_per_officer"] = (
    allocation["total_actual_crime"] /
    allocation["police_officers"]
)

allocation["hotspots_per_100_officers"] = (
    allocation["predicted_hotspot_lsoa_months"] /
    allocation["police_officers"] * 100
)

allocation["predicted_crime_per_total_workforce"] = (
    allocation["total_predicted_crime"] /
    allocation["total_paid_workforce"]
)

allocation["lsoas_per_100_officers"] = (
    allocation["unique_lsoas"] /
    allocation["police_officers"] * 100
)

# Share-based pressure score

# Higher score means predicted demand is high compared with officer share.

allocation["crime_demand_share"] = (
    allocation["total_predicted_crime"] /
    allocation["total_predicted_crime"].sum()
)

allocation["hotspot_share"] = (
    allocation["predicted_hotspot_lsoa_months"] /
    allocation["predicted_hotspot_lsoa_months"].sum()
)

allocation["officer_share"] = (
    allocation["police_officers"] /
    allocation["police_officers"].sum()
)

allocation["workforce_share"] = (
    allocation["total_paid_workforce"] /
    allocation["total_paid_workforce"].sum()
)

allocation["resource_pressure_score"] = (
    0.6 * allocation["crime_demand_share"]
    + 0.4 * allocation["hotspot_share"]
    - allocation["officer_share"]
)

allocation["resource_pressure_score_workforce"] = (
    0.6 * allocation["crime_demand_share"]
    + 0.4 * allocation["hotspot_share"]
    - allocation["workforce_share"]
)

allocation = allocation.sort_values(
    "resource_pressure_score",
    ascending=False
).reset_index(drop=True)

allocation["resource_pressure_rank"] = allocation.index + 1

# Save outputs
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

allocation.to_csv(OUTPUT_PATH, index=False)

summary_cols = [
    "resource_pressure_rank",
    "force_slug",
    "police_force_name",
    "total_predicted_crime",
    "total_actual_crime",
    "predicted_hotspot_lsoa_months",
    "actual_hotspot_lsoa_months",
    "unique_lsoas",
    "hotspot_rate",
    "police_officers",
    "total_paid_workforce",
    "officers_per_100k_population",
    "predicted_crime_per_officer",
    "hotspots_per_100_officers",
    "resource_pressure_score",
]

summary = allocation[summary_cols].copy()
summary.to_csv(SUMMARY_PATH, index=False)

# Print final ranking
print()
print("Police resource allocation ranking:")
print(
    summary.to_string(
        index=False,
        formatters={
            "total_predicted_crime": "{:,.1f}".format,
            "total_actual_crime": "{:,.1f}".format,
            "hotspot_rate": "{:.3f}".format,
            "predicted_crime_per_officer": "{:.3f}".format,
            "hotspots_per_100_officers": "{:.3f}".format,
            "resource_pressure_score": "{:.4f}".format,
        }
    )
)

print()
print("Saved full allocation output to:")
print(OUTPUT_PATH)

print()
print("Saved summary output to:")
print(SUMMARY_PATH)
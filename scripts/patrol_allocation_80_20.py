from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "model_predictions_by_lsoa_month.csv"
WORKFORCE_PATH = PROJECT_ROOT / "data" / "processed" / "police_workforce_selected_forces.csv"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "lsoa_patrol_allocation_80_20.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "patrol_allocation_80_20_summary.csv"


# Assumptions

PATROL_AVAILABILITY_RATE = 0.30
OFFICERS_PER_PATROL_UNIT = 2

NN_ALLOCATION_SHARE = 0.80
EXPLORATION_ALLOCATION_SHARE = 0.20

RANDOM_SEED = 42

MIN_PATROL_UNITS_PER_FORCE_MONTH = 5

MAX_PATROL_UNITS_PER_FORCE_MONTH = 300


pred = pd.read_csv(PREDICTIONS_PATH)
workforce = pd.read_csv(WORKFORCE_PATH)

print("Loaded predictions:", pred.shape)
print("Loaded workforce:", workforce.shape)

required_pred_cols = [
    "LSOA code",
    "target_month",
    "force_slug",
    "police_force_name",
    "actual_crime_count",
    "predicted_crime_count",
]

required_workforce_cols = [
    "force_slug",
    "police_force_name",
    "police_officers",
    "total_paid_workforce",
    "officers_per_100k_population",
]

missing_pred_cols = [col for col in required_pred_cols if col not in pred.columns]
missing_workforce_cols = [col for col in required_workforce_cols if col not in workforce.columns]

if missing_pred_cols:
    raise ValueError(f"Missing columns in predictions file: {missing_pred_cols}")

if missing_workforce_cols:
    raise ValueError(f"Missing columns in workforce file: {missing_workforce_cols}")

pred["target_month"] = pd.to_datetime(pred["target_month"])

pred["predicted_crime_count"] = pd.to_numeric(
    pred["predicted_crime_count"], errors="coerce"
).fillna(0)

pred["actual_crime_count"] = pd.to_numeric(
    pred["actual_crime_count"], errors="coerce"
).fillna(0)

workforce["police_officers"] = pd.to_numeric(
    workforce["police_officers"], errors="coerce"
).fillna(0)

workforce["available_patrol_officers"] = (
    workforce["police_officers"] * PATROL_AVAILABILITY_RATE
)

workforce["available_patrol_units"] = (
    workforce["available_patrol_officers"] / OFFICERS_PER_PATROL_UNIT
)

workforce["available_patrol_units"] = (
    workforce["available_patrol_units"]
    .round()
    .clip(lower=MIN_PATROL_UNITS_PER_FORCE_MONTH, upper=MAX_PATROL_UNITS_PER_FORCE_MONTH)
    .astype(int)
)

print("\nEstimated patrol units per force:")
print(
    workforce[
        [
            "force_slug",
            "police_force_name",
            "police_officers",
            "available_patrol_officers",
            "available_patrol_units",
        ]
    ]
)

pred = pred.merge(
    workforce[
        [
            "force_slug",
            "police_officers",
            "total_paid_workforce",
            "officers_per_100k_population",
            "available_patrol_officers",
            "available_patrol_units",
        ]
    ],
    on="force_slug",
    how="left",
)

if pred["available_patrol_units"].isna().any():
    missing_forces = pred.loc[pred["available_patrol_units"].isna(), "force_slug"].unique()
    raise ValueError(f"Some forces are missing workforce data: {missing_forces}")


# Allocation function

def allocate_for_force_month(
    group: pd.DataFrame,
    force_slug: str,
    police_force_name: str,
    target_month
) -> pd.DataFrame:
    """
    For each police-force-month:
    - 80% of patrol units go to highest predicted crime LSOAs
    - 20% goes to controlled random exploration
    """

    group = group.copy()

    group["force_slug"] = force_slug
    group["police_force_name"] = police_force_name
    group["target_month"] = target_month

    total_units = int(group["available_patrol_units"].iloc[0])

    # number of patrol units does not exceed number of available LSOAs
    total_units = min(total_units, len(group))

    if total_units <= 0:
        return pd.DataFrame()

    nn_units = int(round(total_units * NN_ALLOCATION_SHARE))
    exploration_units = total_units - nn_units

    nn_units = max(1, nn_units)
    exploration_units = max(0, exploration_units)

    if nn_units + exploration_units > len(group):
        exploration_units = max(0, len(group) - nn_units)

    # 80% NN allocation
  
    group_sorted = group.sort_values(
        by="predicted_crime_count",
        ascending=False
    )

    nn_selected = group_sorted.head(nn_units).copy()
    nn_selected["allocation_type"] = "NN_hotspot"
    nn_selected["allocation_reason"] = "Selected because it has one of the highest neural-network predicted crime counts in this police-force-month."

    # 20% controlled random exploration

    remaining = group_sorted.iloc[nn_units:].copy()

    if exploration_units > 0 and len(remaining) > 0:
        remaining["risk_rank_pct"] = remaining["predicted_crime_count"].rank(
            pct=True,
            method="average"
        )

        exploration_candidates = remaining[
            (remaining["risk_rank_pct"] >= 0.30)
            & (remaining["risk_rank_pct"] <= 0.90)
        ].copy()

        if len(exploration_candidates) < exploration_units:
            exploration_candidates = remaining.copy()

        weights = exploration_candidates["predicted_crime_count"].clip(lower=0.01)

        exploration_selected = exploration_candidates.sample(
            n=min(exploration_units, len(exploration_candidates)),
            weights=weights,
            random_state=RANDOM_SEED + hash((force_slug, str(target_month))) % 10000,
            replace=False,
        ).copy()

        exploration_selected["allocation_type"] = "exploration_random"
        exploration_selected["allocation_reason"] = (
            "Selected through controlled random exploration from non-hotspot LSOAs to reduce feedback-loop risk."
        )

        selected = pd.concat([nn_selected, exploration_selected], ignore_index=True)

    else:
        selected = nn_selected.copy()

    selected["patrol_units_allocated"] = 1
    selected["total_force_month_patrol_units"] = total_units
    selected["nn_units_planned"] = nn_units
    selected["exploration_units_planned"] = exploration_units

    selected["nn_allocation_share"] = NN_ALLOCATION_SHARE
    selected["exploration_allocation_share"] = EXPLORATION_ALLOCATION_SHARE
    selected["patrol_availability_rate"] = PATROL_AVAILABILITY_RATE
    selected["officers_per_patrol_unit"] = OFFICERS_PER_PATROL_UNIT

    return selected

allocation_parts = []

for (force_slug, police_force_name, target_month), group in pred.groupby(
    ["force_slug", "police_force_name", "target_month"]
):
    allocated_group = allocate_for_force_month(
        group=group,
        force_slug=force_slug,
        police_force_name=police_force_name,
        target_month=target_month,
    )

    if not allocated_group.empty:
        allocation_parts.append(allocated_group)

allocation = pd.concat(allocation_parts, ignore_index=True)

summary = (
    allocation
    .groupby(["force_slug", "police_force_name", "target_month"])
    .agg(
        total_allocated_lsoas=("LSOA code", "count"),
        nn_hotspot_allocations=("allocation_type", lambda x: (x == "NN_hotspot").sum()),
        exploration_allocations=("allocation_type", lambda x: (x == "exploration_random").sum()),
        total_predicted_crime_in_allocated_areas=("predicted_crime_count", "sum"),
        avg_predicted_crime_in_allocated_areas=("predicted_crime_count", "mean"),
        total_actual_crime_in_allocated_areas=("actual_crime_count", "sum"),
        avg_actual_crime_in_allocated_areas=("actual_crime_count", "mean"),
        police_officers=("police_officers", "first"),
        available_patrol_officers=("available_patrol_officers", "first"),
        available_patrol_units=("available_patrol_units", "first"),
    )
    .reset_index()
)

summary["nn_allocation_actual_share"] = (
    summary["nn_hotspot_allocations"] / summary["total_allocated_lsoas"]
)

summary["exploration_allocation_actual_share"] = (
    summary["exploration_allocations"] / summary["total_allocated_lsoas"]
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

allocation.to_csv(OUTPUT_PATH, index=False)
summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)

print("\nSaved LSOA patrol allocation to:")
print(OUTPUT_PATH)

print("\nSaved allocation summary to:")
print(SUMMARY_OUTPUT_PATH)

print("\nAllocation summary preview:")
print(
    summary[
        [
            "force_slug",
            "police_force_name",
            "target_month",
            "total_allocated_lsoas",
            "nn_hotspot_allocations",
            "exploration_allocations",
            "nn_allocation_actual_share",
            "exploration_allocation_actual_share",
        ]
    ].head(30)
)
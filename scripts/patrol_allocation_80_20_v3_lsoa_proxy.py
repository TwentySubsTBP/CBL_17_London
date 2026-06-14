from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

# --------------------------------------------------
# Patrol Allocation V3
# 80% = Neural Network predicted hotspots
# 20% = LSOA-code proximity exploration
#
# This version avoids latitude/longitude.
# It uses the numeric part of the LSOA code as a rough proxy
# for choosing exploration LSOAs close to NN hotspot LSOAs.
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "model_predictions_by_lsoa_month.csv"
WORKFORCE_PATH = PROJECT_ROOT / "data" / "processed" / "police_workforce_selected_forces.csv"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "lsoa_patrol_allocation_80_20_v3_lsoa_proxy.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "patrol_allocation_80_20_v3_lsoa_proxy_summary.csv"
FORCE_SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "patrol_allocation_80_20_v3_lsoa_proxy_force_summary.csv"

# -----------------------------
# Main assumptions
# -----------------------------
PATROL_AVAILABILITY_RATE = 0.30
OFFICERS_PER_PATROL_UNIT = 2

# Keep this as 0.10 if you want to continue exactly from V2.
# Change to 1.00 if you want all available patrol capacity to be split by 80/20.
TARGETED_PATROL_SHARE = 0.10

NN_ALLOCATION_SHARE = 0.80
EXPLORATION_ALLOCATION_SHARE = 0.20

MAX_LSOA_COVERAGE_SHARE = 0.30
MIN_TARGETED_PATROL_SLOTS = 5

RANDOM_SEED = 42

# Exploration score weighting
# Higher risk and closer LSOA-code proximity are both rewarded.
RISK_WEIGHT = 0.60
PROXIMITY_WEIGHT = 0.40


def stable_random_seed(force_slug: str, target_month) -> int:
    key = f"{force_slug}_{pd.Timestamp(target_month).date()}_{RANDOM_SEED}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def extract_lsoa_number(lsoa_code: str):
    """
    Example:
    E01033010 -> 1033010 as an integer.

    This is not a real geographic distance.
    It is only a rough administrative-code proximity proxy.
    """
    if pd.isna(lsoa_code):
        return np.nan

    digits = "".join(ch for ch in str(lsoa_code) if ch.isdigit())

    if digits == "":
        return np.nan

    return int(digits)


# -----------------------------
# Load data
# -----------------------------
pred = pd.read_csv(PREDICTIONS_PATH)
workforce = pd.read_csv(WORKFORCE_PATH)

print("Loaded predictions:", pred.shape)
print("Loaded workforce:", workforce.shape)

# -----------------------------
# Required columns
# -----------------------------
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

# -----------------------------
# Clean values
# -----------------------------
pred["target_month"] = pd.to_datetime(pred["target_month"])

pred["predicted_crime_count"] = pd.to_numeric(
    pred["predicted_crime_count"], errors="coerce"
).fillna(0)

pred["actual_crime_count"] = pd.to_numeric(
    pred["actual_crime_count"], errors="coerce"
).fillna(0)

pred["lsoa_number_proxy"] = pred["LSOA code"].apply(extract_lsoa_number)

if pred["lsoa_number_proxy"].isna().any():
    missing_count = pred["lsoa_number_proxy"].isna().sum()
    print(f"Warning: {missing_count} rows have missing LSOA number proxy.")

workforce["police_officers"] = pd.to_numeric(
    workforce["police_officers"], errors="coerce"
).fillna(0)

# -----------------------------
# Estimate patrol capacity
# -----------------------------
workforce["available_patrol_officers"] = (
    workforce["police_officers"] * PATROL_AVAILABILITY_RATE
)

workforce["raw_available_patrol_units"] = (
    workforce["available_patrol_officers"] / OFFICERS_PER_PATROL_UNIT
)

workforce["targeted_patrol_units_before_lsoa_cap"] = (
    workforce["raw_available_patrol_units"] * TARGETED_PATROL_SHARE
)

workforce["targeted_patrol_units_before_lsoa_cap"] = (
    workforce["targeted_patrol_units_before_lsoa_cap"].round().astype(int)
)

print("\nEstimated patrol capacity before LSOA coverage cap:")
print(
    workforce[
        [
            "force_slug",
            "police_officers",
            "available_patrol_officers",
            "raw_available_patrol_units",
            "targeted_patrol_units_before_lsoa_cap",
        ]
    ]
)

# -----------------------------
# Merge workforce into predictions
# -----------------------------
pred = pred.merge(
    workforce[
        [
            "force_slug",
            "police_officers",
            "total_paid_workforce",
            "officers_per_100k_population",
            "available_patrol_officers",
            "raw_available_patrol_units",
            "targeted_patrol_units_before_lsoa_cap",
        ]
    ],
    on="force_slug",
    how="left",
)

if pred["targeted_patrol_units_before_lsoa_cap"].isna().any():
    missing_forces = pred.loc[
        pred["targeted_patrol_units_before_lsoa_cap"].isna(),
        "force_slug",
    ].unique()
    raise ValueError(f"Some forces are missing workforce data: {missing_forces}")


def allocate_for_force_month(
    group: pd.DataFrame,
    force_slug: str,
    police_force_name: str,
    target_month,
) -> pd.DataFrame:
    """
    For each police-force-month:
    - Select 80% highest predicted-crime LSOAs.
    - Select 20% exploration LSOAs using:
        moderate risk
        non-hotspot status
        LSOA-code proximity to selected NN hotspots
    """

    group = group.copy()

    group["force_slug"] = force_slug
    group["police_force_name"] = police_force_name
    group["target_month"] = target_month

    n_lsoas = len(group)
    raw_targeted_units = int(group["targeted_patrol_units_before_lsoa_cap"].iloc[0])

    # Cap selected areas so model does not cover too many LSOAs
    lsoa_coverage_cap = int(round(n_lsoas * MAX_LSOA_COVERAGE_SHARE))
    lsoa_coverage_cap = max(MIN_TARGETED_PATROL_SLOTS, lsoa_coverage_cap)

    total_units = min(raw_targeted_units, lsoa_coverage_cap, n_lsoas)
    total_units = max(min(MIN_TARGETED_PATROL_SLOTS, n_lsoas), total_units)

    if total_units <= 0:
        return pd.DataFrame()

    nn_units = int(round(total_units * NN_ALLOCATION_SHARE))
    exploration_units = total_units - nn_units

    if total_units >= 5 and exploration_units == 0:
        exploration_units = 1
        nn_units = total_units - exploration_units

    # -----------------------------
    # 80% NN hotspot allocation
    # -----------------------------
    group_sorted = group.sort_values(
        by="predicted_crime_count",
        ascending=False,
    )

    nn_selected = group_sorted.head(nn_units).copy()

    nn_selected["allocation_type"] = "NN_hotspot"
    nn_selected["allocation_reason"] = (
        "Selected because it has one of the highest neural-network predicted crime counts "
        "in this police-force-month."
    )

    # -----------------------------
    # 20% LSOA-code proximity exploration
    # -----------------------------
    remaining = group_sorted.iloc[nn_units:].copy()

    if exploration_units > 0 and len(remaining) > 0:
        hotspot_lsoa_numbers = nn_selected["lsoa_number_proxy"].dropna().to_numpy()

        remaining["risk_rank_pct"] = remaining["predicted_crime_count"].rank(
            pct=True,
            method="average",
        )

        # Candidate pool: non-hotspots with moderate predicted risk
        exploration_candidates = remaining[
            (remaining["risk_rank_pct"] >= 0.30)
            & (remaining["risk_rank_pct"] <= 0.90)
        ].copy()

        if len(exploration_candidates) < exploration_units:
            exploration_candidates = remaining.copy()

        # Calculate nearest LSOA-code gap to any NN hotspot
        if len(hotspot_lsoa_numbers) > 0:
            candidate_numbers = exploration_candidates["lsoa_number_proxy"].to_numpy()

            nearest_gaps = []

            for num in candidate_numbers:
                if pd.isna(num):
                    nearest_gaps.append(np.nan)
                else:
                    nearest_gaps.append(np.min(np.abs(hotspot_lsoa_numbers - num)))

            exploration_candidates["nearest_hotspot_lsoa_code_gap"] = nearest_gaps
        else:
            exploration_candidates["nearest_hotspot_lsoa_code_gap"] = np.nan

        # Fill missing gaps with large number
        max_gap = exploration_candidates["nearest_hotspot_lsoa_code_gap"].max()

        if pd.isna(max_gap):
            max_gap = 999999

        exploration_candidates["nearest_hotspot_lsoa_code_gap"] = (
            exploration_candidates["nearest_hotspot_lsoa_code_gap"].fillna(max_gap)
        )

        # Smaller gap = closer proxy
        exploration_candidates["proximity_rank_pct"] = (
            1
            - exploration_candidates["nearest_hotspot_lsoa_code_gap"].rank(
                pct=True,
                method="average",
            )
        )

        # Combined score: risk + proxy closeness
        exploration_candidates["exploration_score"] = (
            RISK_WEIGHT * exploration_candidates["risk_rank_pct"]
            + PROXIMITY_WEIGHT * exploration_candidates["proximity_rank_pct"]
        )

        # Weighted random selection using the score.
        # This keeps exploration partly random but pushes it toward closer and moderate-risk LSOAs.
        weights = exploration_candidates["exploration_score"].clip(lower=0.01)

        exploration_selected = exploration_candidates.sample(
            n=min(exploration_units, len(exploration_candidates)),
            weights=weights,
            random_state=stable_random_seed(force_slug, target_month),
            replace=False,
        ).copy()

        exploration_selected["allocation_type"] = "exploration_lsoa_proxy"
        exploration_selected["allocation_reason"] = (
            "Selected through controlled exploration from non-hotspot LSOAs using "
            "moderate risk and LSOA-code proximity to NN hotspot areas."
        )

        selected = pd.concat([nn_selected, exploration_selected], ignore_index=True)

    else:
        selected = nn_selected.copy()

    selected["patrol_units_allocated"] = 1
    selected["total_force_month_patrol_slots"] = total_units
    selected["nn_units_planned"] = nn_units
    selected["exploration_units_planned"] = exploration_units
    selected["number_of_lsoas_in_force_month"] = n_lsoas
    selected["lsoa_coverage_cap"] = lsoa_coverage_cap

    selected["patrol_availability_rate"] = PATROL_AVAILABILITY_RATE
    selected["targeted_patrol_share"] = TARGETED_PATROL_SHARE
    selected["officers_per_patrol_unit"] = OFFICERS_PER_PATROL_UNIT
    selected["max_lsoa_coverage_share"] = MAX_LSOA_COVERAGE_SHARE

    selected["risk_weight"] = RISK_WEIGHT
    selected["proximity_weight"] = PROXIMITY_WEIGHT

    return selected


# -----------------------------
# Apply allocation manually
# -----------------------------
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

# -----------------------------
# Summary
# -----------------------------
summary = (
    allocation
    .groupby(["force_slug", "police_force_name", "target_month"])
    .agg(
        total_allocated_lsoas=("LSOA code", "count"),
        nn_hotspot_allocations=("allocation_type", lambda x: (x == "NN_hotspot").sum()),
        exploration_allocations=("allocation_type", lambda x: (x == "exploration_lsoa_proxy").sum()),
        total_predicted_crime_in_allocated_areas=("predicted_crime_count", "sum"),
        avg_predicted_crime_in_allocated_areas=("predicted_crime_count", "mean"),
        total_actual_crime_in_allocated_areas=("actual_crime_count", "sum"),
        avg_actual_crime_in_allocated_areas=("actual_crime_count", "mean"),
        police_officers=("police_officers", "first"),
        available_patrol_officers=("available_patrol_officers", "first"),
        raw_available_patrol_units=("raw_available_patrol_units", "first"),
        targeted_patrol_units_before_lsoa_cap=("targeted_patrol_units_before_lsoa_cap", "first"),
        total_force_month_patrol_slots=("total_force_month_patrol_slots", "first"),
        number_of_lsoas_in_force_month=("number_of_lsoas_in_force_month", "first"),
        lsoa_coverage_cap=("lsoa_coverage_cap", "first"),
        avg_nearest_hotspot_lsoa_code_gap=("nearest_hotspot_lsoa_code_gap", "mean"),
    )
    .reset_index()
)

summary["nn_allocation_actual_share"] = (
    summary["nn_hotspot_allocations"] / summary["total_allocated_lsoas"]
)

summary["exploration_allocation_actual_share"] = (
    summary["exploration_allocations"] / summary["total_allocated_lsoas"]
)

force_summary = (
    summary
    .groupby(["force_slug", "police_force_name"])
    .agg(
        avg_monthly_allocated_lsoas=("total_allocated_lsoas", "mean"),
        avg_monthly_nn_hotspots=("nn_hotspot_allocations", "mean"),
        avg_monthly_exploration=("exploration_allocations", "mean"),
        avg_monthly_predicted_crime_in_allocated_areas=(
            "total_predicted_crime_in_allocated_areas",
            "mean",
        ),
        avg_lsoa_code_gap_to_nearest_hotspot=("avg_nearest_hotspot_lsoa_code_gap", "mean"),
        police_officers=("police_officers", "first"),
        raw_available_patrol_units=("raw_available_patrol_units", "first"),
        targeted_patrol_units_before_lsoa_cap=(
            "targeted_patrol_units_before_lsoa_cap",
            "first",
        ),
    )
    .reset_index()
    .sort_values("avg_monthly_predicted_crime_in_allocated_areas", ascending=False)
)

# -----------------------------
# Save
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

allocation.to_csv(OUTPUT_PATH, index=False)
summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)
force_summary.to_csv(FORCE_SUMMARY_OUTPUT_PATH, index=False)

print("\nSaved LSOA-code proximity allocation to:")
print(OUTPUT_PATH)

print("\nSaved monthly summary to:")
print(SUMMARY_OUTPUT_PATH)

print("\nSaved force-level summary to:")
print(FORCE_SUMMARY_OUTPUT_PATH)

print("\nForce-level preview:")
print(
    force_summary[
        [
            "force_slug",
            "police_force_name",
            "avg_monthly_allocated_lsoas",
            "avg_monthly_nn_hotspots",
            "avg_monthly_exploration",
            "avg_lsoa_code_gap_to_nearest_hotspot",
        ]
    ]
)
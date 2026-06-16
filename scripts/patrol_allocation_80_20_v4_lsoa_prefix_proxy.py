from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

# --------------------------------------------------
# Patrol Allocation V4
# 80% = Neural Network predicted hotspots
# 20% = LSOA-prefix-proximity exploration
#
# This version does NOT use latitude/longitude.
# It uses LSOA code structure as a rough proximity proxy:
# Example:
#   E01033010 -> last five digits = 33010
#   prefix group = 330
#
# This is not real geographic distance, but it is better than fully random
# exploration when coordinates are unavailable.
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "model_predictions_by_lsoa_month.csv"
WORKFORCE_PATH = PROJECT_ROOT / "data" / "processed" / "police_workforce_selected_forces.csv"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "lsoa_patrol_allocation_80_20_v4_lsoa_prefix_proxy.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "patrol_allocation_80_20_v4_lsoa_prefix_proxy_summary.csv"
FORCE_SUMMARY_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "patrol_allocation_80_20_v4_lsoa_prefix_proxy_force_summary.csv"

# -----------------------------
# Main assumptions
# -----------------------------
PATROL_AVAILABILITY_RATE = 0.30
OFFICERS_PER_PATROL_UNIT = 2

# Recommended for report clarity:
# 30% officers available for patrol, then split by 80/20.
# If you want the older conservative V2/V3 style, change this to 0.10.
TARGETED_PATROL_SHARE = 1.00

NN_ALLOCATION_SHARE = 0.80
EXPLORATION_ALLOCATION_SHARE = 0.20

MAX_LSOA_COVERAGE_SHARE = 0.30
MIN_TARGETED_PATROL_SLOTS = 5

RANDOM_SEED = 42

# Exploration score weights
# Risk should still matter, but prefix proximity helps keep exploration near hotspot blocks.
RISK_WEIGHT = 0.50
SAME_PREFIX_WEIGHT = 0.30
PREFIX_GAP_WEIGHT = 0.15
LOCAL_CODE_GAP_WEIGHT = 0.05


def stable_random_seed(force_slug: str, target_month) -> int:
    key = f"{force_slug}_{pd.Timestamp(target_month).date()}_{RANDOM_SEED}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def extract_lsoa_digits(lsoa_code):
    """
    Example:
    E01033010 -> 01033010
    """
    if pd.isna(lsoa_code):
        return np.nan

    digits = "".join(ch for ch in str(lsoa_code) if ch.isdigit())

    if digits == "":
        return np.nan

    return digits


def add_lsoa_proxy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - lsoa_digits
    - lsoa_last5
    - lsoa_prefix3
    - lsoa_last5_num
    - lsoa_prefix3_num

    Example:
    E01033010
    digits = 01033010
    last5 = 33010
    prefix3 = 330
    """
    df = df.copy()

    df["lsoa_digits"] = df["LSOA code"].apply(extract_lsoa_digits)

    df["lsoa_last5"] = df["lsoa_digits"].apply(
        lambda x: str(x)[-5:] if pd.notna(x) and len(str(x)) >= 5 else np.nan
    )

    df["lsoa_prefix3"] = df["lsoa_last5"].apply(
        lambda x: str(x)[:3] if pd.notna(x) and len(str(x)) >= 3 else np.nan
    )

    df["lsoa_last5_num"] = pd.to_numeric(df["lsoa_last5"], errors="coerce")
    df["lsoa_prefix3_num"] = pd.to_numeric(df["lsoa_prefix3"], errors="coerce")

    return df


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

pred = add_lsoa_proxy_columns(pred)

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
        1. non-hotspot status
        2. moderate predicted risk
        3. same LSOA prefix block as NN hotspots
        4. small LSOA-code gap to nearest NN hotspot
    """

    group = group.copy()

    group["force_slug"] = force_slug
    group["police_force_name"] = police_force_name
    group["target_month"] = target_month

    n_lsoas = len(group)
    raw_targeted_units = int(group["targeted_patrol_units_before_lsoa_cap"].iloc[0])

    # Maximum percentage of LSOAs that can be selected in one force-month
    lsoa_coverage_cap = int(round(n_lsoas * MAX_LSOA_COVERAGE_SHARE))
    lsoa_coverage_cap = max(MIN_TARGETED_PATROL_SLOTS, lsoa_coverage_cap)

    total_units = min(raw_targeted_units, lsoa_coverage_cap, n_lsoas)

    # Ensure small forces still get a minimum number of selected LSOAs if possible
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

    # Add exploration-specific columns to NN rows as blank values
    nn_selected["same_prefix_as_hotspot"] = np.nan
    nn_selected["nearest_hotspot_prefix_gap"] = np.nan
    nn_selected["nearest_hotspot_lsoa_last5_gap"] = np.nan
    nn_selected["exploration_score"] = np.nan
    nn_selected["exploration_pool_rule"] = np.nan

    # -----------------------------
    # 20% LSOA-prefix-proximity exploration
    # -----------------------------
    remaining = group_sorted.iloc[nn_units:].copy()

    if exploration_units > 0 and len(remaining) > 0:
        hotspot_prefixes = set(nn_selected["lsoa_prefix3"].dropna().astype(str))

        hotspot_prefix_nums = nn_selected["lsoa_prefix3_num"].dropna().to_numpy()
        hotspot_last5_nums = nn_selected["lsoa_last5_num"].dropna().to_numpy()

        remaining["risk_rank_pct"] = remaining["predicted_crime_count"].rank(
            pct=True,
            method="average",
        )

        # Candidate pool:
        # Non-hotspot LSOAs with moderate risk.
        exploration_candidates = remaining[
            (remaining["risk_rank_pct"] >= 0.30)
            & (remaining["risk_rank_pct"] <= 0.90)
        ].copy()

        if len(exploration_candidates) < exploration_units:
            exploration_candidates = remaining.copy()

        # Same-prefix flag
        exploration_candidates["same_prefix_as_hotspot"] = (
            exploration_candidates["lsoa_prefix3"].astype(str).isin(hotspot_prefixes)
        ).astype(int)

        # Nearest prefix gap
        nearest_prefix_gaps = []
        for prefix_num in exploration_candidates["lsoa_prefix3_num"].to_numpy():
            if pd.isna(prefix_num) or len(hotspot_prefix_nums) == 0:
                nearest_prefix_gaps.append(np.nan)
            else:
                nearest_prefix_gaps.append(np.min(np.abs(hotspot_prefix_nums - prefix_num)))

        exploration_candidates["nearest_hotspot_prefix_gap"] = nearest_prefix_gaps

        # Nearest local LSOA last-5 code gap
        nearest_last5_gaps = []
        for last5_num in exploration_candidates["lsoa_last5_num"].to_numpy():
            if pd.isna(last5_num) or len(hotspot_last5_nums) == 0:
                nearest_last5_gaps.append(np.nan)
            else:
                nearest_last5_gaps.append(np.min(np.abs(hotspot_last5_nums - last5_num)))

        exploration_candidates["nearest_hotspot_lsoa_last5_gap"] = nearest_last5_gaps

        # Fill missing values with large gaps
        for col in ["nearest_hotspot_prefix_gap", "nearest_hotspot_lsoa_last5_gap"]:
            max_value = exploration_candidates[col].max()
            if pd.isna(max_value):
                max_value = 999999
            exploration_candidates[col] = exploration_candidates[col].fillna(max_value)

        # Smaller gap = better proximity score
        exploration_candidates["prefix_gap_score"] = (
            1
            - exploration_candidates["nearest_hotspot_prefix_gap"].rank(
                pct=True,
                method="average",
            )
        )

        exploration_candidates["local_code_gap_score"] = (
            1
            - exploration_candidates["nearest_hotspot_lsoa_last5_gap"].rank(
                pct=True,
                method="average",
            )
        )

        # Combined exploration score
        exploration_candidates["exploration_score"] = (
            RISK_WEIGHT * exploration_candidates["risk_rank_pct"]
            + SAME_PREFIX_WEIGHT * exploration_candidates["same_prefix_as_hotspot"]
            + PREFIX_GAP_WEIGHT * exploration_candidates["prefix_gap_score"]
            + LOCAL_CODE_GAP_WEIGHT * exploration_candidates["local_code_gap_score"]
        )

        # Prefer same-prefix candidates if there are enough.
        same_prefix_candidates = exploration_candidates[
            exploration_candidates["same_prefix_as_hotspot"] == 1
        ].copy()

        if len(same_prefix_candidates) >= exploration_units:
            final_pool = same_prefix_candidates.copy()
            pool_rule = "same_prefix_only"
        else:
            final_pool = exploration_candidates.copy()
            pool_rule = "mixed_prefix_fallback"

        weights = final_pool["exploration_score"].clip(lower=0.01)

        exploration_selected = final_pool.sample(
            n=min(exploration_units, len(final_pool)),
            weights=weights,
            random_state=stable_random_seed(force_slug, target_month),
            replace=False,
        ).copy()

        exploration_selected["allocation_type"] = "exploration_lsoa_prefix_proxy"
        exploration_selected["allocation_reason"] = (
            "Selected through controlled exploration from non-hotspot LSOAs using "
            "moderate risk and LSOA-prefix proximity to NN hotspot areas."
        )
        exploration_selected["exploration_pool_rule"] = pool_rule

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
    selected["same_prefix_weight"] = SAME_PREFIX_WEIGHT
    selected["prefix_gap_weight"] = PREFIX_GAP_WEIGHT
    selected["local_code_gap_weight"] = LOCAL_CODE_GAP_WEIGHT

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
# Monthly summary
# -----------------------------
summary = (
    allocation
    .groupby(["force_slug", "police_force_name", "target_month"])
    .agg(
        total_allocated_lsoas=("LSOA code", "count"),
        nn_hotspot_allocations=("allocation_type", lambda x: (x == "NN_hotspot").sum()),
        exploration_allocations=("allocation_type", lambda x: (x == "exploration_lsoa_prefix_proxy").sum()),
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
    )
    .reset_index()
)

summary["nn_allocation_actual_share"] = (
    summary["nn_hotspot_allocations"] / summary["total_allocated_lsoas"]
)

summary["exploration_allocation_actual_share"] = (
    summary["exploration_allocations"] / summary["total_allocated_lsoas"]
)

# -----------------------------
# Exploration-only metrics
# -----------------------------
exploration_rows = allocation[
    allocation["allocation_type"] == "exploration_lsoa_prefix_proxy"
].copy()

if not exploration_rows.empty:
    exploration_metrics = (
        exploration_rows
        .groupby(["force_slug", "police_force_name", "target_month"])
        .agg(
            exploration_same_prefix_share=("same_prefix_as_hotspot", "mean"),
            avg_nearest_hotspot_prefix_gap=("nearest_hotspot_prefix_gap", "mean"),
            avg_nearest_hotspot_lsoa_last5_gap=("nearest_hotspot_lsoa_last5_gap", "mean"),
            avg_exploration_score=("exploration_score", "mean"),
        )
        .reset_index()
    )

    summary = summary.merge(
        exploration_metrics,
        on=["force_slug", "police_force_name", "target_month"],
        how="left",
    )
else:
    summary["exploration_same_prefix_share"] = np.nan
    summary["avg_nearest_hotspot_prefix_gap"] = np.nan
    summary["avg_nearest_hotspot_lsoa_last5_gap"] = np.nan
    summary["avg_exploration_score"] = np.nan

# -----------------------------
# Force-level summary
# -----------------------------
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
        avg_exploration_same_prefix_share=("exploration_same_prefix_share", "mean"),
        avg_nearest_hotspot_prefix_gap=("avg_nearest_hotspot_prefix_gap", "mean"),
        avg_nearest_hotspot_lsoa_last5_gap=("avg_nearest_hotspot_lsoa_last5_gap", "mean"),
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

print("\nSaved V4 LSOA-prefix-proxy allocation to:")
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
            "avg_exploration_same_prefix_share",
            "avg_nearest_hotspot_prefix_gap",
            "avg_nearest_hotspot_lsoa_last5_gap",
        ]
    ]
)
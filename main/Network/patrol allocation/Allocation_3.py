from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

# --------------------------------------------------
# Patrol Allocation V5
#
# Main idea:
# - Use 30% of officers as patrol-available.
# - Convert officers into patrol units.
# - Use ALL available patrol units.
# - 80% patrol units go to NN hotspot LSOAs.
# - 20% patrol units go to exploration LSOAs.
# - Hotspots are top 10% LSOAs by NN predicted crime count.
# - Exploration uses non-hotspot, moderate-risk, LSOA-prefix-proxy areas.
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PREDICTIONS_PATH = PROJECT_ROOT /"main"/"data"/"for training" / "model_predictions_by_lsoa_month.csv"
WORKFORCE_PATH = PROJECT_ROOT /"main"/"data"/"for training" / "police_workforce_selected_forces.csv"

OUTPUT_PATH = PROJECT_ROOT /"main"/"data"/ "outputs" / "lsoa_patrol_allocation_80_20_v5_unit_intensity.csv"
SUMMARY_OUTPUT_PATH = PROJECT_ROOT/"main"/"data"/  "outputs" / "patrol_allocation_80_20_v5_unit_intensity_summary.csv"
FORCE_SUMMARY_OUTPUT_PATH = PROJECT_ROOT/"main"/"data"/ "outputs" / "patrol_allocation_80_20_v5_unit_intensity_force_summary.csv"

# -----------------------------
# Main assumptions
# -----------------------------
PATROL_AVAILABILITY_RATE = 0.30
OFFICERS_PER_PATROL_UNIT = 2
TARGETED_PATROL_SHARE = 1.00

NN_ALLOCATION_SHARE = 0.80
EXPLORATION_ALLOCATION_SHARE = 0.20

# This defines hotspot areas:
# top 10% LSOAs by predicted_crime_count within each force-month.
HOTSPOT_LSOA_SHARE = 0.10

# Exploration area count is around 25% of hotspot area count,
# because 20/80 = 0.25.
EXPLORATION_TO_HOTSPOT_AREA_RATIO = EXPLORATION_ALLOCATION_SHARE / NN_ALLOCATION_SHARE

MIN_HOTSPOT_LSOAS = 5
MIN_EXPLORATION_LSOAS = 1

RANDOM_SEED = 42

# Weights for exploration scoring
RISK_WEIGHT = 0.50
SAME_PREFIX_WEIGHT = 0.30
PREFIX_GAP_WEIGHT = 0.15
LOCAL_CODE_GAP_WEIGHT = 0.05


def stable_random_seed(force_slug: str, target_month) -> int:
    key = f"{force_slug}_{pd.Timestamp(target_month).date()}_{RANDOM_SEED}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def extract_lsoa_digits(lsoa_code):
    if pd.isna(lsoa_code):
        return np.nan

    digits = "".join(ch for ch in str(lsoa_code) if ch.isdigit())

    if digits == "":
        return np.nan

    return digits


def add_lsoa_proxy_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def distribute_integer_units(df: pd.DataFrame, total_units: int, weight_col: str) -> pd.DataFrame:
    """
    Distributes integer patrol units across selected LSOAs.
    Higher weight = more patrol units.
    Ensures the total allocated patrol units equals total_units.
    """

    df = df.copy()
    total_units = int(round(total_units))

    if total_units <= 0 or df.empty:
        return df.iloc[0:0].copy()

    # If there are more selected LSOAs than patrol units, keep only the strongest rows.
    if len(df) > total_units:
        df = df.sort_values(weight_col, ascending=False).head(total_units).copy()

    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0).clip(lower=0)

    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=df.index)

    # Give every selected LSOA at least 1 unit if possible.
    if total_units >= len(df):
        base_units = pd.Series(1, index=df.index, dtype=int)
        remaining_units = total_units - len(df)
    else:
        base_units = pd.Series(0, index=df.index, dtype=int)
        remaining_units = total_units

    raw_extra = weights / weights.sum() * remaining_units
    extra_units = np.floor(raw_extra).astype(int)

    allocated_units = base_units + extra_units

    remainder = remaining_units - int(extra_units.sum())

    if remainder > 0:
        fractional_order = (raw_extra - extra_units).sort_values(ascending=False).index
        allocated_units.loc[fractional_order[:remainder]] += 1

    df["patrol_units_allocated"] = allocated_units.astype(int)

    return df[df["patrol_units_allocated"] > 0].copy()


# -----------------------------
# Load data
# -----------------------------
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
# Clean data
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
# Patrol capacity
# -----------------------------
workforce["available_patrol_officers"] = (
    workforce["police_officers"] * PATROL_AVAILABILITY_RATE
)

workforce["available_patrol_units"] = (
    workforce["available_patrol_officers"] / OFFICERS_PER_PATROL_UNIT
)

workforce["targeted_patrol_units"] = (
    workforce["available_patrol_units"] * TARGETED_PATROL_SHARE
).round().astype(int)

print("\nEstimated patrol capacity:")
print(
    workforce[
        [
            "force_slug",
            "police_officers",
            "available_patrol_officers",
            "available_patrol_units",
            "targeted_patrol_units",
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
            "targeted_patrol_units",
        ]
    ],
    on="force_slug",
    how="left",
)

if pred["targeted_patrol_units"].isna().any():
    missing_forces = pred.loc[pred["targeted_patrol_units"].isna(), "force_slug"].unique()
    raise ValueError(f"Some forces are missing workforce data: {missing_forces}")


def allocate_for_force_month(
    group: pd.DataFrame,
    force_slug: str,
    police_force_name: str,
    target_month,
) -> pd.DataFrame:

    group = group.copy()

    group["force_slug"] = force_slug
    group["police_force_name"] = police_force_name
    group["target_month"] = target_month

    n_lsoas = len(group)

    total_patrol_units = int(group["targeted_patrol_units"].iloc[0])

    if total_patrol_units <= 0 or n_lsoas <= 0:
        return pd.DataFrame()

    nn_patrol_units = int(round(total_patrol_units * NN_ALLOCATION_SHARE))
    exploration_patrol_units = total_patrol_units - nn_patrol_units

    # -----------------------------
    # Define hotspot candidate areas
    # -----------------------------
    hotspot_lsoa_count = int(np.ceil(n_lsoas * HOTSPOT_LSOA_SHARE))
    hotspot_lsoa_count = max(MIN_HOTSPOT_LSOAS, hotspot_lsoa_count)
    hotspot_lsoa_count = min(hotspot_lsoa_count, n_lsoas, nn_patrol_units)

    group_sorted = group.sort_values("predicted_crime_count", ascending=False)

    hotspot_candidates = group_sorted.head(hotspot_lsoa_count).copy()

    hotspot_candidates["allocation_type"] = "NN_hotspot"
    hotspot_candidates["allocation_reason"] = (
        "Top 10% NN-predicted crime LSOA within this police-force-month."
    )

    hotspot_candidates["selection_group"] = "top_10_percent_nn_hotspot_candidate"

    nn_selected = distribute_integer_units(
        df=hotspot_candidates,
        total_units=nn_patrol_units,
        weight_col="predicted_crime_count",
    )

    # -----------------------------
    # Exploration candidate areas
    # -----------------------------
    remaining = group_sorted.iloc[hotspot_lsoa_count:].copy()

    exploration_selected = pd.DataFrame()

    if exploration_patrol_units > 0 and len(remaining) > 0:
        hotspot_prefixes = set(hotspot_candidates["lsoa_prefix3"].dropna().astype(str))
        hotspot_prefix_nums = hotspot_candidates["lsoa_prefix3_num"].dropna().to_numpy()
        hotspot_last5_nums = hotspot_candidates["lsoa_last5_num"].dropna().to_numpy()

        remaining["risk_rank_pct"] = remaining["predicted_crime_count"].rank(
            pct=True,
            method="average",
        )

        exploration_candidates = remaining[
            (remaining["risk_rank_pct"] >= 0.30)
            & (remaining["risk_rank_pct"] <= 0.90)
        ].copy()

        if len(exploration_candidates) < MIN_EXPLORATION_LSOAS:
            exploration_candidates = remaining.copy()

        exploration_candidates["same_prefix_as_hotspot"] = (
            exploration_candidates["lsoa_prefix3"].astype(str).isin(hotspot_prefixes)
        ).astype(int)

        nearest_prefix_gaps = []
        for prefix_num in exploration_candidates["lsoa_prefix3_num"].to_numpy():
            if pd.isna(prefix_num) or len(hotspot_prefix_nums) == 0:
                nearest_prefix_gaps.append(np.nan)
            else:
                nearest_prefix_gaps.append(np.min(np.abs(hotspot_prefix_nums - prefix_num)))

        exploration_candidates["nearest_hotspot_prefix_gap"] = nearest_prefix_gaps

        nearest_last5_gaps = []
        for last5_num in exploration_candidates["lsoa_last5_num"].to_numpy():
            if pd.isna(last5_num) or len(hotspot_last5_nums) == 0:
                nearest_last5_gaps.append(np.nan)
            else:
                nearest_last5_gaps.append(np.min(np.abs(hotspot_last5_nums - last5_num)))

        exploration_candidates["nearest_hotspot_lsoa_last5_gap"] = nearest_last5_gaps

        for col in ["nearest_hotspot_prefix_gap", "nearest_hotspot_lsoa_last5_gap"]:
            max_value = exploration_candidates[col].max()
            if pd.isna(max_value):
                max_value = 999999
            exploration_candidates[col] = exploration_candidates[col].fillna(max_value)

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

        exploration_candidates["exploration_score"] = (
            RISK_WEIGHT * exploration_candidates["risk_rank_pct"]
            + SAME_PREFIX_WEIGHT * exploration_candidates["same_prefix_as_hotspot"]
            + PREFIX_GAP_WEIGHT * exploration_candidates["prefix_gap_score"]
            + LOCAL_CODE_GAP_WEIGHT * exploration_candidates["local_code_gap_score"]
        )

        same_prefix_candidates = exploration_candidates[
            exploration_candidates["same_prefix_as_hotspot"] == 1
        ].copy()

        if len(same_prefix_candidates) >= MIN_EXPLORATION_LSOAS:
            final_pool = same_prefix_candidates.copy()
            pool_rule = "same_prefix_preferred"
        else:
            final_pool = exploration_candidates.copy()
            pool_rule = "mixed_prefix_fallback"

        exploration_lsoa_count = int(np.ceil(hotspot_lsoa_count * EXPLORATION_TO_HOTSPOT_AREA_RATIO))
        exploration_lsoa_count = max(MIN_EXPLORATION_LSOAS, exploration_lsoa_count)
        exploration_lsoa_count = min(exploration_lsoa_count, len(final_pool), exploration_patrol_units)

        final_pool = final_pool.sample(
            n=exploration_lsoa_count,
            weights=final_pool["exploration_score"].clip(lower=0.01),
            random_state=stable_random_seed(force_slug, target_month),
            replace=False,
        ).copy()

        final_pool["allocation_type"] = "exploration_lsoa_prefix_proxy"
        final_pool["allocation_reason"] = (
            "20% exploration allocation using non-hotspot moderate-risk LSOAs "
            "with LSOA-prefix proximity to NN hotspots."
        )
        final_pool["selection_group"] = "exploration_non_hotspot_prefix_proxy"
        final_pool["exploration_pool_rule"] = pool_rule

        exploration_selected = distribute_integer_units(
            df=final_pool,
            total_units=exploration_patrol_units,
            weight_col="exploration_score",
        )

    selected = pd.concat([nn_selected, exploration_selected], ignore_index=True)

    selected["total_force_month_patrol_units"] = total_patrol_units
    selected["nn_patrol_units_planned"] = nn_patrol_units
    selected["exploration_patrol_units_planned"] = exploration_patrol_units
    selected["number_of_lsoas_in_force_month"] = n_lsoas
    selected["hotspot_lsoa_share"] = HOTSPOT_LSOA_SHARE

    selected["patrol_availability_rate"] = PATROL_AVAILABILITY_RATE
    selected["targeted_patrol_share"] = TARGETED_PATROL_SHARE
    selected["officers_per_patrol_unit"] = OFFICERS_PER_PATROL_UNIT

    return selected


# -----------------------------
# Apply allocation
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
        selected_lsoas=("LSOA code", "count"),
        total_patrol_units_allocated=("patrol_units_allocated", "sum"),
        nn_selected_lsoas=("allocation_type", lambda x: (x == "NN_hotspot").sum()),
        exploration_selected_lsoas=("allocation_type", lambda x: (x == "exploration_lsoa_prefix_proxy").sum()),
        nn_patrol_units_allocated=(
            "patrol_units_allocated",
            lambda x: x[allocation.loc[x.index, "allocation_type"] == "NN_hotspot"].sum(),
        ),
        exploration_patrol_units_allocated=(
            "patrol_units_allocated",
            lambda x: x[allocation.loc[x.index, "allocation_type"] == "exploration_lsoa_prefix_proxy"].sum(),
        ),
        total_predicted_crime_in_selected_lsoas=("predicted_crime_count", "sum"),
        avg_predicted_crime_in_selected_lsoas=("predicted_crime_count", "mean"),
        police_officers=("police_officers", "first"),
        available_patrol_officers=("available_patrol_officers", "first"),
        available_patrol_units=("available_patrol_units", "first"),
        targeted_patrol_units=("targeted_patrol_units", "first"),
        number_of_lsoas_in_force_month=("number_of_lsoas_in_force_month", "first"),
    )
    .reset_index()
)

summary["nn_patrol_unit_share"] = (
    summary["nn_patrol_units_allocated"] / summary["total_patrol_units_allocated"]
)

summary["exploration_patrol_unit_share"] = (
    summary["exploration_patrol_units_allocated"] / summary["total_patrol_units_allocated"]
)

# -----------------------------
# Exploration metrics
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

# -----------------------------
# Force summary
# -----------------------------
force_summary = (
    summary
    .groupby(["force_slug", "police_force_name"])
    .agg(
        avg_monthly_selected_lsoas=("selected_lsoas", "mean"),
        avg_monthly_total_patrol_units=("total_patrol_units_allocated", "mean"),
        avg_monthly_nn_selected_lsoas=("nn_selected_lsoas", "mean"),
        avg_monthly_exploration_selected_lsoas=("exploration_selected_lsoas", "mean"),
        avg_monthly_nn_patrol_units=("nn_patrol_units_allocated", "mean"),
        avg_monthly_exploration_patrol_units=("exploration_patrol_units_allocated", "mean"),
        avg_nn_patrol_unit_share=("nn_patrol_unit_share", "mean"),
        avg_exploration_patrol_unit_share=("exploration_patrol_unit_share", "mean"),
        avg_exploration_same_prefix_share=("exploration_same_prefix_share", "mean"),
        avg_nearest_hotspot_prefix_gap=("avg_nearest_hotspot_prefix_gap", "mean"),
        avg_nearest_hotspot_lsoa_last5_gap=("avg_nearest_hotspot_lsoa_last5_gap", "mean"),
        police_officers=("police_officers", "first"),
        available_patrol_units=("available_patrol_units", "first"),
        targeted_patrol_units=("targeted_patrol_units", "first"),
    )
    .reset_index()
    .sort_values("avg_monthly_total_patrol_units", ascending=False)
)

# -----------------------------
# Save
# -----------------------------
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

allocation.to_csv(OUTPUT_PATH, index=False)
summary.to_csv(SUMMARY_OUTPUT_PATH, index=False)
force_summary.to_csv(FORCE_SUMMARY_OUTPUT_PATH, index=False)

print("\nSaved V5 unit-intensity allocation to:")
print(OUTPUT_PATH)

print("\nSaved monthly summary to:")
print(SUMMARY_OUTPUT_PATH)

print("\nSaved force-level summary to:")
print(FORCE_SUMMARY_OUTPUT_PATH)

print("\nForce-level preview:")
pd.set_option("display.max_columns", None)
print(force_summary)
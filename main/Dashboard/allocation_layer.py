from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Patrol Allocation Dashboard Layer
# Default method = V5 unit-intensity allocation


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = PROJECT_ROOT/ "main" /"data"/ "outputs" / "model_predictions_by_lsoa_month.csv"
WORKFORCE_PATH = PROJECT_ROOT / "main" /"data"/"for dashboard"/"police_workforce_selected_forces.csv"

DEFAULT_CONFIG = {
    "show_allocation": False,
    "use_v5_defaults": True,
    "officer_count_multiplier": 1.00,
    "additional_officers_per_force": 0,
    "patrol_availability_rate": 0.30,
    "officers_per_patrol_unit": 2,
    "targeted_patrol_share": 1.00,
    "nn_allocation_share": 0.80,
    "hotspot_lsoa_share": 0.10,
}

RANDOM_SEED = 42
MIN_HOTSPOT_LSOAS = 5
MIN_EXPLORATION_LSOAS = 1

# Exploration scoring weights. These match the V5 allocation logic.
RISK_WEIGHT = 0.50
SAME_PREFIX_WEIGHT = 0.30
PREFIX_GAP_WEIGHT = 0.15
LOCAL_CODE_GAP_WEIGHT = 0.05


# Utility functions

def stable_random_seed(force_slug: str, target_month) -> int:
    """Create a stable seed per police-force/month so results are reproducible."""
    key = f"{force_slug}_{pd.Timestamp(target_month).date()}_{RANDOM_SEED}"
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)


def extract_lsoa_digits(lsoa_code) -> str | float:
    """Extract numeric part of an LSOA code. Example: E01033010 -> 01033010."""
    if pd.isna(lsoa_code):
        return np.nan

    digits = "".join(ch for ch in str(lsoa_code) if ch.isdigit())
    return digits if digits else np.nan


def add_lsoa_proxy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add LSOA-code proxy columns for the exploration part.

    This is not real geographic distance. It is a fallback proximity proxy
    because the processed model dataset does not contain latitude/longitude.
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


def distribute_integer_units(
    df: pd.DataFrame,
    total_units: int,
    weight_col: str,
) -> pd.DataFrame:
    """
    Distribute integer patrol units across selected LSOAs.

    Higher weight = more patrol units. The total allocated equals total_units.
    If there are more selected LSOAs than units, only the strongest rows remain.
    """
    df = df.copy()
    total_units = int(round(total_units))

    if total_units <= 0 or df.empty:
        return df.iloc[0:0].copy()

    if len(df) > total_units:
        df = df.sort_values(weight_col, ascending=False).head(total_units).copy()

    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0).clip(lower=0)
    if weights.sum() <= 0:
        weights = pd.Series(1.0, index=df.index)

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

# Data loading

@st.cache_data(show_spinner=False)
def load_allocation_base_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load model predictions and workforce data for allocation."""
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PREDICTIONS_PATH}")

    if not WORKFORCE_PATH.exists():
        raise FileNotFoundError(f"Missing workforce file: {WORKFORCE_PATH}")

    pred = pd.read_csv(PREDICTIONS_PATH)
    workforce = pd.read_csv(WORKFORCE_PATH)

    required_pred_cols = {
        "LSOA code",
        "target_month",
        "force_slug",
        "police_force_name",
        "actual_crime_count",
        "predicted_crime_count",
    }
    required_workforce_cols = {
        "force_slug",
        "police_officers",
        "total_paid_workforce",
        "officers_per_100k_population",
    }

    missing_pred = sorted(required_pred_cols - set(pred.columns))
    missing_workforce = sorted(required_workforce_cols - set(workforce.columns))

    if missing_pred:
        raise ValueError(f"Prediction file missing columns: {missing_pred}")

    if missing_workforce:
        raise ValueError(f"Workforce file missing columns: {missing_workforce}")

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

    return pred, workforce

# Sidebar controls

def render_allocation_sidebar_controls(sidebar) -> dict:
    """Render patrol allocation controls and return parameter dictionary."""
    sidebar.markdown("---")
    sidebar.header("Patrol Allocation Settings")

    params = DEFAULT_CONFIG.copy()

    params["show_allocation"] = sidebar.checkbox(
        "Show patrol allocation map",
        value=DEFAULT_CONFIG["show_allocation"],
    )

    if not params["show_allocation"]:
        return params

    params["use_v5_defaults"] = sidebar.checkbox(
        "Use V5 default settings",
        value=True,
        help=(
            "Default V5: 30% officer availability, 2 officers per unit, "
            "100% targeted patrol share, 80/20 NN/exploration split, "
            "top 10% LSOAs as hotspot candidates."
        ),
    )

    if params["use_v5_defaults"]:
        sidebar.info(
            "Using V5 defaults: 30% patrol availability, 2 officers per patrol unit, "
            "top 10% LSOAs as hotspots, 80% NN allocation, 20% exploration."
        )
        return params

    with sidebar.expander("Advanced patrol scenario controls", expanded=True):
        params["officer_count_multiplier"] = sidebar.slider(
            "Officer count multiplier",
            min_value=0.25,
            max_value=2.00,
            value=DEFAULT_CONFIG["officer_count_multiplier"],
            step=0.05,
            help="Scales the official officer count for scenario testing.",
        )

        params["additional_officers_per_force"] = sidebar.number_input(
            "Additional officers per force",
            min_value=-10000,
            max_value=10000,
            value=DEFAULT_CONFIG["additional_officers_per_force"],
            step=50,
            help="Adds or removes the same number of officers from each force for scenario testing.",
        )

        params["patrol_availability_rate"] = sidebar.slider(
            "Patrol availability rate (%)",
            min_value=5,
            max_value=80,
            value=int(DEFAULT_CONFIG["patrol_availability_rate"] * 100),
            step=5,
            help="Share of officers assumed available for patrol-related work.",
        ) / 100

        params["officers_per_patrol_unit"] = sidebar.number_input(
            "Officers per patrol unit",
            min_value=1,
            max_value=6,
            value=DEFAULT_CONFIG["officers_per_patrol_unit"],
            step=1,
        )

        params["targeted_patrol_share"] = sidebar.slider(
            "Patrol share controlled by model (%)",
            min_value=10,
            max_value=100,
            value=int(DEFAULT_CONFIG["targeted_patrol_share"] * 100),
            step=5,
            help="Share of available patrol units used in this data-driven allocation model.",
        ) / 100

        params["nn_allocation_share"] = sidebar.slider(
            "NN allocation share (%)",
            min_value=50,
            max_value=100,
            value=int(DEFAULT_CONFIG["nn_allocation_share"] * 100),
            step=5,
            help="Share of patrol units allocated to neural-network hotspot LSOAs.",
        ) / 100

        params["hotspot_lsoa_share"] = sidebar.slider(
            "Hotspot LSOA share (%)",
            min_value=1,
            max_value=30,
            value=int(DEFAULT_CONFIG["hotspot_lsoa_share"] * 100),
            step=1,
            help="Top percentage of LSOAs by predicted crime used as hotspot candidates.",
        ) / 100

    sidebar.caption(
        f"Exploration allocation share: {1 - params['nn_allocation_share']:.0%}"
    )
    return params


# Allocation engine

def calculate_workforce_capacity(
    workforce: pd.DataFrame,
    officer_count_multiplier: float,
    additional_officers_per_force: int,
    patrol_availability_rate: float,
    officers_per_patrol_unit: int,
    targeted_patrol_share: float,
) -> pd.DataFrame:
    """Calculate patrol capacity for each force under the selected scenario."""
    workforce = workforce.copy()

    workforce["scenario_police_officers"] = (
        workforce["police_officers"] * officer_count_multiplier
        + additional_officers_per_force
    ).clip(lower=0)

    workforce["available_patrol_officers"] = (
        workforce["scenario_police_officers"] * patrol_availability_rate
    )

    workforce["available_patrol_units"] = (
        workforce["available_patrol_officers"] / max(1, officers_per_patrol_unit)
    )

    workforce["targeted_patrol_units"] = (
        workforce["available_patrol_units"] * targeted_patrol_share
    ).round().astype(int).clip(lower=0)

    return workforce


def allocate_for_force_month(
    group: pd.DataFrame,
    force_slug: str,
    police_force_name: str,
    target_month,
    nn_allocation_share: float,
    hotspot_lsoa_share: float,
) -> pd.DataFrame:
    """Apply V5 unit-intensity allocation for one force-month."""
    group = group.copy()
    group["force_slug"] = force_slug
    group["police_force_name"] = police_force_name
    group["target_month"] = target_month

    n_lsoas = len(group)
    total_patrol_units = int(group["targeted_patrol_units"].iloc[0])

    if total_patrol_units <= 0 or n_lsoas <= 0:
        return pd.DataFrame()

    nn_patrol_units = int(round(total_patrol_units * nn_allocation_share))
    nn_patrol_units = min(max(nn_patrol_units, 0), total_patrol_units)
    exploration_patrol_units = total_patrol_units - nn_patrol_units

    group_sorted = group.sort_values("predicted_crime_count", ascending=False)

    # Hotspots = top 10% by default, or dashboard-selected percentage.
    hotspot_lsoa_count = int(np.ceil(n_lsoas * hotspot_lsoa_share))
    hotspot_lsoa_count = max(MIN_HOTSPOT_LSOAS, hotspot_lsoa_count)
    hotspot_lsoa_count = min(hotspot_lsoa_count, n_lsoas, max(nn_patrol_units, 1))

    hotspot_candidates = group_sorted.head(hotspot_lsoa_count).copy()
    hotspot_candidates["allocation_type"] = "NN_hotspot"
    hotspot_candidates["allocation_reason"] = "Top NN-predicted hotspot LSOA."
    hotspot_candidates["selection_group"] = "top_nn_hotspot_candidate"

    nn_selected = distribute_integer_units(
        df=hotspot_candidates,
        total_units=nn_patrol_units,
        weight_col="predicted_crime_count",
    )

    remaining = group_sorted.iloc[hotspot_lsoa_count:].copy()
    exploration_selected = pd.DataFrame()

    if exploration_patrol_units > 0 and not remaining.empty:
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

        if exploration_candidates.empty:
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

        if not same_prefix_candidates.empty:
            final_pool = same_prefix_candidates.copy()
            pool_rule = "same_prefix_preferred"
        else:
            final_pool = exploration_candidates.copy()
            pool_rule = "mixed_prefix_fallback"

        exploration_to_hotspot_ratio = (
            (1 - nn_allocation_share) / nn_allocation_share
            if nn_allocation_share > 0
            else 1.0
        )
        exploration_lsoa_count = int(np.ceil(hotspot_lsoa_count * exploration_to_hotspot_ratio))
        exploration_lsoa_count = max(MIN_EXPLORATION_LSOAS, exploration_lsoa_count)
        exploration_lsoa_count = min(
            exploration_lsoa_count,
            len(final_pool),
            exploration_patrol_units,
        )

        final_pool = final_pool.sample(
            n=exploration_lsoa_count,
            weights=final_pool["exploration_score"].clip(lower=0.01),
            random_state=stable_random_seed(force_slug, target_month),
            replace=False,
        ).copy()

        final_pool["allocation_type"] = "exploration_lsoa_prefix_proxy"
        final_pool["allocation_reason"] = (
            "Exploration allocation using non-hotspot moderate-risk LSOAs "
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
    selected["hotspot_lsoa_share"] = hotspot_lsoa_share

    return selected


@st.cache_data(show_spinner=False)
def compute_patrol_allocation_for_month(
    selected_month_str: str,
    officer_count_multiplier: float,
    additional_officers_per_force: int,
    patrol_availability_rate: float,
    officers_per_patrol_unit: int,
    targeted_patrol_share: float,
    nn_allocation_share: float,
    hotspot_lsoa_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute patrol allocation for the selected month and current scenario settings."""
    pred, workforce = load_allocation_base_data()
    selected_month = pd.to_datetime(selected_month_str)

    pred = pred[pred["target_month"] == selected_month].copy()
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()

    workforce = calculate_workforce_capacity(
        workforce=workforce,
        officer_count_multiplier=officer_count_multiplier,
        additional_officers_per_force=additional_officers_per_force,
        patrol_availability_rate=patrol_availability_rate,
        officers_per_patrol_unit=officers_per_patrol_unit,
        targeted_patrol_share=targeted_patrol_share,
    )

    pred = pred.merge(
        workforce[
            [
                "force_slug",
                "police_officers",
                "scenario_police_officers",
                "available_patrol_officers",
                "available_patrol_units",
                "targeted_patrol_units",
            ]
        ],
        on="force_slug",
        how="left",
    )

    pred = pred.dropna(subset=["targeted_patrol_units"])
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()

    allocation_parts = []
    for (force_slug, police_force_name, target_month), group in pred.groupby(
        ["force_slug", "police_force_name", "target_month"]
    ):
        allocated_group = allocate_for_force_month(
            group=group,
            force_slug=force_slug,
            police_force_name=police_force_name,
            target_month=target_month,
            nn_allocation_share=nn_allocation_share,
            hotspot_lsoa_share=hotspot_lsoa_share,
        )
        if not allocated_group.empty:
            allocation_parts.append(allocated_group)

    if not allocation_parts:
        return pd.DataFrame(), pd.DataFrame()

    allocation = pd.concat(allocation_parts, ignore_index=True)
    allocation["nn_patrol_units"] = np.where(
        allocation["allocation_type"] == "NN_hotspot",
        allocation["patrol_units_allocated"],
        0,
    )
    allocation["exploration_patrol_units"] = np.where(
        allocation["allocation_type"] == "exploration_lsoa_prefix_proxy",
        allocation["patrol_units_allocated"],
        0,
    )

    summary = (
        allocation
        .groupby(["force_slug", "police_force_name"], as_index=False)
        .agg(
            selected_lsoas=("LSOA code", "count"),
            total_patrol_units=("patrol_units_allocated", "sum"),
            nn_patrol_units=("nn_patrol_units", "sum"),
            exploration_patrol_units=("exploration_patrol_units", "sum"),
            avg_predicted_crime=("predicted_crime_count", "mean"),
            available_patrol_units=("available_patrol_units", "first"),
            targeted_patrol_units=("targeted_patrol_units", "first"),
            scenario_police_officers=("scenario_police_officers", "first"),
        )
    )

    summary["nn_share"] = summary["nn_patrol_units"] / summary["total_patrol_units"]
    summary["exploration_share"] = (
        summary["exploration_patrol_units"] / summary["total_patrol_units"]
    )

    return allocation, summary


# Streamlit render function

def render_patrol_allocation_section(
    df_map_filtered: pd.DataFrame,
    geojson,
    viewport: dict,
    selected_month_ts,
    scope_text: str,
    params: dict,
) -> None:
    """Render patrol allocation map, tables, and CSV download inside the dashboard."""
    if not params.get("show_allocation", False):
        return

    month_label = pd.to_datetime(selected_month_ts).strftime("%Y-%m")
    exploration_share = 1 - params["nn_allocation_share"]

    st.markdown("---")
    st.subheader(f"Patrol Resource Allocation ({month_label})")
    st.caption("Allocation layer version: V5 defaults + tabs + CSV download")

    allocation, summary = compute_patrol_allocation_for_month(
        selected_month_str=pd.to_datetime(selected_month_ts).strftime("%Y-%m-%d"),
        officer_count_multiplier=params["officer_count_multiplier"],
        additional_officers_per_force=int(params["additional_officers_per_force"]),
        patrol_availability_rate=params["patrol_availability_rate"],
        officers_per_patrol_unit=int(params["officers_per_patrol_unit"]),
        targeted_patrol_share=params["targeted_patrol_share"],
        nn_allocation_share=params["nn_allocation_share"],
        hotspot_lsoa_share=params["hotspot_lsoa_share"],
    )

    if allocation.empty:
        st.warning("No patrol allocation rows found for this selected month.")
        return

    allocation_map = (
        allocation
        .groupby("LSOA code", as_index=False)
        .agg(
            patrol_units_allocated=("patrol_units_allocated", "sum"),
            allocation_type=("allocation_type", "first"),
            predicted_crime_count=("predicted_crime_count", "max"),
            actual_crime_count=("actual_crime_count", "max"),
            police_force_name=("police_force_name", "first"),
            force_slug=("force_slug", "first"),
        )
        .rename(columns={"LSOA code": "LSOA_code"})
    )

    df_alloc_map = df_map_filtered.merge(
        allocation_map,
        left_on="LSOA21CD",
        right_on="LSOA_code",
        how="left",
    )

    df_alloc_map["patrol_units_allocated"] = df_alloc_map["patrol_units_allocated"].fillna(0)
    df_alloc_map["predicted_crime_count"] = df_alloc_map["predicted_crime_count"].fillna(0)
    df_alloc_map["actual_crime_count"] = df_alloc_map["actual_crime_count"].fillna(0)
    df_alloc_map["allocation_type"] = df_alloc_map["allocation_type"].fillna("not_selected")
    df_alloc_map["police_force_name"] = df_alloc_map["police_force_name"].fillna("Not allocated")

    selected_lsoas_on_map = int((df_alloc_map["patrol_units_allocated"] > 0).sum())
    total_units_on_map = int(df_alloc_map["patrol_units_allocated"].sum())
    total_units_all_forces = int(allocation["patrol_units_allocated"].sum())

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Selected LSOAs on map", selected_lsoas_on_map)
    metric_col2.metric("Patrol units on map", total_units_on_map)
    metric_col3.metric("All-force patrol units", total_units_all_forces)
    metric_col4.metric(
        "NN / exploration",
        f"{params['nn_allocation_share']:.0%} / {exploration_share:.0%}",
    )

    if selected_lsoas_on_map == 0:
        st.warning(
            "No allocated LSOAs matched the current map scope. This may be a city filter issue or an LSOA code-version mismatch."
        )

    tab_map, tab_tables, tab_download = st.tabs(
        ["Allocation Map", "Allocation Tables", "Download CSV"]
    )

    with tab_map:
        fig_alloc = px.choropleth_mapbox(
            df_alloc_map,
            geojson=geojson,
            locations="LSOA21CD",
            featureidkey="properties.LSOA21CD",
            color="patrol_units_allocated",
            color_continuous_scale="Blues",
            mapbox_style="carto-positron",
            zoom=viewport["zoom"],
            center={"lat": viewport["lat"], "lon": viewport["lon"]},
            opacity=0.65,
            labels={
                "patrol_units_allocated": "Patrol units",
                "LSOA21NM": "Zone name",
            },
            hover_data=[
                "LSOA21NM",
                "City_Hub",
                "police_force_name",
                "allocation_type",
                "patrol_units_allocated",
                "predicted_crime_count",
                "actual_crime_count",
            ],
        )

        fig_alloc.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=600,
        )

        st.plotly_chart(fig_alloc, use_container_width=True)

    with tab_tables:
        st.markdown("### Patrol allocation summary by police force")

        st.dataframe(
            summary.sort_values("total_patrol_units", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Top allocated LSOAs")

        top_lsoas = (
            allocation_map
            .sort_values("patrol_units_allocated", ascending=False)
            .head(50)
        )

        st.dataframe(
            top_lsoas[
                [
                    "LSOA_code",
                    "police_force_name",
                    "allocation_type",
                    "patrol_units_allocated",
                    "predicted_crime_count",
                    "actual_crime_count",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab_download:
        st.markdown("### Download patrol allocation output")

        allocation_download = allocation.copy().rename(
            columns={"LSOA code": "LSOA_code"}
        )

        download_cols = [
            "LSOA_code",
            "target_month",
            "force_slug",
            "police_force_name",
            "allocation_type",
            "patrol_units_allocated",
            "predicted_crime_count",
            "actual_crime_count",
            "total_force_month_patrol_units",
            "nn_patrol_units_planned",
            "exploration_patrol_units_planned",
            "number_of_lsoas_in_force_month",
            "hotspot_lsoa_share",
        ]

        available_download_cols = [
            col for col in download_cols if col in allocation_download.columns
        ]

        allocation_download = allocation_download[available_download_cols].copy()

        allocation_csv = allocation_download.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download LSOA patrol allocation CSV",
            data=allocation_csv,
            file_name=f"patrol_allocation_{month_label.replace('-', '_')}.csv",
            mime="text/csv",
        )

        st.markdown(
            "This CSV shows how many patrol units are allocated to each selected LSOA."
        )

        st.dataframe(
            allocation_download.sort_values(
                "patrol_units_allocated",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )


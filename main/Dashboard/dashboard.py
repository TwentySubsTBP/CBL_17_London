"""
Tactical Crime Forecasting Dashboard
------------------------------------
Adapted to run on `model_predictions_by_lsoa_month_dashboard.csv`, which has columns:
    LSOA code, Month, actual, predicted

This single file powers everything:
  * Map layer        -> predicted crime for the selected month (as a 0-1
                        hotspot-risk score, or raw predicted count)
  * Trend analytics  -> historical `actual` crime over all months
  * Performance       -> real MAE / RMSE from actual vs predicted, plus a
                        persistence baseline for honest comparison

Geometry is NOT in the CSV, so a geojson skeleton (keyed by LSOA21CD) is
assumed to exist, matching the original dashboard's conventions. Adjust the
CONFIG block below to point at your files.

Run with:  streamlit run dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Allocation layer is optional: keep the app runnable if the module is absent.
try:
    from allocation_layer import (
        render_allocation_sidebar_controls,
        render_patrol_allocation_section,
    )
    HAS_ALLOCATION = True
except Exception:
    HAS_ALLOCATION = False

# --------------------------------------------------------------------------
# CONFIG  -- point these at your files
# --------------------------------------------------------------------------
# Paths are resolved relative to THIS script's folder, so it works no matter
# what directory you launch `streamlit run` from.
BASE_DIR = Path(__file__).resolve().parent


def resolve_path(filename):
    """Find a file next to this script, in the current dir, or one level up."""
    candidates = [BASE_DIR / filename, Path.cwd() / filename, BASE_DIR.parent / filename]
    for c in candidates:
        if c.exists():
            return c
    return BASE_DIR / filename  # default for a clear "not found" message


PREDICTIONS_CSV = resolve_path("../data/for dashboard/model_predictions_by_lsoa_month_dashboard.csv")
GEOJSON_PATH = resolve_path("../data/for dashboard/major_cities_2021_skeleton.geojson")
FEATURE_ID_KEY = "properties.LSOA21CD"
PROP_CODE, PROP_NAME, PROP_LA = "LSOA21CD", "LSOA21NM", "LocalAuthority"

LSOA_COL, MONTH_COL = "LSOA code", "Month"

st.set_page_config(layout="wide", page_title="Tactical Crime Forecasting Dashboard")
st.title("Tactical Crime Forecasting Dashboard")
st.markdown("Powered by neural-network crime-count predictions and historical timelines")

LONDON_BOROUGHS = [
    "City of London", "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
    "Camden", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney", "Hammersmith and Fulham",
    "Haringey", "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington",
    "Kensington and Chelsea", "Kingston upon Thames", "Lambeth", "Lewisham", "Merton",
    "Newham", "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
    "Waltham Forest", "Wandsworth", "Westminster",
]

CITY_VIEWPORTS = {
    "All Cities": {"lat": 53.1, "lon": -1.6, "zoom": 6.0},
    "London": {"lat": 51.5074, "lon": -0.1278, "zoom": 9.5},
    "Birmingham": {"lat": 52.4862, "lon": -1.8904, "zoom": 11.0},
    "Manchester": {"lat": 53.4808, "lon": -2.2426, "zoom": 11.0},
    "Liverpool": {"lat": 53.4084, "lon": -2.9916, "zoom": 11.0},
    "Leeds": {"lat": 53.8008, "lon": -1.5491, "zoom": 10.5},
    "Sheffield": {"lat": 53.3811, "lon": -1.4701, "zoom": 11.0},
    "Newcastle upon Tyne": {"lat": 54.9783, "lon": -1.6178, "zoom": 11.0},
    "Leicester": {"lat": 52.6369, "lon": -1.1398, "zoom": 11.5},
    "Nottingham": {"lat": 52.9548, "lon": -1.1581, "zoom": 11.5},
    "Hull": {"lat": 53.7443, "lon": -0.3325, "zoom": 11.5},
}


def normalize_to_hub(la_name):
    if la_name in LONDON_BOROUGHS:
        return "London"
    if la_name in ("Kingston upon Hull", "Hull"):
        return "Hull"
    return la_name


# --------------------------------------------------------------------------
# DATA PIPELINE
# --------------------------------------------------------------------------
@st.cache_data
def load_dashboard_datasets():
    if not Path(PREDICTIONS_CSV).exists():
        st.error(
            f"Predictions CSV not found. Expected at:\n\n`{PREDICTIONS_CSV}`\n\n"
            "Place `model_predictions_by_lsoa_month_dashboard.csv` in the same folder as this "
            "script (main/Dashboard/), or edit PREDICTIONS_CSV in the CONFIG block."
        )
        st.stop()

    preds = pd.read_csv(PREDICTIONS_CSV)
    preds[LSOA_COL] = preds[LSOA_COL].astype(str).str.strip()
    preds[MONTH_COL] = pd.to_datetime(preds[MONTH_COL])
    preds = preds.rename(columns={"actual": "crime_count", "predicted": "predicted_count"})

    geojson_data, master_geo_df = None, None
    if Path(GEOJSON_PATH).exists():
        with open(GEOJSON_PATH, "r") as f:
            geojson_data = json.load(f)
        records = []
        for feature in geojson_data["features"]:
            props = feature["properties"]
            records.append({
                PROP_CODE: props.get(PROP_CODE),
                PROP_NAME: props.get(PROP_NAME),
                PROP_LA: props.get(PROP_LA),
            })
        master_geo_df = pd.DataFrame(records)
        master_geo_df[PROP_CODE] = master_geo_df[PROP_CODE].astype(str).str.strip()
        master_geo_df["City_Hub"] = master_geo_df[PROP_LA].apply(normalize_to_hub)

    return preds, master_geo_df, geojson_data


preds, master_geo_df, geojson = load_dashboard_datasets()

if geojson is None:
    st.error(
        f"Geojson not found at '{GEOJSON_PATH}'. The map and city grouping need it. "
        "Update GEOJSON_PATH in the CONFIG block."
    )
    st.stop()


def within_scope_risk(frame, value_col="predicted_count"):
    """Percentile-rank predictions to a 0-1 hotspot-risk score within the frame."""
    if len(frame) <= 1:
        frame = frame.copy()
        frame["hotspot_probability"] = 0.0
        return frame
    frame = frame.copy()
    frame["hotspot_probability"] = frame[value_col].rank(pct=True)
    return frame


# --------------------------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------------------------
st.sidebar.header("Operational Scope")
raw_cities = sorted({c for c in master_geo_df["City_Hub"].dropna().unique() if c})
selected_city = st.sidebar.selectbox("Select City", ["All Cities"] + raw_cities, index=0)

map_metric = st.sidebar.radio(
    "Map colour", ["Hotspot risk (0-1)", "Predicted crime count"], index=0
)

st.sidebar.markdown("---")
st.sidebar.header("Temporal Scope")
available_months = sorted(preds[MONTH_COL].unique())
month_strings = [pd.to_datetime(m).strftime("%Y-%m") for m in available_months]
selected_month_str = st.sidebar.selectbox(
    "Select Target Operational Month", month_strings, index=len(month_strings) - 1
)
selected_month_ts = pd.to_datetime(selected_month_str)

allocation_params = (
    render_allocation_sidebar_controls(st.sidebar) if HAS_ALLOCATION else None
)

# --------------------------------------------------------------------------
# BUILD SCOPED FRAMES
# --------------------------------------------------------------------------
# trends: full history for the selected city
preds_geo = preds.merge(master_geo_df, left_on=LSOA_COL, right_on=PROP_CODE, how="inner")
if selected_city != "All Cities":
    preds_geo = preds_geo[preds_geo["City_Hub"] == selected_city]
    scope_text = selected_city
else:
    scope_text = "Nationwide Hubs"

# map: selected month, merged onto the full geographic skeleton
month_preds = preds[preds[MONTH_COL] == selected_month_ts]
df_map = master_geo_df.merge(month_preds, left_on=PROP_CODE, right_on=LSOA_COL, how="left")
df_map["matched_in_csv"] = df_map[LSOA_COL].notna()
df_map["predicted_count"] = df_map["predicted_count"].fillna(0.0)
df_map["crime_count"] = df_map["crime_count"].fillna(0.0)
if selected_city != "All Cities":
    df_map = df_map[df_map["City_Hub"] == selected_city]
df_map = within_scope_risk(df_map)

viewport = CITY_VIEWPORTS.get(selected_city, {"lat": 52.5, "lon": -1.5, "zoom": 6.0})

with st.sidebar.expander("Data Integrity Diagnostics"):
    st.write(f"Total map polygons: {len(df_map):,}")
    st.write(f"Matched to prediction rows: {int(df_map['matched_in_csv'].sum()):,}")
    st.write(f"Historical data points (selected city): {len(preds_geo):,}")
    st.write(f"Predicted crime in {selected_month_str}: {df_map['predicted_count'].sum():,.0f}")

# =====================================================================
# ROW 1: CHOROPLETH MAP
# =====================================================================
st.subheader(f"{scope_text} Strategic Crime Hotspot Map ({selected_month_str})")

if map_metric.startswith("Hotspot"):
    color_col, color_label, crange, cscale = "hotspot_probability", "Hotspot Risk", [0, 1], "Reds"
else:
    color_col, color_label, crange, cscale = "predicted_count", "Predicted crimes", None, "Reds"

fig_map = px.choropleth_mapbox(
    df_map,
    geojson=geojson,
    locations=PROP_CODE,
    featureidkey=FEATURE_ID_KEY,
    color=color_col,
    color_continuous_scale=cscale,
    range_color=crange,
    mapbox_style="carto-positron",
    zoom=viewport["zoom"],
    center={"lat": viewport["lat"], "lon": viewport["lon"]},
    opacity=0.6,
    labels={color_col: color_label, PROP_NAME: "Zone Name"},
    hover_data=[PROP_NAME, "predicted_count", "crime_count", "City_Hub"],
)
fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600)
st.plotly_chart(fig_map, use_container_width=True)

if HAS_ALLOCATION:
    try:
        render_patrol_allocation_section(
            df_map_filtered=df_map, geojson=geojson, viewport=viewport,
            selected_month_ts=selected_month_ts, scope_text=scope_text, params=allocation_params,
        )
    except FileNotFoundError as e:
        st.info(
            "Patrol-allocation layer skipped — a data file it needs is missing:\n\n"
            f"`{e}`\n\n"
            "The map, trends, and performance diagnostics below work without it. "
            "Add that file (or fix its path in allocation_layer.py) to enable allocation."
        )
    except Exception as e:
        st.exception(e)  # full traceback with the real file + line number

st.markdown("---")

# =====================================================================
# ROW 2: TACTICAL TREND ANALYTICS
# =====================================================================
st.subheader(f"Tactical Trend Analytics (Contextualised to {selected_month_str})")
c1, c2 = st.columns(2)

with c1:
    top_codes = (
        df_map.sort_values("hotspot_probability", ascending=False).head(5)[PROP_CODE].values
    )
    series = preds_geo[preds_geo[LSOA_COL].isin(top_codes)].sort_values(MONTH_COL)
    fig = px.line(
        series, x=MONTH_COL, y="crime_count", color=PROP_NAME, markers=True,
        title=f"Historical Crime Volume for Top 5 Hotspot Zones ({scope_text})",
        labels={"crime_count": "Monthly Crimes", MONTH_COL: "Month", PROP_NAME: "Zone Name"},
    )
    fig.add_vline(x=selected_month_ts, line_width=2, line_dash="dash", line_color="black")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    agg = preds_geo.groupby(MONTH_COL)["crime_count"].sum().reset_index().sort_values(MONTH_COL)
    fig = px.line(
        agg, x=MONTH_COL, y="crime_count", markers=True,
        title=f"Aggregate Monthly Crime Trend ({scope_text})",
        labels={"crime_count": "Total Crime Count", MONTH_COL: "Month"},
    )
    fig.add_vline(x=selected_month_ts, line_width=2, line_dash="dash", line_color="black")
    fig.update_traces(line_color="#2b5c8f")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=350)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# =====================================================================
# ROW 3: PERFORMANCE DIAGNOSTICS (real MAE / RMSE vs baseline)
# =====================================================================
st.subheader("Model Performance Metrics & Error Diagnostics")

# evaluate on the selected city's full history (actual vs predicted)
ev = preds_geo.dropna(subset=["crime_count", "predicted_count"]).copy()
ev["abs_error"] = (ev["predicted_count"] - ev["crime_count"]).abs()
ev["sq_error"] = (ev["predicted_count"] - ev["crime_count"]) ** 2

mae = ev["abs_error"].mean() if len(ev) else 0.0
rmse = np.sqrt(ev["sq_error"].mean()) if len(ev) else 0.0

# persistence baseline within scope: predict previous month's actual
base = ev.sort_values([LSOA_COL, MONTH_COL]).copy()
base["persist"] = base.groupby(LSOA_COL)["crime_count"].shift(1)
base = base.dropna(subset=["persist"])
base_mae = (base["persist"] - base["crime_count"]).abs().mean() if len(base) else float("nan")
improve = (base_mae - mae) / base_mae * 100 if base_mae and not np.isnan(base_mae) else float("nan")

m1, m2, m3 = st.columns(3)
m1.metric("Model MAE", f"{mae:.3f}")
m2.metric("Model RMSE", f"{rmse:.3f}")
m3.metric("vs persistence baseline", f"{improve:+.1f}%" if not np.isnan(improve) else "n/a",
          help="Percent lower MAE than predicting last month's count. Positive = better than the naive baseline.")

p1, p2 = st.columns(2)
with p1:
    by_lsoa = (
        ev.groupby(PROP_NAME, observed=False)["abs_error"].mean()
        .reset_index(name="MAE").sort_values("MAE", ascending=False).head(10)
    )
    fig = px.bar(
        by_lsoa, x="MAE", y=PROP_NAME, orientation="h", color="MAE", color_continuous_scale="Reds",
        title=f"Top 10 High-Error Zones (Overall MAE: {mae:.3f})",
        labels={PROP_NAME: "LSOA Zone", "MAE": "Mean Absolute Error"},
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380,
                      coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with p2:
    fig = px.histogram(
        ev, x="abs_error", nbins=40, color_discrete_sequence=["#2b5c8f"],
        title="Absolute Error Distribution",
        labels={"abs_error": "Absolute Error  |predicted - actual|"},
    )
    fig.add_vline(x=mae, line_width=3, line_dash="dash", line_color="#d95f02",
                  annotation_text="  Mean abs error", annotation_position="top right")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=380, yaxis_title="Count of LSOA-months")
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Error metrics use actual vs predicted counts from the predictions file, evaluated over "
    "the selected city's full history. The persistence baseline is the honest reference: a model "
    "is only adding value when it beats 'next month = last month'."
)
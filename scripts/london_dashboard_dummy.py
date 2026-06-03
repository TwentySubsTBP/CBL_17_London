import streamlit as st
import pandas as pd
import plotly.express as px
import json

# Set up the page layout to widescreen
st.set_page_config(layout="wide")

st.title("🚨 London Tactical Crime Forecasting Dashboard")
st.markdown("Powered by Spatial CNN Hotspot Predictions")


# --- DATA LOADING (Cached for speed) ---
@st.cache_data
def load_london_data():
    # Load CNN outputs
    df = pd.read_csv("scripts/london_predictions.csv")

    # Load simplified LSOA boundaries
    with open("scripts/london_lsoa_simplified.json", "r") as f:
        geojson = json.load(f)

    return df, geojson


df, geojson = load_london_data()

# --- SIDEBAR: TACTICAL DISPATCH ADVICE ---
st.sidebar.header("⚡ High-Priority Dispatches")
st.sidebar.markdown("Top 3 predicted LSOA hotspots requiring immediate proactive patrol:")

# Get the top 3 highest risk LSOAs
top_hotspots = df.sort_values(by="hotspot_probability", ascending=False).head(3)

for i, (_, row) in enumerate(top_hotspots.iterrows(), 1):
    with st.sidebar.container():
        st.sidebar.error(f"**Priority {i}: Zone {row['LSOA_code']}**")
        st.sidebar.write(f"📈 **Hotspot Probability:** {row['hotspot_probability']:.1%}")
        st.sidebar.caption(
            "💡 *Advice:* Deploy high-visibility foot patrols. Focus on transit exit points during shift crossover.")
        st.sidebar.markdown("---")

# --- MAIN SCREEN: THE MAP ---
import plotly.express as px

# --- MAP RENDERING SECTION ---
st.subheader("📍 London Crime Hotspot Map")

# Double-check that your columns don't have hidden spaces
df['LSOA_code'] = df['LSOA_code'].str.strip()

fig = px.choropleth_mapbox(
    df,
    geojson=geojson,
    locations="LSOA_code",               # The column name in your CSV data table
    featureidkey="properties.LSOA_code", # CRITICAL: The exact path inside your GeoJSON file
    color="hotspot_probability",         # What dictates the color of the neighborhoods
    color_continuous_scale="Reds",       # Red scale for hotspots
    range_color=[0, 1],                  # Probability goes from 0.0 to 1.0
    mapbox_style="carto-positron",       # Clean, readable base map background
    zoom=9.5,                            # Zoom level focused right on London
    center={"lat": 51.5074, "lon": -0.1278}, # Lat/Long coordinates for central London
    opacity=0.6,                         # Let the streets show through the color blocks
    labels={"hotspot_probability": "Hotspot Risk"}
)

fig.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=600
)

# Display the map in the Streamlit app
st.plotly_chart(fig, use_container_width=True)
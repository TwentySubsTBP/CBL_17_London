import pandas as pd
from pathlib import Path
import geopandas as gpd
import plotly.express as px
import json

data_directory = Path("main\data\stop_and_search_23_26_10_forces")

target_forces = {
     "city-of-london",
    "metropolitan",
    "west-midlands",
    "merseyside",
    "west-yorkshire",
    "south-yorkshire",
    "northumbria",
    "leicestershire",
    "nottinghamshire",
    "humberside",
}

data_frames = []

for map in data_directory.iterdir():
    for file in map.iterdir():
        force = file.stem[8:].replace("-stop-and-search", "")
        print(force)
        if force not in target_forces:
            continue
        print(f'Loading: {file.name}')
        df = pd.read_csv(file)
        df["Force"] = force
        data_frames.append(df)
if not data_frames:
    raise ValueError("No stop and search files found.")

stop_and_search_df = pd.concat(data_frames, ignore_index = True)
print(f"Rows: {len(stop_and_search_df):,}")
print(f"Columns: {len(stop_and_search_df.columns)}")

df = stop_and_search_df[[
    "Date",
    "Latitude",
    "Longitude",
    "Force"
]]

print(df.shape)

#################################
before = len(df)
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df = df.dropna(subset=["Longitude", "Latitude"]).copy()
after = len(df)
print("Deleted:", before - after)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
    crs="EPSG:4326"
)
print(gdf.head())

# from https://www.data.gov.uk/dataset/6785cb41-914a-444f-9a54-7981171132a3/lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-v4-and-rural-urban-classification
lsoa_boundaries = gpd.read_file(r"main\src\LSOA_(2021)_EW_BSC_V4_to_Rural_Urban_Classification.geojson")
print(lsoa_boundaries.crs)
print(gdf.crs)

gdf = gpd.sjoin(gdf, lsoa_boundaries, how="left", predicate="within")
gdf = gdf[[
    'Date',
    'Force',
    'LSOA21CD',
    'LSOA21NM',
    'LSOA21NMW',
]]
print(gdf.head())
print(gdf.columns)

counts = gdf.groupby("LSOA21CD").size().reset_index(name="n_crimes")
print(counts.head(), counts.shape)
lsoa_active = lsoa_boundaries.merge(counts, on="LSOA21CD", how="inner")
geojson = json.loads(lsoa_active.to_json())

fig = px.choropleth_mapbox(
    lsoa_active,
    geojson=geojson,
    locations="LSOA21CD",
    featureidkey="properties.LSOA21CD",
    color="n_crimes",
    color_continuous_scale="OrRd",
    mapbox_style="carto-positron",
    zoom=9,
    center={"lat": 51.5, "lon": -0.1},
    opacity=0.6
)

fig.show()
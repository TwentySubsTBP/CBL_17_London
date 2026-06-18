import pandas as pd
from pathlib import Path
import geopandas as gpd
import plotly.express as px
import json
import numpy as np
import os

# Stop and search represents police activity
# Source: https://www.gov.uk/government/publications/police-powers-and-procedures-in-england-and-wales-201112-user-guide/user-guide-to-police-powers-and-procedures#stop-and-search-1

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
        #print(force)
        if force not in target_forces:
            continue
        #print(f'Loading: {file.name}')
        df = pd.read_csv(file)
        df["Force"] = force
        data_frames.append(df)
if not data_frames:
    raise ValueError("No stop and search files found.")

stop_and_search_df = pd.concat(data_frames, ignore_index = True)
# print(f"Rows: {len(stop_and_search_df):,}")
# print(f"Columns: {len(stop_and_search_df.columns)}")

#################################
before = len(stop_and_search_df)
stop_and_search_df["Longitude"] = pd.to_numeric(stop_and_search_df["Longitude"], errors="coerce")
stop_and_search_df["Latitude"] = pd.to_numeric(stop_and_search_df["Latitude"], errors="coerce")
stop_and_search_df = stop_and_search_df.dropna(subset=["Longitude", "Latitude"]).copy()
after = len(stop_and_search_df)
print("Deleted:", round(((before - after)/before)*100, 1), "%", '\n')

gdf = gpd.GeoDataFrame(
    stop_and_search_df,
    geometry=gpd.points_from_xy(stop_and_search_df["Longitude"], stop_and_search_df["Latitude"]),
    crs="EPSG:4326"
)

# from https://www.data.gov.uk/dataset/6785cb41-914a-444f-9a54-7981171132a3/lower-layer-super-output-areas-december-2021-boundaries-ew-bsc-v4-and-rural-urban-classification
lsoa_boundaries = gpd.read_file(r"main\src\LSOA_(2021)_EW_BSC_V4_to_Rural_Urban_Classification.geojson")

gdf = gpd.sjoin(gdf, lsoa_boundaries, how="left", predicate="within")
gdf = gdf[[
    'Date',
    'Force',
    'LSOA21CD',
    'LSOA21NM',
    'LSOA21NMW',
    'Outcome linked to object of search',
]]

print("Joined gdf for exposure estimation:")
gdf["Month"] = pd.to_datetime(gdf["Date"]).dt.to_period("M")
# REMOVE OUT-OF-SAMPLE MONTHS
gdf = gdf[gdf["Month"] >= pd.Period("2023-05", freq="M")].copy()
print(gdf.head(5), '\n')

##################################
# STEP 1: LSOA-month counts (BASIS + force info behouden)

estimated_exposure_df = (
    gdf.groupby(["LSOA21CD", "Month"])
    .size()
    .reset_index(name="n_police_activity")
)

# Forces per LSOA-month (lijst van unieke forces)
lsoa_force_lookup = (
    gdf.groupby(["LSOA21CD", "Month"])["Force"]
    .agg(lambda x: list(pd.unique(x)))
    .reset_index(name="Forces_in_LSOA_month")
)

# Merge forces in
estimated_exposure_df = estimated_exposure_df.merge(
    lsoa_force_lookup,
    on=["LSOA21CD", "Month"],
    how="left"
)

##################################
# Normalize exposure for LSOA area

lsoa = gpd.read_file(r"main\src\LSOA_(2021)_EW_BSC_V4_to_Rural_Urban_Classification.geojson")
lsoa_proj = lsoa.to_crs(27700)

# Compute area in square meters and square kilometers
lsoa_proj["area_m2"] = lsoa_proj.geometry.area
lsoa_proj["area_km2"] = lsoa_proj["area_m2"] / 1e6
lsoa_area = lsoa_proj[["LSOA21CD", "area_m2", "area_km2"]].copy()

estimated_exposure_df = estimated_exposure_df.merge(
    lsoa_area,
    on="LSOA21CD",
    how="left"
)

# Police activity density within each LSOA
estimated_exposure_df["activity_density"] = (
    estimated_exposure_df["n_police_activity"] /
    estimated_exposure_df["area_km2"]
)

# Monthly total density
monthly_density_total = (
    estimated_exposure_df
    .groupby("Month")["activity_density"]
    .sum()
    .reset_index(name="monthly_density_total")
)

estimated_exposure_df = estimated_exposure_df.merge(
    monthly_density_total,
    on="Month",
    how="left"
)

# Area-normalized exposure
estimated_exposure_df["estimated_exposure_lsoa_month"] = (
    estimated_exposure_df["activity_density"] /
    estimated_exposure_df["monthly_density_total"]
)

estimated_exposure_df = estimated_exposure_df.drop(columns=[
    "area_km2",
    "area_m2",
    "activity_density",
    "monthly_density_total"
], errors="ignore")

##################################

alpha = 2  # Strength of the downweighting

estimated_exposure_df["downweight"] = np.exp(
    -alpha * np.log1p(estimated_exposure_df["estimated_exposure_lsoa_month"])
)

print("Estimated exposure dataframe, with area-normalisation and downweigths:")
print(estimated_exposure_df.head(50), "\n")

# Top 10 highest normalized exposure
top_10 = estimated_exposure_df.sort_values(
    "estimated_exposure_lsoa_month",
    ascending=False
).head(10)

print("Top 10 highest area-normalized exposure:")
print(top_10, '\n')

# 5 lowest normalized exposure
bottom_5 = estimated_exposure_df.sort_values(
    "estimated_exposure_lsoa_month",
    ascending=True
).head(5)

print("Bottom 5 lowest area-normalized exposure:")
print(bottom_5, '\n')

##############################

print(estimated_exposure_df["downweight"].describe())

##############################

# # CSV: all columns
# estimated_exposure_df.to_csv(
#     "estimated_exposure_lsoa_month_full.csv",
#     index=False
# )

# # Parquet: Only usefull columns
# export_df = estimated_exposure_df[[
#     "LSOA21CD",
#     "Month",
#     "estimated_exposure_lsoa_month",
#     "downweight"
# ]]

# export_df.to_parquet(
#     "estimated_exposure_lsoa_month_clean.parquet",
#     index=False
# )

# print("CSV saved to:", os.path.abspath("estimated_exposure_lsoa_month_full.csv"))
# print("Parquet saved to:", os.path.abspath("estimated_exposure_lsoa_month_clean.parquet"))

##################################
# Visualize stop and searches per LSOA (no temporal patterns visible)

counts = gdf.groupby("LSOA21CD").size().reset_index(name="n_patroulle")
lsoa_active = lsoa_boundaries.merge(counts, on="LSOA21CD", how="inner")
geojson = json.loads(lsoa_active.to_json())
fig = px.choropleth_mapbox(
    lsoa_active,
    geojson=geojson,
    locations="LSOA21CD",
    featureidkey="properties.LSOA21CD",
    color="n_patroulle",
    color_continuous_scale="OrRd",
    mapbox_style="carto-positron",
    zoom=9,
    center={"lat": 51.5, "lon": -0.1},
    opacity=0.6
)
fig.show()

import pandas as pd
from pathlib import Path
import geopandas as gpd
import plotly.express as px
import json

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

########################################
# Estimate exposure
police_activity_lsoa_month = gdf.groupby(['LSOA21CD', 'Force', 'Month']).size().reset_index(name = "n_police_activity")
# print("Police activity per LSOA per Month:")
# print(police_activity_lsoa_month.head(), '\n')

duplicate_forces = {}
for _, row in police_activity_lsoa_month.iterrows():
    lsoa = row["LSOA21CD"]
    force = row["Force"]
    if lsoa not in duplicate_forces:
        duplicate_forces[lsoa] = []
        duplicate_forces[lsoa].append(force)
    else:
        if force not in duplicate_forces[lsoa]:
            duplicate_forces[lsoa].append(force)

count_duplicates = 0
# print("LSOA's with multiple forces:")
for lsoa, forces in duplicate_forces.items():
    if len(forces) > 1:
        count_duplicates += 1
        # print(lsoa, forces, '\n')

multiple_forces_percentage = round((count_duplicates / len(duplicate_forces))* 100, 1)
print("Percentage of LSOA's with multiple forces", multiple_forces_percentage, '%', '\n')

monthly_total_per_force = (
    police_activity_lsoa_month
    .groupby(["Force", "Month"])["n_police_activity"]
    .sum()
    .reset_index(name="n_police_activity_monthly_total")
)
# print("Monthly total of police activity, per force:")
# print(monthly_total_per_force.head(), '\n')

estimated_exposure_df = police_activity_lsoa_month.merge(
    monthly_total_per_force,
    on=["Force", "Month"],
    how="left"
)

estimated_exposure_df["estimated_exposure_lsoa_month_force"] = (
    estimated_exposure_df["n_police_activity"] /
    estimated_exposure_df["n_police_activity_monthly_total"]
)

print("Estimated exposure of each LSOA per month relative to the total police activity in the force that month:")
print(estimated_exposure_df.head(), '\n')

# Top 20 highest exposure
top_20 = estimated_exposure_df.sort_values(
    "estimated_exposure_lsoa_month_force",
    ascending=False
).head(20)

print("Top 20 highest exposure:")
print(top_20, '\n')

# 5 lowest exposure
bottom_5 = estimated_exposure_df.sort_values(
    "estimated_exposure_lsoa_month_force",
    ascending=True
).head(5)

print("Bottom 5 lowest exposure:")
print(bottom_5, '\n')

total_per_force = (
    gdf
    .groupby("Force")
    .size()
    .reset_index(name="n_police_activity_total")
)
print("Total police activity per force:")
print(total_per_force, '\n')

#######################################

counts = gdf.groupby("LSOA21CD").size().reset_index(name="n_patroulle")

# print("Count dataframe (police activity count per LSOA):")
# print(counts.head(), counts.shape, counts["n_patroulle"].max())

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

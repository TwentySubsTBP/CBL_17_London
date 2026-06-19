import os
import glob
import pandas as pd
import pyarrow
import json

# =====================================================================
# 1. SETUP CONFIGURATION & GEOJSON WHITELIST
# =====================================================================
# Find the exact absolute folder directory where source_lsoa_crime_data.py is currently saved
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. DATA_ROOT: Back out to 'data' directory, then step down into 'raw/extracted'
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../data", "raw", "extracted"))

# 2. GEOJSON_PATH: Point directly to the file sitting right next to this script
GEOJSON_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/for training/major_cities_2021_skeleton.geojson"))

TARGET_FORCES = [
    "city-of-london", "metropolitan", "west-midlands", "merseyside",
    "west-yorkshire", "south-yorkshire", "northumbria", "leicestershire",
    "nottinghamshire", "humberside"
]

TARGET_CRIMES = [
    "Violence and sexual offences",
    "Burglary",
    "Robbery",
    "Vehicle crime",
    "Shoplifting"
]

# Load the exact boundaries used by your dashboard to serve as the master whitelist
print(f"Loading official city boundaries from {GEOJSON_PATH}...")
if not os.path.exists(GEOJSON_PATH):
    raise FileNotFoundError(f"Missing critical skeleton map file at: {GEOJSON_PATH}. Run your map generation script first.")

with open(GEOJSON_PATH, "r") as f:
    geojson_data = json.load(f)

# Extract a set of valid 2021 LSOA codes for high-speed O(1) filtering
valid_lsoa_whitelist = set(
    feature["properties"]["LSOA21CD"]
    for feature in geojson_data["features"]
    if feature["properties"].get("LSOA21CD")
)
print(f"Successfully loaded reference whitelist with {len(valid_lsoa_whitelist)} valid city LSOAs.")

raw_data_chunks = []

# =====================================================================
# 2. FILE CRAWLING & FILTERING
# =====================================================================
print("Scanning folders and filtering target crimes by LSOA code...")

for month_folder in sorted(os.listdir(DATA_ROOT)):
    folder_path = os.path.join(DATA_ROOT, month_folder)

    if os.path.isdir(folder_path):
        for force in TARGET_FORCES:
            file_name = f"{month_folder}-{force}-street.csv"
            file_path = os.path.join(folder_path, file_name)

            if os.path.exists(file_path):
                try:
                    df_chunk = pd.read_csv(file_path, usecols=["Month", "LSOA code", "Crime type"])
                    df_chunk = df_chunk.dropna(subset=["LSOA code"])
                    df_chunk["LSOA code"] = df_chunk["LSOA code"].str.strip()

                    # Keep ONLY rows matching your 5 target crimes
                    df_chunk = df_chunk[df_chunk["Crime type"].isin(TARGET_CRIMES)]

                    raw_data_chunks.append(df_chunk)
                except Exception as e:
                    print(f"Error reading file {file_name}: {e}")

df_raw_master = pd.concat(raw_data_chunks, axis=0, ignore_index=True)
print(f"Total raw rows gathered from police files: {df_raw_master.shape[0]}")

# ADDED: Drop any LSOA code not explicitly inside your target city boundaries
print("Enforcing urban boundary constraints against GeoJSON whitelist...")
df_raw_master = df_raw_master[df_raw_master["LSOA code"].isin(valid_lsoa_whitelist)]
print(f"Cleaned raw rows remaining within city scopes: {df_raw_master.shape[0]}")

# =====================================================================
# 3. TOTAL SUBSET AGGREGATION
# =====================================================================
print("Summing target crimes into a single local total per LSOA code...")
df_time_series = df_raw_master.groupby(["LSOA code", "Month"]).size().reset_index(name="crime_count")

# =====================================================================
# 4. ZERO-CRIME GRID FIX (Enforced via Whitelist Mappings)
# =====================================================================
print("Squaring the timeline grid...")
# CHANGED: We use the complete whitelist array instead of unique data values.
# This guarantees that the matrix includes all expected neighborhoods, even if they had 0 crimes.
all_lsoas = sorted(list(valid_lsoa_whitelist))
all_months = df_time_series["Month"].unique()

full_grid_index = pd.MultiIndex.from_product([all_lsoas, all_months], names=["LSOA code", "Month"])
df_time_series = df_time_series.set_index(["LSOA code", "Month"]).reindex(full_grid_index, fill_value=0).reset_index()

# =====================================================================
# 5. SPATIAL NEIGHBOR AGGREGATION (Code Mapping)
# =====================================================================
print("Calculating neighbor crime totals using spatial codes...")
df_neighbors = pd.read_parquet("../data/for training/lsoa_neighbors.parquet")

df_neighbors.columns = ["lsoa", "neighbor_lsoa"]

df_lookup = df_time_series[["LSOA code", "Month", "crime_count"]].copy()

df_spatial_join = df_neighbors.merge(
    df_lookup,
    left_on="neighbor_lsoa",
    right_on="LSOA code",
    how="inner"
)

df_spatial_totals = (df_spatial_join.groupby(["lsoa", "Month"])["crime_count"]
                     .sum()
                     .reset_index(name="neighbor_crime_count"))

df_time_series = df_time_series.merge(
    df_spatial_totals,
    left_on=["LSOA code", "Month"],
    right_on=["lsoa", "Month"],
    how="left"
).drop(columns=["lsoa"])

df_time_series["neighbor_crime_count"] = df_time_series["neighbor_crime_count"].fillna(0)

df_time_series["Month"] = pd.to_datetime(df_time_series["Month"])
df_time_series = df_time_series.sort_values(["LSOA code", "Month"]).reset_index(drop=True)

# =====================================================================
# 6. CNN FEATURE GENERATION
# =====================================================================
print("Generating lag features and moving averages...")
grouped_local = df_time_series.groupby("LSOA code")["crime_count"]
grouped_spatial = df_time_series.groupby("LSOA code")["neighbor_crime_count"]

df_time_series["crime_1m_ago"] = grouped_local.shift(1)
df_time_series["crime_3m_ago"] = grouped_local.shift(3)
df_time_series["crime_6m_ago"] = grouped_local.shift(6)
df_time_series["yearly_avg"] = grouped_local.transform(lambda x: x.shift(1).rolling(window=12).mean())

df_time_series["neighbor_1m_ago"] = grouped_spatial.shift(1)
df_time_series["neighbor_3m_ago"] = grouped_spatial.shift(3)
df_time_series["neighbor_6m_ago"] = grouped_spatial.shift(6)

# =====================================================================
# 7. FINAL FILTERING & SAVE
# =====================================================================
df_final = df_time_series.fillna(0).reset_index(drop=True)
print(df_final)
print(f"\n--- SUCCESS ---")
print(f"Final shape: {df_final.shape[0]} rows x {df_final.shape[1]} columns")


df_final.to_csv("../data/for training/crime_data.csv", index=False)
df_final.to_parquet("../data/for training/crime_data.parquet", index=False)

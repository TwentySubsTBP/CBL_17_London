import os
import glob
import pandas as pd
import pyarrow

# =====================================================================
# 1. SETUP CONFIGURATION
# =====================================================================
DATA_ROOT = "./data/raw/extracted"

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

raw_data_chunks = []

# =====================================================================
# 2. FILE CRAWLING & FILTERING (Switching to LSOA code)
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
                    # CHANGED: Loading 'LSOA code' instead of 'LSOA name'
                    df_chunk = pd.read_csv(file_path, usecols=["Month", "LSOA code", "Crime type"])
                    df_chunk = df_chunk.dropna(subset=["LSOA code"])

                    # Keep ONLY rows matching your 5 crimes
                    df_chunk = df_chunk[df_chunk["Crime type"].isin(TARGET_CRIMES)]

                    raw_data_chunks.append(df_chunk)
                except Exception as e:
                    print(f"Error reading file {file_name}: {e}")

df_raw_master = pd.concat(raw_data_chunks, axis=0, ignore_index=True)

# =====================================================================
# 3. TOTAL SUBSET AGGREGATION
# =====================================================================
print("Summing target crimes into a single local total per LSOA code...")
# CHANGED: Grouping strictly by LSOA code and Month
df_time_series = df_raw_master.groupby(["LSOA code", "Month"]).size().reset_index(name="crime_count")

# =====================================================================
# 4. ZERO-CRIME GRID FIX (LSOA code x Month)
# =====================================================================
print("Squaring the timeline grid...")
all_lsoas = df_time_series["LSOA code"].unique()
all_months = df_time_series["Month"].unique()

# CHANGED: Index built using codes
full_grid_index = pd.MultiIndex.from_product([all_lsoas, all_months], names=["LSOA code", "Month"])
df_time_series = df_time_series.set_index(["LSOA code", "Month"]).reindex(full_grid_index, fill_value=0).reset_index()
# =====================================================================
# 5. SPATIAL NEIGHBOR AGGREGATION (Code Mapping)
# =====================================================================
print("Calculating neighbor crime totals using spatial codes...")
df_neighbors = pd.read_parquet("lsoa_neighbors.parquet")
df_neighbors.columns = ["lsoa", "neighbor_lsoa"]

# Isolate local crime snapshot for neighbor lookup
df_lookup = df_time_series[["LSOA code", "Month", "crime_count"]].copy()

# Map neighbors to their monthly crime totals via their LSOA codes
df_spatial_join = df_neighbors.merge(
    df_lookup,
    left_on="neighbor_lsoa",
    right_on="LSOA code",
    how="inner"
)

# Sum all neighbor crimes up for each primary LSOA code
df_spatial_totals = (df_spatial_join.groupby(["lsoa", "Month"])["crime_count"]
                     .sum()
                     .reset_index(name="neighbor_crime_count"))
print(df_spatial_totals)
# Merge neighbor totals back into the master timeline dataframe
df_time_series = df_time_series.merge(
    df_spatial_totals,
    left_on=["LSOA code", "Month"],
    right_on=["lsoa", "Month"],
    how="left"
).drop(columns=["lsoa"])

df_time_series["neighbor_crime_count"] = df_time_series["neighbor_crime_count"].fillna(0)

# Sort everything chronologically by neighborhood block
df_time_series["Month"] = pd.to_datetime(df_time_series["Month"])
df_time_series = df_time_series.sort_values(["LSOA code", "Month"]).reset_index(drop=True)

# =====================================================================
# 6. CNN FEATURE GENERATION
# =====================================================================
print("Generating lag features and moving averages...")
grouped_local = df_time_series.groupby("LSOA code")["crime_count"]
grouped_spatial = df_time_series.groupby("LSOA code")["neighbor_crime_count"]

# Local features

df_time_series["crime_1m_ago"] = grouped_local.shift(1)
df_time_series["crime_3m_ago"] = grouped_local.shift(3)
df_time_series["crime_6m_ago"] = grouped_local.shift(6)
df_time_series["yearly_avg"] = grouped_local.transform(lambda x: x.shift(1).rolling(window=12).mean())


# =====================================================================
# 7. FINAL FILTERING & SAVE
# =====================================================================
df_final = df_time_series.fillna(0).reset_index(drop=True)
print(df_final)
print(f"\n--- SUCCESS ---")
print(f"Final shape: {df_final.shape[0]} rows x {df_final.shape[1]} columns")


#df_final.to_csv("cnn_code_based_crime_pipeline.csv", index=False)
df_final.to_parquet("crime_data.parquet")
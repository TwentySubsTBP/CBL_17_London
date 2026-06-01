from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "extracted"

CRIME_FEATURES_PATH = PROJECT_ROOT / "scripts" / "crime_data.parquet"

OUTPUT_PATH = PROJECT_ROOT / "scripts" / "crime_data_with_force_stop_search.parquet"



TARGET_FORCES = {
    "city-of-london": "City of London Police",
    "metropolitan": "Metropolitan Police Service",
    "west-midlands": "West Midlands Police",
    "merseyside": "Merseyside Police",
    "west-yorkshire": "West Yorkshire Police",
    "south-yorkshire": "South Yorkshire Police",
    "northumbria": "Northumbria Police",
    "leicestershire": "Leicestershire Police",
    "nottinghamshire": "Nottinghamshire Police",
    "humberside": "Humberside Police",
}
def parse_police_filename(path: Path):
    stem = path.stem

    month = stem[:7]
    rest = stem[8:]

    if rest.endswith("-street"):
        force_slug = rest.replace("-street", "")
        file_type = "street"

    elif rest.endswith("-stop-and-search"):
        force_slug = rest.replace("-stop-and-search", "")
        file_type = "stop-and-search"

    elif rest.endswith("-outcomes"):
        force_slug = rest.replace("-outcomes", "")
        file_type = "outcomes"

    else:
        force_slug = None
        file_type = None

    return month, force_slug, file_type


# 1. Load base LSOA-month crime features

print("Loading crime feature parquet...")

crime_df = pd.read_parquet(CRIME_FEATURES_PATH)

# Standardise column names carefully
if "LSOA code" not in crime_df.columns:
    raise ValueError("Expected column 'LSOA code' in crime feature parquet.")

if "Month" not in crime_df.columns:
    raise ValueError("Expected column 'Month' in crime feature parquet.")

crime_df["Month"] = pd.to_datetime(crime_df["Month"]).dt.to_period("M").astype(str)

print(f"Crime feature rows: {len(crime_df):,}")
print(f"Crime feature columns: {list(crime_df.columns)}")


# 2. Build LSOA → police force mapping from street crime files

print("\nBuilding LSOA to police force mapping from street crime files...")

mapping_parts = []

street_files = list(RAW_DATA_DIR.rglob("*-street.csv"))

for file in street_files:
    month, force_slug, file_type = parse_police_filename(file)

    if file_type != "street":
        continue

    if force_slug not in TARGET_FORCES:
        continue

    try:
        usecols = ["LSOA code"]
        temp = pd.read_csv(file, usecols=usecols)
    except Exception as e:
        print(f"Skipping {file.name}: {e}")
        continue

    temp = temp.dropna(subset=["LSOA code"])
    temp = temp.drop_duplicates()
    temp["force_slug"] = force_slug
    temp["police_force_name"] = TARGET_FORCES[force_slug]

    mapping_parts.append(temp)

if not mapping_parts:
    raise ValueError("No street crime files found for selected forces.")

lsoa_force_map = pd.concat(mapping_parts, ignore_index=True)

# If an LSOA appears multiple times, keep the most common force assignment
lsoa_force_map = (
    lsoa_force_map
    .groupby(["LSOA code", "force_slug", "police_force_name"])
    .size()
    .reset_index(name="n")
    .sort_values(["LSOA code", "n"], ascending=[True, False])
    .drop_duplicates("LSOA code")
    [["LSOA code", "force_slug", "police_force_name"]]
)

print(f"Unique LSOA-force mappings: {len(lsoa_force_map):,}")



# 3. Merge police force into crime features

crime_df = crime_df.merge(
    lsoa_force_map,
    on="LSOA code",
    how="left"
)

missing_force = crime_df["force_slug"].isna().sum()
print(f"Rows missing police force after merge: {missing_force:,}")



# 4. Aggregate stop-and-search by police force and month

print("\nAggregating stop-and-search files by force and month...")

stop_parts = []

stop_files = list(RAW_DATA_DIR.rglob("*-stop-and-search.csv"))

for file in stop_files:
    month, force_slug, file_type = parse_police_filename(file)

    if file_type != "stop-and-search":
        continue

    if force_slug not in TARGET_FORCES:
        continue

    try:
        temp = pd.read_csv(file)
    except Exception as e:
        print(f"Skipping {file.name}: {e}")
        continue

    temp["Month"] = month
    temp["force_slug"] = force_slug
    temp.columns = [c.strip() for c in temp.columns]

    stop_parts.append(temp)

if not stop_parts:
    print("Warning: No stop-and-search files found. Stop-search columns will be zero.")
    stop_monthly = pd.DataFrame(columns=["Month", "force_slug"])
else:
    stop_df = pd.concat(stop_parts, ignore_index=True)

    print(f"Stop-search rows loaded: {len(stop_df):,}")

    stop_monthly = (
        stop_df
        .groupby(["force_slug", "Month"])
        .size()
        .reset_index(name="stop_search_count")
    )
    if "Outcome" in stop_df.columns:
        outcome_pivot = (
            stop_df
            .assign(Outcome=stop_df["Outcome"].fillna("Unknown"))
            .groupby(["force_slug", "Month", "Outcome"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        outcome_pivot.columns = [
            "ss_outcome_" + str(c).lower().replace(" ", "_").replace("-", "_")
            if c not in ["force_slug", "Month"]
            else c
            for c in outcome_pivot.columns
        ]

        stop_monthly = stop_monthly.merge(
            outcome_pivot,
            on=["force_slug", "Month"],
            how="left"
        )
    if "Object of search" in stop_df.columns:
        object_pivot = (
            stop_df
            .assign(**{"Object of search": stop_df["Object of search"].fillna("Unknown")})
            .groupby(["force_slug", "Month", "Object of search"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )

        object_pivot.columns = [
            "ss_object_" + str(c).lower().replace(" ", "_").replace("-", "_")
            if c not in ["force_slug", "Month"]
            else c
            for c in object_pivot.columns
        ]

        stop_monthly = stop_monthly.merge(
            object_pivot,
            on=["force_slug", "Month"],
            how="left"
        )

print(f"Stop-search monthly rows: {len(stop_monthly):,}")


# 5. Merge stop-search force-month features into crime features
print("\nMerging stop-search features into crime features...")

merged_df = crime_df.merge(
    stop_monthly,
    on=["force_slug", "Month"],
    how="left"
)

# Fill missing stop-search values with 0
stop_cols = [c for c in merged_df.columns if c.startswith("stop_search") or c.startswith("ss_")]
merged_df[stop_cols] = merged_df[stop_cols].fillna(0)

# Add safer stop-search intensity features
if "crime_count" in merged_df.columns and "stop_search_count" in merged_df.columns:
    merged_df["stop_search_per_crime"] = (
        merged_df["stop_search_count"] / (merged_df["crime_count"] + 1)
    )


print("\nSaving enriched parquet...")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

merged_df.to_parquet(OUTPUT_PATH, index=False)

print(f"Saved to: {OUTPUT_PATH}")
print(f"Final rows: {len(merged_df):,}")
print(f"Final columns: {len(merged_df.columns):,}")

print("\nColumns:")
for col in merged_df.columns:
    print(f"- {col}")
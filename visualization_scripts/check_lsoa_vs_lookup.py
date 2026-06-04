"""
check_lsoa_vs_lookup.py
=======================
 
Take the unique LSOA codes from the training parquet and classify them
against the ONS 2011→2021 lookup table.
 
Each LSOA code in the parquet should fall into one of three buckets:
  1. Found as LSOA21CD only       — clean 2021 code (the normal case).
  2. Found as LSOA11CD only       — stale 2011 code; needs remapping.
  3. Found in both columns        — code unchanged in 2021 (CHGIND='U').
  X. Not found in either column   — unrecognised; investigate manually.
"""
 
from pathlib import Path
import pandas as pd
 
# ---- Config: adjust to your filenames -------------------------------------
PARQUET_PATH = Path(r"resources/crime_lsoa_month_filtered.parquet")
LOOKUP_PATH  = Path(r"visualization_scripts\LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv")
print(f"Loading parquet: {PARQUET_PATH}")
df = pd.read_parquet(PARQUET_PATH)
print(f"  Columns: {df.columns.tolist()}")

# Auto-detect the LSOA code column
candidates = [c for c in df.columns if "lsoa" in c.lower() and "code" in c.lower()]
if not candidates:
    raise KeyError(
        f"No LSOA code column found. Columns: {df.columns.tolist()}"
    )
LSOA_COL = candidates[0]
print(f"  Using column: '{LSOA_COL}'")

parquet_codes = set(df[LSOA_COL].dropna().astype(str).unique())
print(f"  Unique LSOA codes: {len(parquet_codes):,}\n")
# ---------------------------------------------------------------------------
 
# ---- Load parquet codes ---------------------------------------------------
print(f"Loading parquet: {PARQUET_PATH}")
df = pd.read_parquet(PARQUET_PATH)
parquet_codes = set(df[LSOA_COL].dropna().astype(str).unique())
print(f"  Unique LSOA codes: {len(parquet_codes):,}\n")
 
# ---- Load lookup ----------------------------------------------------------
print(f"Loading lookup: {LOOKUP_PATH}")
lookup = pd.read_csv(LOOKUP_PATH, usecols=["LSOA11CD", "LSOA21CD", "CHGIND"])
 
codes_11 = set(lookup["LSOA11CD"].dropna().astype(str).unique())
codes_21 = set(lookup["LSOA21CD"].dropna().astype(str).unique())
print(f"  Unique LSOA11CD values: {len(codes_11):,}")
print(f"  Unique LSOA21CD values: {len(codes_21):,}\n")
 
# ---- Classify each parquet code ------------------------------------------
in_2021_only = parquet_codes & codes_21 - codes_11    # clean 2021 code, changed in 2021
in_2011_only = parquet_codes & codes_11 - codes_21    # stale 2011 code, needs remap
in_both      = parquet_codes & codes_21 & codes_11    # unchanged (CHGIND='U')
in_neither   = parquet_codes - codes_21 - codes_11    # unrecognised
 
print("=" * 70)
print("CLASSIFICATION OF PARQUET LSOA CODES")
print("=" * 70)
print(f"  In 2021 only (clean, changed in 2021):   {len(in_2021_only):>7,}")
print(f"  In both columns (unchanged, CHGIND='U'): {len(in_both):>7,}")
print(f"  In 2011 only (STALE — needs remapping):  {len(in_2011_only):>7,}")
print(f"  In neither (UNRECOGNISED):               {len(in_neither):>7,}")
print(f"                                            {'─' * 7}")
print(f"  Total:                                   {len(parquet_codes):>7,}")
print("=" * 70)
 
# ---- For the stale codes, show the CHGIND breakdown ----------------------
if in_2011_only:
    stale = lookup[lookup["LSOA11CD"].isin(in_2011_only)]
    print("\nCHGIND breakdown for stale 2011 codes:")
    print(stale.drop_duplicates("LSOA11CD")["CHGIND"].value_counts().to_string())
    print("  (S = split, M = merged, X = irregular)")
 
if in_neither:
    print(f"\nSample unrecognised codes: {sorted(in_neither)[:10]}")
    print("  These exist in neither LSOA11CD nor LSOA21CD — check for typos,")
    print("  Welsh codes (W01...), or non-LSOA values in the source data.")
 

 

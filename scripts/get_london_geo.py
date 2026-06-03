import os
import json
import requests
import pandas as pd
import geopandas as gpd

# 1. Setup project paths
# 1. Setup project paths
current_dir = os.path.dirname(os.path.abspath(__file__)) # Points to CBL_17_London/scripts
project_root = os.path.dirname(current_dir)              # Points to CBL_17_London

# FIX HERE: Use current_dir because the CSV is right next to this script!
csv_path = os.path.join(current_dir, "london_predictions.csv")
output_path = os.path.join(current_dir, "london_lsoa_simplified.json")
if not os.path.exists(csv_path):
    print(f"❌ Could not find {csv_path}!")
    print("Please run your smart dummy generator script first to create the CSV file.")
    exit()

# 2. Read your actual London LSOA codes from your dummy data
print("📖 Reading your local LSOA codes...")
df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

# Automatically find whichever column holds your LSOA codes
code_col = [c for c in df.columns if 'LSOA' in c or 'code' in c.lower()][0]
lsoa_codes = df[code_col].dropna().unique().tolist()
print(f"📍 Found {len(lsoa_codes)} unique LSOA codes in your dataset.")

# 3. Define the live Government ONS API endpoints (Trying 2011 boundaries first)
ons_url = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/LSOA_2011_Boundaries_Super_Generalised_Clipped_BSC_EW_V4/FeatureServer/0/query"
code_field = "LSOA11CD"

# Split your codes into batches of 1,000 so the API doesn't choke
chunks = [lsoa_codes[i:i + 1000] for i in range(0, len(lsoa_codes), 1000)]
gdfs = []

print("🌐 Querying the live UK Government ONS Portal via API chunks...")
for idx, chunk in enumerate(chunks):
    # Format the codes for an SQL 'IN' statement: ('E01000001', 'E01000002', ...)
    formatted_codes = ",".join([f"'{c}'" for c in chunk])

    # Payload for the government database
    payload = {
        'where': f"{code_field} IN ({formatted_codes})",
        'outFields': code_field,
        'outSR': '4326',  # Directly request GPS coordinates (Web-ready)
        'f': 'geojson'  # Directly request GeoJSON format
    }

    try:
        # Use POST to safely handle large list of codes without breaking URL length limits
        response = requests.post(ons_url, data=payload)
        if response.status_code == 200:
            gdf_chunk = gpd.read_file(response.text, driver="GeoJSON")
            if not gdf_chunk.empty:
                gdfs.append(gdf_chunk)
                print(f"  ✅ Received batch {idx + 1}/{len(chunks)} ({len(gdf_chunk)} shapes)")
        else:
            print(f"  ❌ Batch {idx + 1} failed with status code: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Error fetching batch {idx + 1}: {e}")

# 4. Merge, clean, and standardize
if gdfs:
    print("Merge layers and finalizing format...")
    gdf_final = pd.concat(gdfs, ignore_index=True)
    gdf_final = gpd.GeoDataFrame(gdf_final, geometry='geometry')

    # Ensure standard naming conventions for your Streamlit dashboard
    gdf_final = gdf_final.rename(columns={code_field: 'LSOA_code'})

    # Save the file
    gdf_final.to_file(output_path, driver="GeoJSON")
    print(f"\n✅ SUCCESS! Saved {len(gdf_final)} matches to: {output_path}")
    print("🚀 You are fully clear to run your dashboard now!")
else:
    print("\n❌ Failed to fetch boundaries. Double-check your internet connection.")
import os
import geopandas as gpd

# 1. Define the exact primary city/hub strings as they appear in ONS 2021 LSOA names
# This matches the "Primary City / Hub" column from your technical guide sheet.
TARGET_CITIES = [
    "Manchester", "Birmingham", "Liverpool", "Leeds", "Sheffield",
    "Newcastle upon Tyne", "Leicester", "Nottingham", "Kingston upon Hull"
]

# For London, we include the City of London plus all 32 Greater London boroughs
LONDON_BOROUGHS = [
    "City of London", "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley",
    "Camden", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney", "Hammersmith and Fulham",
    "Haringey", "Harrow", "Havering", "Hillingdon", "Hounslow", "Islington",
    "Kensington and Chelsea", "Kingston upon Thames", "Lambeth", "Lewisham", "Merton",
    "Newham", "Redbridge", "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
    "Waltham Forest", "Wandsworth", "Westminster"
]

# Combine into a single master list of allowed urban areas
allowed_urban_hubs = TARGET_CITIES + LONDON_BOROUGHS

print("Streaming 2021 LSOA Boundaries directly from ONS GeoPortal...")
# Official ONS Dec 2021 Super Generalised boundaries URL
ons_geojson_url = "https://opendata.arcgis.com/datasets/b976e08d5c894df3901963469bd4f84f_0.geojson"
gdf = gpd.read_file(ons_geojson_url)

# 2. Extract the Local Authority prefix from the LSOA21NM name string
# Example: "Manchester 018B" -> "Manchester"
gdf['LocalAuthority'] = gdf['LSOA21NM'].apply(lambda x: x.rsplit(' ', 1)[0] if isinstance(x, str) else '')

# 3. Filter strictly by our major city hubs (Keeps ALL LSOAs for these cities!)
filtered_gdf = gdf[gdf['LocalAuthority'].isin(allowed_urban_hubs)].copy()

# 4. Clean up the attributes so Qlik gets a lightweight, un-fragmented file
# We keep only the unique LSOA 2021 Code and the clean geometry shapes
output_gdf = filtered_gdf[['LSOA21CD', 'LSOA21NM', 'LocalAuthority', 'geometry']]

# Force standard WGS84 Lat/Long projection for Qlik Map layers
output_gdf = output_gdf.to_crs("EPSG:4326")

# 5. Save the file locally
output_filename = "Dashboard/major_cities_2021_skeleton.geojson"
output_gdf.to_file(output_filename, driver="GeoJSON")

print(f"Success! Filtered down to {len(output_gdf)} urban LSOA boundaries.")
print(f"File saved as: {os.path.abspath(output_filename)}")
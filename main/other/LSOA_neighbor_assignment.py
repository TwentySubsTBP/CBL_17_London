import geopandas as gpd
import pyarrow 
import fastparquet

"""
This script assigns neighbors to LSOAs using geopandas 
"""

lsoa = gpd.read_file(r"data\LSOA_(2021)_EW_BSC_V4_to_Rural_Urban_Classification.geojson")
lsoa = lsoa.to_crs("EPSG:27700")

neighbors = gpd.sjoin(
    lsoa[["LSOA21CD", "geometry"]],
    lsoa[["LSOA21CD", "geometry"]],
    how="inner",
    predicate="touches",
    lsuffix="left",
    rsuffix="right"
)

neighbors = neighbors.rename(columns={
    "LSOA21CD_left": "lsoa",
    "LSOA21CD_right": "neighbor_lsoa"
})

neighbors = neighbors[neighbors["lsoa"] != neighbors["neighbor_lsoa"]]
neighbors = neighbors[["lsoa", "neighbor_lsoa"]].drop_duplicates()


neighbors.to_parquet(
    r"main/data/for training/lsoa_neighbors.parquet",
    index=False
)

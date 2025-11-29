import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

# --- Paths ---
input_path = "data/raw/bus_stops.geojson"
output_dir = "data/preprocessed"
output_path = os.path.join(output_dir, "bus_stops_dtla.geojson")
os.makedirs(output_dir, exist_ok=True)

# --- Load raw JSON ---
print("Loading bus stops dataset...")
with open(input_path, "r") as f:
    data = json.load(f)

features = data["features"]
print(f"Total features in raw dataset: {len(features)}")

# --- Extract attributes ---
attributes_df = pd.json_normalize([feat["properties"] for feat in features])
print("Columns in dataset:", attributes_df.columns.tolist())

# --- Create geometry from lat/lon ---
geometry = [Point(lon, lat) for lon, lat in zip(attributes_df["LONG"], attributes_df["LAT"])]
gdf = gpd.GeoDataFrame(attributes_df, geometry=geometry, crs="EPSG:4326")  # lat/lon
print("CRS:", gdf.crs)
print("Total bus stops:", len(gdf))

# Define DTLA Bounds
DTLA_BOUNDS = {
    "min_lat": 34.02,
    "max_lat": 34.08,
    "min_lon": -118.28,
    "max_lon": -118.23
}

# --- Filter bus stops within bounding box ---
filtered_gdf = gdf[
    (gdf.geometry.y >= DTLA_BOUNDS["min_lat"]) &
    (gdf.geometry.y <= DTLA_BOUNDS["max_lat"]) &
    (gdf.geometry.x >= DTLA_BOUNDS["min_lon"]) &
    (gdf.geometry.x <= DTLA_BOUNDS["max_lon"])
]

print("Filtered bus stops in DTLA bbox:", len(filtered_gdf))

# --- Save filtered bus stops to GeoJSON ---
filtered_gdf.to_file(output_path, driver="GeoJSON")
print(f"✅ Saved {len(filtered_gdf)} bus stops to {output_path}")

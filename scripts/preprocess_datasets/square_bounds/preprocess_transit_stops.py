import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

# --- Paths ---
input_path = "data/raw/la_major_transit_stops.geojson"
output_dir = "data/preprocessed"
output_path = os.path.join(output_dir, "la_metro_transit_stops_dtla.geojson")
os.makedirs(output_dir, exist_ok=True)

# --- Load raw JSON ---
print("Loading LA Metro transit stops dataset...")
with open(input_path, "r") as f:
    data = json.load(f)

features = data["features"]
print(f"Total features in raw dataset: {len(features)}")

# --- Extract attributes ---
attributes_df = pd.json_normalize([feat["attributes"] for feat in features])
print("Columns in dataset:", attributes_df.columns.tolist())

# --- Create geometry from lat/lon ---
geometry = [Point(lon, lat) for lon, lat in zip(attributes_df["lon"], attributes_df["lat"])]
gdf = gpd.GeoDataFrame(attributes_df, geometry=geometry, crs="EPSG:4326")  # lat/lon
print("CRS:", gdf.crs)
print("Total transit stops:", len(gdf))

# --- Define DTLA bounding box ---
DTLA_BOUNDS = {
    "min_lat": 34.02,
    "max_lat": 34.08,
    "min_lon": -118.28,
    "max_lon": -118.23
}

# --- Filter for DTLA stops ---
dtla_gdf = gdf[
    (gdf.geometry.y >= DTLA_BOUNDS["min_lat"]) &
    (gdf.geometry.y <= DTLA_BOUNDS["max_lat"]) &
    (gdf.geometry.x >= DTLA_BOUNDS["min_lon"]) &
    (gdf.geometry.x <= DTLA_BOUNDS["max_lon"])
]

print("Filtered DTLA transit stops:", len(dtla_gdf))

# --- Save filtered stops to GeoJSON ---
dtla_gdf.to_file(output_path, driver="GeoJSON")
print(f"✅ Saved {len(dtla_gdf)} DTLA transit stops to {output_path}")


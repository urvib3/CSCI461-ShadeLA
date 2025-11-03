import geopandas as gpd
import os

# Define input and output paths
input_path = "data/la_food_assistance.geojson"
output_dir = "preprocessed"
output_path = os.path.join(output_dir, "la_food_assistance_dtla.geojson")

# Create folder if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)

print("🥫 Loading Food Assistance dataset...")
gdf = gpd.read_file(input_path)

# Ensure CRS is WGS84
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    print("⚙️ Reprojecting CRS to EPSG:4326...")
    gdf = gdf.to_crs(epsg=4326)

# Define Downtown LA bounding box (approx.)
minx, miny, maxx, maxy = -118.267, 34.034, -118.236, 34.061

# Filter to DTLA
dtla_gdf = gdf.cx[minx:maxx, miny:maxy]

print(f"✅ Found {len(dtla_gdf)} DTLA Food Assistance records.")

# Save filtered data
dtla_gdf.to_file(output_path, driver="GeoJSON")
print(f"💾 Saved to {output_path}")

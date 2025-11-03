import geopandas as gpd
import os

# Define input and output paths
input_path = "data/la_schools_colleges_universities.geojson"
output_dir = "preprocessed"
output_path = os.path.join(output_dir, "la_schools_dtla.geojson")

# Create folder if it doesn’t exist
os.makedirs(output_dir, exist_ok=True)

print("Loading school dataset...")
gdf = gpd.read_file(input_path)

print("CRS before:", gdf.crs)

# Convert CRS if needed
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)
    print("Converted to EPSG:4326")

print("CRS now:", gdf.crs)
print("Total schools:", len(gdf))

# Define approximate DTLA bounding box
DTLA_BOUNDS = {
    "min_lat": 34.02,
    "max_lat": 34.08,
    "min_lon": -118.28,
    "max_lon": -118.23
}

# Filter for DTLA area
dtla_gdf = gdf[
    (gdf.geometry.y >= DTLA_BOUNDS["min_lat"]) &
    (gdf.geometry.y <= DTLA_BOUNDS["max_lat"]) &
    (gdf.geometry.x >= DTLA_BOUNDS["min_lon"]) &
    (gdf.geometry.x <= DTLA_BOUNDS["max_lon"])
]

print("Filtered schools:", len(dtla_gdf))

# Save to new folder
dtla_gdf.to_file(output_path, driver="GeoJSON")

print(f"✅ Saved {len(dtla_gdf)} DTLA schools to {output_path}")
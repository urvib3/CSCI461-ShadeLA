import geopandas as gpd
import os

# Paths
input_path = "data/la_hospitals_clinics_live.geojson"
output_dir = "preprocessed"
output_path = os.path.join(output_dir, "la_hospitals_clinics_dtla.geojson")

os.makedirs(output_dir, exist_ok=True)

print("📥 Loading hospital dataset...")
gdf = gpd.read_file(input_path)

print("CRS before:", gdf.crs)

# If the CRS is not EPSG:4326, convert it
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)
    print("Converted to EPSG:4326")

print("CRS now:", gdf.crs)

# Define DTLA bounding box in lat/lon
DTLA_BOUNDS = {
    "min_lat": 34.02,
    "max_lat": 34.08,
    "min_lon": -118.28,
    "max_lon": -118.23
}

# Filter for hospitals within DTLA
dtla_gdf = gdf[
    (gdf.geometry.y >= DTLA_BOUNDS["min_lat"]) &
    (gdf.geometry.y <= DTLA_BOUNDS["max_lat"]) &
    (gdf.geometry.x >= DTLA_BOUNDS["min_lon"]) &
    (gdf.geometry.x <= DTLA_BOUNDS["max_lon"])
]

# Save preprocessed data
dtla_gdf.to_file(output_path, driver="GeoJSON")

print("Total hospitals:", len(gdf))
print("Example coordinates:")
print(gdf.geometry.y.head(), gdf.geometry.x.head())
print("Filtered hospitals:", len(dtla_gdf))


print(f"✅ Saved {len(dtla_gdf)} DTLA hospital records to {output_path}")

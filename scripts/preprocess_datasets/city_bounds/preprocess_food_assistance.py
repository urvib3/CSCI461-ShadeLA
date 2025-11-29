import geopandas as gpd
import os

input_path="data/raw/la_food_assistance.geojson"
county_path="data/raw/la_city_boundaries.geojson"
target_county="Inglewood"
output_dir="preprocessed"

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"food_assistance_{target_county.lower()}.geojson")

print("🥫 Loading Food Assistance dataset...")
gdf = gpd.read_file(input_path)

# Ensure WGS84
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    print("⚙️ Reprojecting Food Assistance CRS to EPSG:4326...")
    gdf = gdf.to_crs(epsg=4326)

# ---------------------------------------------------------
# Load county boundaries
# ---------------------------------------------------------
print("🗺️ Loading County Boundary dataset...")
county_gdf = gpd.read_file(county_path)

# Fix CRS if needed
if county_gdf.crs is None or county_gdf.crs.to_epsg() != 4326:
    print("⚙️ Reprojecting County CRS to EPSG:4326...")
    county_gdf = county_gdf.to_crs(epsg=4326)

# ---------------------------------------------------------
# Extract the target county polygon
# ---------------------------------------------------------
county_field = None
for col in county_gdf.columns:
    if any(k in col.lower() for k in ["city", "name", "label", "jurisdiction"]):
        county_field = col
        break


if county_field is None:
    raise ValueError("❌ Could not find a county name field in GIS dataset.")

print(f"🔍 Searching for county '{target_county}' in '{county_field}'...")
county_row = county_gdf[county_gdf[county_field].str.lower() == target_county.lower()]

if county_row.empty:
    raise ValueError(f"❌ County '{target_county}' not found in the dataset.")

county_geom = county_row.geometry.unary_union

# ---------------------------------------------------------
# Spatial intersection
# ---------------------------------------------------------
print("📍 Filtering Food Assistance points inside county polygon...")
gdf = gdf.to_crs(county_gdf.crs)  # ensure match
gdf["in_county"] = gdf.geometry.within(county_geom)

county_points = gdf[gdf["in_county"]].drop(columns=["in_county"])

print(f"✅ Found {len(county_points)} food assistance points in {target_county}.")

# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------
county_points.to_file(output_path, driver="GeoJSON")
print(f"💾 Saved to {output_path}")

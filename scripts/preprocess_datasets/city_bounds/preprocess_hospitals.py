import geopandas as gpd
import os

# ---------------------------------------------------------
# Paths and target
# ---------------------------------------------------------
input_path = "data/raw/la_hospitals_clinics_live.geojson"
county_path = "data/raw/la_city_boundaries.geojson"
target_city = "Inglewood"
output_dir = "preprocessed"
output_path = os.path.join(output_dir, f"hospitals_{target_city.lower()}.geojson")

os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# Load hospital dataset
# ---------------------------------------------------------
print("🏥 Loading hospital dataset...")
gdf = gpd.read_file(input_path)

# Ensure WGS84
if gdf.crs is None or gdf.crs.to_epsg() != 4326:
    print("⚙️ Reprojecting hospital CRS to EPSG:4326...")
    gdf = gdf.to_crs(epsg=4326)

# ---------------------------------------------------------
# Load city/county boundaries
# ---------------------------------------------------------
print("🗺️ Loading city boundary dataset...")
county_gdf = gpd.read_file(county_path)

# Fix CRS if needed
if county_gdf.crs is None or county_gdf.crs.to_epsg() != 4326:
    print("⚙️ Reprojecting boundary CRS to EPSG:4326...")
    county_gdf = county_gdf.to_crs(epsg=4326)

# ---------------------------------------------------------
# Extract the target city polygon
# ---------------------------------------------------------
city_field = None
for col in county_gdf.columns:
    if any(k in col.lower() for k in ["city", "name", "label", "jurisdiction"]):
        city_field = col
        break

if city_field is None:
    raise ValueError("❌ Could not find a city name field in GIS dataset.")

print(f"🔍 Searching for city '{target_city}' in '{city_field}'...")
city_row = county_gdf[county_gdf[city_field].str.lower() == target_city.lower()]

if city_row.empty:
    raise ValueError(f"❌ City '{target_city}' not found in the dataset.")

city_geom = city_row.geometry.unary_union

# ---------------------------------------------------------
# Spatial intersection
# ---------------------------------------------------------
print(f"📍 Filtering hospitals inside {target_city}...")
gdf = gdf.to_crs(county_gdf.crs)  # ensure CRS match
gdf["in_city"] = gdf.geometry.within(city_geom)

city_hospitals = gdf[gdf["in_city"]].drop(columns=["in_city"])

print(f"✅ Found {len(city_hospitals)} hospitals in {target_city}.")

# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------
city_hospitals.to_file(output_path, driver="GeoJSON")
print(f"💾 Saved to {output_path}")

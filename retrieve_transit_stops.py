import requests
import json
import time

# BASE_URL=https://services8.arcgis.com/TNoJFjk1LsD45Juj/ArcGIS/rest/services/Transit_Network-LACMTA%20Stop%20Locations/FeatureServer/0/query 
BASE_URL = "https://services8.arcgis.com/TNoJFjk1LsD45Juj/arcgis/rest/services/Major_Transit_Stops_By_GTFS/FeatureServer/47/query"

params = {
    "where": "1=1",           # All features
    "outFields": "*",         # All attributes
    "returnGeometry": "true", # Include geometry
    "f": "json",               # Must use 'json'
    "resultOffset": 0, # Start index "
    "resultRecordCount": 2000 # Number of records per batch
}

all_features = []
batch = 0

while True:
    print(f"Fetching batch {batch} (offset={params['resultOffset']})...")

    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        print("Error:", response.text)
        break

    data = response.json()
    print("data: ", data)
    features = data.get("features", [])
    
    if not features:
        print("No more data.")
        break

    all_features.extend(features)
    params["resultOffset"] += params["resultRecordCount"]
    batch += 1

    # Respect ArcGIS rate limits
    time.sleep(0.3)

# Combine all features into one GeoJSON FeatureCollection
geojson_data = {
    "type": "FeatureCollection",
    "features": all_features
}

# Save file
# output_path = "data/la_metro_stops.geojson"
output_path = "data/la_major_transit_stops.geojson"
with open(output_path, "w") as f:
    json.dump(geojson_data, f)

print(f"✅ Saved {len(all_features)} stops to {output_path}")

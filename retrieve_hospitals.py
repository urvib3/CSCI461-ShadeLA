import requests
import json
import time

BASE_URL = "https://public.gis.lacounty.gov/public/rest/services/LACounty_Dynamic/LMS_Data_Public/MapServer/77/query"

params = {
    "where": "1=1",
    "outFields": "*",
    "returnGeometry": "true",
    "f": "json",
    "resultOffset": 0,
    "resultRecordCount": 2000
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
    features = data.get("features", [])
    if not features:
        print("No more data.")
        break

    all_features.extend(features)
    params["resultOffset"] += params["resultRecordCount"]
    batch += 1
    time.sleep(0.3)

geojson_data = {
    "type": "FeatureCollection",
    "features": all_features
}

output_path = "data/la_hospitals_clinics_live.geojson"
with open(output_path, "w") as f:
    json.dump(geojson_data, f)

print(f"✅ Saved {len(all_features)} records to {output_path}")

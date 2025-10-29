import geopandas as gpd
import json
import os

# Path to your dataset
DATA_PATH = "data/la_hospitals_clinics.geojson"

def load_hospitals():
    if not os.path.exists(DATA_PATH):
        print(" Error: Hospital dataset not found at", DATA_PATH)
        return
    
    try:
        # Load the GeoJSON file using GeoPandas
        hospitals = gpd.read_file(DATA_PATH)

        print("Successfully loaded Hospitals and Clinics dataset.")
        print(f"Total records: {len(hospitals)}")

        # Show the first few rows and columns for inspection
        print(hospitals.head())

        # Optionally export a simplified sample for teammates
        sample_path = "data/sample_hospitals.geojson"
        hospitals.head(5).to_file(sample_path, driver="GeoJSON")
        print(f" Sample saved to {sample_path}")

    except Exception as e:
        print(" Error loading dataset:", e)

if __name__ == "__main__":
    load_hospitals()

import shutil
import os

os.makedirs("data", exist_ok=True)

source_path = "data/la_schools_colleges_universities.geojson"  # downloaded file
destination_path = "data/la_schools_colleges_universities_copy.geojson"

shutil.copy(source_path, destination_path)

print(f" Copied {source_path} → {destination_path}")
import pandas as pd
import geopandas as gpd
import numpy as np
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend
from pulp import *
import sys, os

# ensure Python can see your MILP folder
sys.path.append(os.path.abspath("../"))  # one level up from your notebook

# now import the function
from MILP.distance_optimizer import optimize_shade_placement

use_only_major_transit_stops = False
limit_scope_dtla = True


# --- Load DTLA data ---
bus_stops = gpd.read_file("../../461/data/bus_stops.geojson").to_crs(3857)
bus_stops_dtla = gpd.read_file("../../data/preprocessed/bus_stops_dtla.geojson").to_crs(3857)
major_transit_stops = gpd.read_file("../../data/raw/la_major_transit_stops.geojson").to_crs(3857)
major_transit_stops_dtla = gpd.read_file("../../data/preprocessed/la_major_transit_stops_dtla.geojson").to_crs(3857)

schools_dtla = gpd.read_file("../../data/preprocessed/la_schools_dtla.geojson").to_crs(3857)
hospitals_dtla = gpd.read_file("../../data/preprocessed/la_hospitals_clinics_dtla.geojson").to_crs(3857)
food_dtla = gpd.read_file("../../data/preprocessed/la_food_assistance_dtla.geojson").to_crs(3857)
schools = gpd.read_file("../../data/raw/la_schools_colleges_universities.geojson").to_crs(3857)
hospitals = gpd.read_file("../../data/raw/la_hospitals_clinics_live.geojson").to_crs(3857)
food = gpd.read_file("../../data/raw/la_food_assistance.geojson").to_crs(3857)

ucla_shade_heat = gpd.read_file("../../data/layers/heat_layer.geojson").to_crs(3857)
ucla_shade_socio = gpd.read_file("../../data/layers/socioeconomic_layer.geojson").to_crs(3857)

if limit_scope_dtla: 
    # Get all the possible shade locations for dtla
    possible_shade_locations = major_transit_stops_dtla if use_only_major_transit_stops else bus_stops_dtla

    # --- Combine all public service facilities ---
    public_points = gpd.GeoDataFrame(pd.concat(
    [schools_dtla[['geometry']], hospitals_dtla[['geometry']], food_dtla[['geometry']]],
    ignore_index=True), crs=3857)

else: 
    # Get all the possible shade locations for la county
    possible_shade_locations = major_transit_stops if use_only_major_transit_stops else bus_stops

    # --- Combine all public service facilities ---
    public_points = gpd.GeoDataFrame(pd.concat(
    [schools[['geometry']], hospitals[['geometry']], food[['geometry']]],
    ignore_index=True), crs=3857)

# --- Combine heat and shade layers with bus stops ---
processed_shade_stops = gpd.sjoin(possible_shade_locations, ucla_shade_heat, how = 'left')
processed_shade_stops = processed_shade_stops.drop(columns=['index_right'])
processed_shade_stops = gpd.sjoin(processed_shade_stops, ucla_shade_socio, how = 'left')
print(processed_shade_stops.columns)

# --- Add together heat and socioeconomic layers to visualize point priority by these 2 objectives
processed_shade_stops['heat_socio_layer'] = (
    processed_shade_stops['heat_layer'] + processed_shade_stops['socioeconomic_layer']
)

# --- Run the MILP optimizer ---
optimized_shades = optimize_shade_placement(
    candidate_points=processed_shade_stops,
    public_points=public_points,
    max_shades=30,
    use_spacing=True,
    use_public=True,
    use_heat=True,
    use_socioeconomic=True,
    spacing_threshold=500,
    public_service_threshold=300,
)

print(f"Selected {len(optimized_shades)} optimal shade sites.")

shade_type = "Major Transit" if use_only_major_transit_stops else "Buses"
shade_area = "DTLA" if limit_scope_dtla else "LAC"
num_shades = str(len(optimized_shades))
optimized_shades.to_file(f"../../data/optimized_shades_{shade_type}_{shade_area}_{num_shades}.geojson", driver="GeoJSON")
print(f"Saved {len(optimized_shades)} optimized shade locations to ../../data/optimized_shades.geojson")

# --- Visualize ---
fig, axes = plt.subplots(1, 2, figsize=(24, 8))

# Plot the first graph visualizing optimal shade locations against proximity to public service buildings
possible_shade_locations.plot(ax=axes[0], color='orange', label='{shade_type} Stops', alpha=0.5)
public_points.plot(ax=axes[0], color='gray', label='Public Facilities', alpha=0.4)
optimized_shades.plot(ax=axes[0], color='purple', marker='*', markersize=120, alpha = 0.4, label='Optimal Shade Locations')

cx.add_basemap(axes[0], source=cx.providers.CartoDB.PositronNoLabels)
axes[0].legend()
axes[0].set_title(f"{shade_area} Shade Placement Optimization (MILP) on {shade_type}", fontsize=16)
axes[0].set_axis_off()

# Plot the second graph visualizing heat and socioeconomic priority layers
processed_shade_stops.plot(
    ax=axes[1],
    cmap = 'YlOrRd',
    column="heat_socio_layer",
    markersize=50,
    alpha=0.7
)
optimized_shades.plot(ax=axes[1], color='purple', marker='*', markersize=120, alpha=0.4, label='Optimal Shade Locations')
cx.add_basemap(axes[1], source=cx.providers.CartoDB.PositronNoLabels)
axes[1].legend()
axes[1].set_title(f"{shade_area} {shade_type} Stops Colored by Heat + Socioeconomic Layer", fontsize=16)
axes[1].set_axis_off()

plt.tight_layout()
plt.savefig(f"shade_optimization_images_{shade_type}_{shade_area}_{num_shades}.png", dpi=300, bbox_inches='tight')

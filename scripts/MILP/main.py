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
from MILP.baselines import (
    select_random,
    select_top_by_heat_socio,
    select_greedy,
    select_greedy_heat_only,
    compute_metrics,
)

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

shade_type = "Major Transit" if use_only_major_transit_stops else "Buses"
shade_area = "DTLA" if limit_scope_dtla else "LAC"

# Run scenarios for k in {50, 100}
for k in [50, 100]:
    print(f"\n====== Running scenario with k={k} ======")
    # --- Run the MILP optimizer ---
    optimized_shades = optimize_shade_placement(
        candidate_points=processed_shade_stops,
        public_points=public_points,
        max_shades=k,
        use_spacing=True,
        use_public=True,
        use_heat=True,
        use_socioeconomic=True,
        spacing_threshold=500,
        public_service_threshold=300,
    )

    print(f"Selected {len(optimized_shades)} optimal shade sites.")
    num_shades = str(k)
    optimized_shades.to_file(
        f"../../data/optimized_shades_{shade_type}_{shade_area}_{num_shades}.geojson",
        driver="GeoJSON",
    )
    print(
        f"Saved {len(optimized_shades)} optimized shade locations to ../../data/optimized_shades_{shade_type}_{shade_area}_{num_shades}.geojson"
    )

    # --- Baselines ---
    greedy_shades = select_greedy(
        processed_shade_stops,
        public_points,
        k=k,
        spacing_threshold=500,
        public_service_threshold=300,
    )
    greedy_heat_shades = select_greedy_heat_only(
        processed_shade_stops,
        k=k,
        spacing_threshold=500,
    )
    top_hs_shades = select_top_by_heat_socio(processed_shade_stops, k=k)
    random_shades = select_random(processed_shade_stops, k=k, seed=42)

    # Save baseline outputs
    greedy_shades.to_file(
        f"../../data/optimized_shades_Greedy_{shade_area}_{k}.geojson", driver="GeoJSON"
    )
    greedy_heat_shades.to_file(
        f"../../data/optimized_shades_GreedyHeat_{shade_area}_{k}.geojson", driver="GeoJSON"
    )
    top_hs_shades.to_file(
        f"../../data/optimized_shades_TopHeatSocio_{shade_area}_{k}.geojson", driver="GeoJSON"
    )
    random_shades.to_file(
        f"../../data/optimized_shades_Random_{shade_area}_{k}.geojson", driver="GeoJSON"
    )

    # --- Metrics ---
    milp_metrics = compute_metrics(
        optimized_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=500,
        public_service_threshold=300,
    )
    greedy_metrics = compute_metrics(
        greedy_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=500,
        public_service_threshold=300,
    )
    greedy_heat_metrics = compute_metrics(
        greedy_heat_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=500,
        public_service_threshold=300,
    )
    top_metrics = compute_metrics(
        top_hs_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=500,
        public_service_threshold=300,
    )
    rand_metrics = compute_metrics(
        random_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=500,
        public_service_threshold=300,
    )

    def _fmt(m):
        return (
            f"n={m['count']}, heat_socio_sum={m['sum_heat_socio']:.2f}, "
            f"top-decile hits={m['top_decile_hits']}/{m['top_decile_total']}, "
            f"public_access={m['public_access_score']:.2f}, close_pairs<{500}m={m['close_pairs']}"
        )

    print("\n=== Comparison Metrics (MILP vs Greedy vs Baselines) ===")
    print("MILP        : ", _fmt(milp_metrics))
    print("Greedy      : ", _fmt(greedy_metrics))
    print("Greedy-Heat : ", _fmt(greedy_heat_metrics))
    print("Top-K       : ", _fmt(top_metrics))
    print("Random      : ", _fmt(rand_metrics))

    # --- Visualize (combined overlay) ---
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))

    # Plot the first graph visualizing optimal shade locations against proximity to public service buildings
    possible_shade_locations.plot(
        ax=axes[0], color='orange', label=f'{shade_type} Stops', alpha=0.5
    )
    public_points.plot(ax=axes[0], color='gray', label='Public Facilities', alpha=0.4)
    optimized_shades.plot(
        ax=axes[0], color='purple', marker='*', markersize=120, alpha=0.6, label='MILP'
    )
    greedy_shades.plot(
        ax=axes[0], color='green', marker='*', markersize=100, alpha=0.6, label='Greedy'
    )
    greedy_shades.plot(
        ax=axes[0], color='green', marker='*', markersize=100, alpha=0.6, label='Greedy'
    )
    greedy_heat_shades.plot(
        ax=axes[0], color='red', marker='o', markersize=80, alpha=0.6, label='Greedy-Heat'
    )
    top_hs_shades.plot(
        ax=axes[0], color='blue', marker='v', markersize=60, alpha=0.6, label='Top-Heat+Socio'
    )
    random_shades.plot(
        ax=axes[0], color='black', marker='x', markersize=40, alpha=0.6, label='Random'
    )

    cx.add_basemap(axes[0], source=cx.providers.CartoDB.PositronNoLabels)
    axes[0].legend()
    axes[0].set_title(
        f"{shade_area} Shade Placement Optimization (k={k}) on {shade_type}", fontsize=16
    )
    axes[0].set_axis_off()

    # Plot the second graph visualizing heat and socioeconomic priority layers
    processed_shade_stops.plot(
        ax=axes[1], cmap='YlOrRd', column='heat_socio_layer', markersize=50, alpha=0.7
    )
    optimized_shades.plot(
        ax=axes[1], color='purple', marker='*', markersize=120, alpha=0.6, label='MILP'
    )
    greedy_shades.plot(
        ax=axes[1], color='green', marker='*', markersize=100, alpha=0.6, label='Greedy'
    )
    greedy_heat_shades.plot(
        ax=axes[1], color='red', marker='o', markersize=80, alpha=0.6, label='Greedy-Heat'
    )
    top_hs_shades.plot(
        ax=axes[1], color='blue', marker='v', markersize=60, alpha=0.6, label='Top-Heat+Socio'
    )
    random_shades.plot(
        ax=axes[1], color='black', marker='x', markersize=40, alpha=0.6, label='Random'
    )
    cx.add_basemap(axes[1], source=cx.providers.CartoDB.PositronNoLabels)
    axes[1].legend()
    axes[1].set_title(
        f"{shade_area} {shade_type} Stops Colored by Heat + Socioeconomic Layer (k={k})",
        fontsize=16,
    )
    axes[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(
        f"shade_optimization_images_{shade_type}_{shade_area}_{num_shades}_with_baselines.png",
        dpi=300,
        bbox_inches='tight',
    )

    # --- Additional Visualization: Separate panels for each strategy (site placements vs public facilities) ---
    fig2, axes2 = plt.subplots(1, 5, figsize=(40, 8))

    strategies = [
        (optimized_shades, 'MILP', 'purple', '*', 120),
        (greedy_shades, 'Greedy', 'green', '*', 100),
        (greedy_heat_shades, 'Greedy-Heat', 'red', 'o', 90),
        (top_hs_shades, 'Top-Heat+Socio', 'blue', 'v', 80),
        (random_shades, 'Random', 'black', 'x', 60),
    ]

    for ax, (gdf, name, color, marker, msize) in zip(axes2, strategies):
        possible_shade_locations.plot(ax=ax, color='orange', alpha=0.35, markersize=8)
        public_points.plot(ax=ax, color='gray', alpha=0.25, markersize=8)
        gdf.plot(ax=ax, color=color, marker=marker, markersize=msize, alpha=0.7, label=name)
        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels)
        ax.set_title(f"{name} Placement", fontsize=14)
        ax.set_axis_off()
        ax.legend(loc='lower left', fontsize=9)

    fig2.suptitle(
        f"{shade_area} {shade_type} Shade Placement Strategies (k={k})", fontsize=18
    )
    plt.tight_layout()
    fig2.savefig(
        f"shade_optimization_strategies_{shade_type}_{shade_area}_{num_shades}.png",
        dpi=300,
        bbox_inches='tight',
    )

    # --- Additional Visualization: Heat + Socio Layer comparison MILP vs Greedy ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 8))
    processed_shade_stops.plot(
        ax=axes3[0], cmap='YlOrRd', column='heat_socio_layer', markersize=30, alpha=0.6
    )
    optimized_shades.plot(
        ax=axes3[0], color='purple', marker='*', markersize=120, alpha=0.7, label='MILP'
    )
    cx.add_basemap(axes3[0], source=cx.providers.CartoDB.PositronNoLabels)
    axes3[0].set_title('MILP on Heat+Socio Layer', fontsize=14)
    axes3[0].set_axis_off()
    axes3[0].legend()

    processed_shade_stops.plot(
        ax=axes3[1], cmap='YlOrRd', column='heat_socio_layer', markersize=30, alpha=0.6
    )
    greedy_shades.plot(
        ax=axes3[1], color='green', marker='*', markersize=120, alpha=0.7, label='Greedy'
    )
    cx.add_basemap(axes3[1], source=cx.providers.CartoDB.PositronNoLabels)
    axes3[1].set_title('Greedy on Heat+Socio Layer', fontsize=14)
    axes3[1].set_axis_off()
    axes3[1].legend()

    plt.tight_layout()
    fig3.savefig(
        f"shade_heat_socio_milp_vs_greedy_{shade_type}_{shade_area}_{num_shades}.png",
        dpi=300,
        bbox_inches='tight',
    )

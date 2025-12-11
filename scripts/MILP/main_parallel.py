import pandas as pd
import geopandas as gpd
import numpy as np
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # use non-interactive backend
from pulp import *
import sys, os

from concurrent.futures import ProcessPoolExecutor

# ensure Python can see your MILP folder
sys.path.append(os.path.abspath("../"))  # one level up from your notebook

# now import the function
from MILP.milp import optimize_shade_placement
from MILP.baselines import (
    select_random,
    select_greedy,
    select_greedy_heat_only,
    select_top_by_heat,
    select_top_by_socio, 
    select_top_by_public,
    select_top_by_heat_socio,
    select_top_by_heat_socio_public,
)
from MILP.plotting_utils import plot_milp_with_key_baselines
from utils import circle_weighted_join, compute_metrics, parse_args

args = parse_args()

use_only_major_transit_stops = args.use_only_major_transit_stops
limit_scope_dtla = args.limit_scope_dtla
limit_scope_inglewood = args.limit_scope_inglewood
k_values = args.k_values
walking_threshold = args.walking_threshold  # meters
spacing_threshold = walking_threshold  # meters
public_service_threshold = walking_threshold  # meters
complex_polygon_intersection = args.complex_polygon_intersection # Whether to intersect shade's full coverage polygon with UCLA Shade layer instead of point-polygon intersection
weights = {
            'public': 0.1,
            'heat': 0.1,
            'socio': 0.1,
            'spacing': 1.0,
        }

# Set up directories to store results
results_dir = "results_cpi" if complex_polygon_intersection else "results"
fig_subdir = os.path.join(results_dir, "figures")
shade_placements_subdir = os.path.join(results_dir, "shade_placements")
metrics_subdir = os.path.join(results_dir, "metrics")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(fig_subdir, exist_ok=True)
os.makedirs(shade_placements_subdir, exist_ok=True)
os.makedirs(metrics_subdir, exist_ok=True)

# --- Load LA, DTLA, Inglewood data ---
bus_stops = gpd.read_file("../../461/data/bus_stops.geojson").to_crs(3857)
bus_stops_dtla = gpd.read_file("../../data/preprocessed/bus_stops_dtla.geojson").to_crs(3857)
bus_stops_inglewood = gpd.read_file("../../data/preprocessed/bus_stops_inglewood.geojson").to_crs(3857)
major_transit_stops = gpd.read_file("../../data/raw/la_major_transit_stops.geojson").to_crs(3857)
major_transit_stops_dtla = gpd.read_file("../../data/preprocessed/la_major_transit_stops_dtla.geojson").to_crs(3857)

schools = gpd.read_file("../../data/raw/la_schools_colleges_universities.geojson").to_crs(3857)
schools_dtla = gpd.read_file("../../data/preprocessed/la_schools_dtla.geojson").to_crs(3857)
schools_inglewood = gpd.read_file("../../data/preprocessed/schools_inglewood.geojson").to_crs(3857)

hospitals = gpd.read_file("../../data/raw/la_hospitals_clinics_live.geojson").to_crs(3857)
hospitals_dtla = gpd.read_file("../../data/preprocessed/la_hospitals_clinics_dtla.geojson").to_crs(3857)
hospitals_inglewood = gpd.read_file("../../data/preprocessed/hospitals_inglewood.geojson").to_crs(3857)

food = gpd.read_file("../../data/raw/la_food_assistance.geojson").to_crs(3857)
food_dtla = gpd.read_file("../../data/preprocessed/la_food_assistance_dtla.geojson").to_crs(3857)
food_inglewood = gpd.read_file("../../data/preprocessed/food_assistance_inglewood.geojson").to_crs(3857)

ucla_shade_heat = gpd.read_file("../../data/layers/heat_layer.geojson").to_crs(3857)
ucla_shade_socio = gpd.read_file("../../data/layers/socioeconomic_layer.geojson").to_crs(3857)

if limit_scope_inglewood and use_only_major_transit_stops:
    raise NotImplementedError("Cannot limit to Inglewood and use only major transit stops simultaneously. Use bus stops with Inglewood")
elif limit_scope_dtla: # Optimize against only DTLA
    # Get all the possible shade locations for dtla
    possible_shade_locations = major_transit_stops_dtla if use_only_major_transit_stops else bus_stops_dtla

    # --- Combine all public service facilities ---
    public_points = gpd.GeoDataFrame(pd.concat(
    [schools_dtla[['geometry']], hospitals_dtla[['geometry']], food_dtla[['geometry']]],
    ignore_index=True), crs=3857)
elif limit_scope_inglewood: # Optimize against only Inglewood
    # Get all the possible shade locations for inglewood
    possible_shade_locations = bus_stops_inglewood

    # --- Combine all public service facilities ---
    public_points = gpd.GeoDataFrame(pd.concat(
    [schools_inglewood[['geometry']], hospitals_inglewood[['geometry']], food_inglewood[['geometry']]],
    ignore_index=True), crs=3857)
else: # Optimize against all of Los Angeles
    # Get all the possible shade locations for la county
    possible_shade_locations = major_transit_stops if use_only_major_transit_stops else bus_stops

    # --- Combine all public service facilities ---
    public_points = gpd.GeoDataFrame(pd.concat(
    [schools[['geometry']], hospitals[['geometry']], food[['geometry']]],
    ignore_index=True), crs=3857)

# --- Combine heat and shade layers with bus stops ---
if complex_polygon_intersection: # actually intersect the circle coverage of a shade with polygons (more advanced)
    # First join: heat layer
    processed_shade_stops = circle_weighted_join(
        possible_shade_locations,
        ucla_shade_heat,
        value_col="heat_layer",    # change to your column name
        radius=spacing_threshold
    )

    # Second join: socioeconomic layer
    processed_shade_stops = circle_weighted_join(
        processed_shade_stops,
        ucla_shade_socio,
        value_col="socioeconomic_layer",    # change to your column name
        radius=spacing_threshold
    )

    print(processed_shade_stops.columns)

else: # intersect layers by directly assigning shades the value of the encompassing polygon
    processed_shade_stops = gpd.sjoin(possible_shade_locations, ucla_shade_heat, how = 'left')
    processed_shade_stops = processed_shade_stops.drop(columns=['index_right'])
    processed_shade_stops = gpd.sjoin(processed_shade_stops, ucla_shade_socio, how = 'left')
    print(processed_shade_stops.columns)

# --- Add together heat and socioeconomic layers to visualize point priority by these 2 objectives
processed_shade_stops['heat_socio_layer'] = (
    processed_shade_stops['heat_layer'] + processed_shade_stops['socioeconomic_layer']
)

shade_type = "Major Transit" if use_only_major_transit_stops else "Buses"
shade_area = "LAC"
if limit_scope_inglewood:
    shade_area = "Inglewood"
elif limit_scope_dtla:
    shade_area = "DTLA"

# Run scenarios for k in {50, 100}
for k in k_values:
    print(f"\n====== Running scenario with k={k} ======")
    num_shades = k

    # --- Run MILP optimizers in parallel ---
    # --- Define the 5 MILP configs with names ---
    milp_configs = [
        ("milp_shades", dict(
            candidate_points=processed_shade_stops,
            public_points=public_points,
            max_shades=k,
            use_spacing=True,
            use_public=True,
            use_heat=True,
            use_socioeconomic=True,
            spacing_threshold=spacing_threshold,
            public_service_threshold=public_service_threshold, 
            weights=weights,
        )),
        ("milp_heat_shades", dict(
            candidate_points=processed_shade_stops,
            public_points=public_points,
            max_shades=k,
            use_spacing=False,
            use_public=False,
            use_heat=True,
            use_socioeconomic=False,
            spacing_threshold=spacing_threshold,
            public_service_threshold=public_service_threshold,
            weights=weights,
        )),
        ("milp_socio_shades", dict(
            candidate_points=processed_shade_stops,
            public_points=public_points,
            max_shades=k,
            use_spacing=False,
            use_public=False,
            use_heat=False,
            use_socioeconomic=True,
            spacing_threshold=spacing_threshold,
            public_service_threshold=public_service_threshold,
            weights=weights,
        )),
        ("milp_public_shades", dict(
            candidate_points=processed_shade_stops,
            public_points=public_points,
            max_shades=k,
            use_spacing=False,
            use_public=True,
            use_heat=False,
            use_socioeconomic=False,
            spacing_threshold=spacing_threshold,
            public_service_threshold=public_service_threshold,
            weights=weights,
        )),
        ("milp_heat_socio_public_shades", dict(
            candidate_points=processed_shade_stops,
            public_points=public_points,
            max_shades=k,
            use_spacing=False,
            use_public=True,
            use_heat=True,
            use_socioeconomic=True,
            spacing_threshold=spacing_threshold,
            public_service_threshold=public_service_threshold,
            weights=weights,
        )),
    ]

    def run_milp(args):
        return optimize_shade_placement(**args)

    # Top-level helper for mapping
    def run_milp_from_tuple(config_tuple):
        _, args = config_tuple
        return run_milp(args)

    # Run jobs concurrently using top-level helper and milp configs
    with ProcessPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(run_milp_from_tuple, milp_configs))

    milp_results = {name: result for (name, _), result in zip(milp_configs, results)}

    # Access as:
    milp_shades = milp_results["milp_shades"]
    milp_heat_shades = milp_results["milp_heat_shades"]
    milp_socio_shades = milp_results["milp_socio_shades"]
    milp_public_shades = milp_results["milp_public_shades"]
    milp_heat_socio_public_shades = milp_results["milp_heat_socio_public_shades"]

    # Greedy Baselines
    greedy_shades = select_greedy(
        processed_shade_stops,
        public_points,
        k=k,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        weights=weights,
    )
    greedy_heat_shades = select_greedy_heat_only(
        processed_shade_stops,
        k=k,
        spacing_threshold=spacing_threshold,
    )

    # Top-K Baselines
    top_heat_shades = select_top_by_heat(processed_shade_stops, k=k)
    top_socio_shades = select_top_by_socio(processed_shade_stops, k=k)
    top_public_shades = select_top_by_public(processed_shade_stops, public_points, public_service_threshold=300, k=k)
    top_heat_socio_shades = select_top_by_heat_socio(processed_shade_stops, k=k)
    top_heat_socio_public_shades = select_top_by_heat_socio_public(processed_shade_stops, public_points, public_service_threshold=300, k=k) 

    # Random Baselines
    random_shades = select_random(processed_shade_stops, k=k, seed=42)

    # Save baseline outputs
    # MILP and MILP Baselines
    milp_shades.to_file(
        os.path.join(shade_placements_subdir, f"optimized_shades_MILP_{shade_area}_{k}_{walking_threshold}.geojson"), driver="GeoJSON"
    )
    greedy_shades.to_file(
        os.path.join(shade_placements_subdir, f"optimized_shades_Greedy_{shade_area}_{k}_{walking_threshold}.geojson"), driver="GeoJSON"
    )
    # --- Metrics ---

    # MILP
    milp_metrics = compute_metrics(
        selected=milp_shades,
        candidates=processed_shade_stops,
        public_points=public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )
    milp_heat_metrics = compute_metrics(
        milp_heat_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )
    milp_socio_metrics = compute_metrics(
        milp_socio_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )
    milp_public_metrics = compute_metrics(
        milp_public_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )
    milp_heat_socio_public_metrics = compute_metrics(
        milp_heat_socio_public_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )

    # Top-K
    top_heat_metrics = compute_metrics(
        top_heat_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )
    top_socio_metrics = compute_metrics(
        top_socio_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )
    top_public_metrics = compute_metrics(
        top_public_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )
    top_heat_socio_metrics = compute_metrics(
        top_heat_socio_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )
    top_heat_socio_public_metrics = compute_metrics(
        top_heat_socio_public_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )

    # Greedy 
    greedy_metrics = compute_metrics(
        greedy_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )
    greedy_heat_metrics = compute_metrics(
        greedy_heat_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )
    # Random
    rand_metrics = compute_metrics(
        random_shades,
        processed_shade_stops,
        public_points,
        spacing_threshold=spacing_threshold,
        public_service_threshold=public_service_threshold,
        ucla_shade_heat=ucla_shade_heat,
        ucla_shade_socio=ucla_shade_socio,
    )

    def _fmt(m):
        return (
            f"n={m['count']},"
            f"heat_sum={m['sum_heat']:.2f}, socio_sum={m['sum_socio']:.2f},"
            f"heat_socio_sum={m['sum_heat_socio']:.2f}, "
            f"top-decile hits={m['top_decile_hits']}/{m['top_decile_total']}, "
            f"public_access={m['public_access_score']:.2f}, close_pairs<{500}m={m['close_pairs']}, "
            f"area_coverage={m['area_coverage']:.2f}, "
            f"heat_coverage={m['heat_coverage']:.2f}, socio_coverage={m['socio_coverage']:.2f}"
        )

    # --- Build the metrics string ---
    metrics_text = "\n=== Comparison Metrics (MILP vs Greedy vs Baselines) ===\n"
    metrics_text += f"Top-K Heat       : {_fmt(top_heat_metrics)}\n"
    metrics_text += f"Top-K Socio      : {_fmt(top_socio_metrics)}\n"
    metrics_text += f"Top-K Public     : {_fmt(top_public_metrics)}\n"
    metrics_text += f"Top-K HS         : {_fmt(top_heat_socio_metrics)}\n"
    metrics_text += f"Top-K HSP        : {_fmt(top_heat_socio_public_metrics)}\n"
    metrics_text += f"MILP-Heat        : {_fmt(milp_heat_metrics)}\n"
    metrics_text += f"MILP-Socio       : {_fmt(milp_socio_metrics)}\n"
    metrics_text += f"MILP-Public      : {_fmt(milp_public_metrics)}\n"
    metrics_text += f"MILP-HSP         : {_fmt(milp_heat_socio_public_metrics)}\n"
    metrics_text += f"MILP             : {_fmt(milp_metrics)}\n"
    metrics_text += f"Greedy           : {_fmt(greedy_metrics)}\n"
    metrics_text += f"Random           : {_fmt(rand_metrics)}\n"

    # --- Print to console ---
    print(metrics_text)

    # --- Save to file ---
    out_file_path = os.path.join(metrics_subdir, f"shade_comparison_metrics_{shade_type}_{shade_area}_{num_shades}_{walking_threshold}.txt")
    with open(out_file_path, "w") as f:
        f.write(metrics_text)


    # --- Visualize Shade Placements for MILP and key baselines on separate plots ---
    out_file_path = os.path.join(fig_subdir, f"shade_placement_comparison_{shade_type}_{shade_area}_{num_shades}_{walking_threshold}.png")
    plot_milp_with_key_baselines(
        milp_heat=milp_heat_shades, 
        milp_socio=milp_socio_shades, 
        milp_public=milp_public_shades, 
        milp_hsp=milp_heat_socio_public_shades, 
        milp=milp_shades, 
        greedy=greedy_shades,
        processed_shade_stops=processed_shade_stops,
        public_points=public_points,
        out_file_path=out_file_path)

    
    '''
    # --- Visualize (combined overlay) ---
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))

    # Plot the first graph visualizing optimal shade locations against proximity to public service buildings
    possible_shade_locations.plot(
        ax=axes[0], color='orange', label=f'{shade_type} Stops', alpha=0.5
    )
    public_points.plot(ax=axes[0], color='gray', label='Public Facilities', alpha=0.4)
    milp_shades.plot(
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
    top_heat_socio_shades.plot(
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
    milp_shades.plot(
        ax=axes[1], color='purple', marker='*', markersize=120, alpha=0.6, label='MILP'
    )
    greedy_shades.plot(
        ax=axes[1], color='green', marker='*', markersize=100, alpha=0.6, label='Greedy'
    )
    greedy_heat_shades.plot(
        ax=axes[1], color='red', marker='o', markersize=80, alpha=0.6, label='Greedy-Heat'
    )
    top_heat_socio_shades.plot(
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
        os.path.join(fig_subdir, f"shade_optimization_images_{shade_type}_{shade_area}_{num_shades}_with_baselines.png"),
        dpi=300,
        bbox_inches='tight',
    )
    '''

    '''
    # --- Additional Visualization: Separate panels for each strategy (site placements vs public facilities) ---
    fig2, axes2 = plt.subplots(1, 5, figsize=(40, 8))

    strategies = [
        (milp_shades, 'MILP', 'purple', '*', 120),
        (greedy_shades, 'Greedy', 'green', '*', 100),
        (greedy_heat_shades, 'Greedy-Heat', 'red', 'o', 90),
        (top_heat_socio_shades, 'Top-Heat+Socio', 'blue', 'v', 80),
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
        os.path.join(fig_subdir, f"shade_optimization_strategies_{shade_type}_{shade_area}_{num_shades}.png"),
        dpi=300,
        bbox_inches='tight',
    )
    '''

    '''
    # --- Additional Visualization: Heat + Socio Layer comparison MILP vs Greedy ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 8))
    processed_shade_stops.plot(
        ax=axes3[0], cmap='YlOrRd', column='heat_socio_layer', markersize=30, alpha=0.6
    )
    milp_shades.plot(
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
    out_visual_file_path = os.path.join(fig_subdir, f"shade_heat_socio_milp_vs_greedy_{shade_type}_{shade_area}_{num_shades}.png")
    fig3.savefig(
        out_visual_file_path,
        dpi=300,
        bbox_inches='tight',
    )
    print(f"Saved visualizations to {out_visual_file_path}")
    '''
"""
transit_stop_analysis.py

Fully updated version with:
1. No outlines on histogram bars
2. Minimalist axes (left + bottom only, 1px, no ticks)
3. Public-services plot counts *public services covered* instead of stops covered
4. Pastel color scheme
5. Combined subplot outputs
6. Corrected CLI + main()
"""

import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict
from utils import _compute_distance_matrix, _compute_public_dist_matrix

# Soft pastel colors
PASTEL_COLORS = ['#AEC6CF', '#FFB347', '#77DD77']


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_datasets(base_dir: str = "..") -> Dict[str, gpd.GeoDataFrame]:
    print("Loading datasets...")

    bus_stops_inglewood = gpd.read_file("../../data/preprocessed/bus_stops_inglewood.geojson").to_crs(3857)
    bus_stops_dtla = gpd.read_file("../../data/preprocessed/bus_stops_dtla.geojson").to_crs(3857)
    major_transit_stops_dtla = gpd.read_file("../../data/preprocessed/la_major_transit_stops_dtla.geojson").to_crs(3857)

    schools_dtla = gpd.read_file("../../data/preprocessed/la_schools_dtla.geojson").to_crs(3857)
    hospitals_dtla = gpd.read_file("../../data/preprocessed/la_hospitals_clinics_dtla.geojson").to_crs(3857)
    food_dtla = gpd.read_file("../../data/preprocessed/la_food_assistance_dtla.geojson").to_crs(3857)

    schools_inglewood = gpd.read_file("../../data/preprocessed/schools_inglewood.geojson").to_crs(3857)
    hospitals_inglewood = gpd.read_file("../../data/preprocessed/hospitals_inglewood.geojson").to_crs(3857)
    food_inglewood = gpd.read_file("../../data/preprocessed/food_assistance_inglewood.geojson").to_crs(3857)

    public_points_dtla = gpd.GeoDataFrame(
        pd.concat([schools_dtla[['geometry']], hospitals_dtla[['geometry']], food_dtla[['geometry']]], ignore_index=True),
        crs=3857,
    )

    public_points_inglewood = gpd.GeoDataFrame(
        pd.concat([schools_inglewood[['geometry']], hospitals_inglewood[['geometry']], food_inglewood[['geometry']]], ignore_index=True),
        crs=3857,
    )

    print("Datasets loaded.")
    return {
        'bus stops inglewood': bus_stops_inglewood,
        'bus stops dtla': bus_stops_dtla,
        'major transit stops dtla': major_transit_stops_dtla,
        'public points dtla': public_points_dtla,
        'public points inglewood': public_points_inglewood,
    }


# ---- Distance / Counting Helpers ----

def nearest_neighbor_distances(dist_matrix: np.ndarray) -> np.ndarray:
    dm = dist_matrix.copy()
    np.fill_diagonal(dm, np.inf)
    return dm.min(axis=1)


def count_pairs_below_threshold(dist_matrix: np.ndarray, threshold: float) -> float:
    n = dist_matrix.shape[0]
    if n < 2:
        return 0
    # Upper triangle excluding diagonal
    triu_indices = np.triu_indices(n, k=1)
    pairs_below = dist_matrix[triu_indices] < threshold
    total_pairs = len(triu_indices[0])
    return 100 * np.count_nonzero(pairs_below) / total_pairs



def count_public_services_covered(public_dist_matrix: np.ndarray, threshold: float) -> float:
    """
    Count *public services* (columns) with ≥1 stop within threshold distance.
    """
    if public_dist_matrix.size == 0:
        return 0
    covered = np.any(public_dist_matrix <= threshold, axis=0)
    return 100 * np.count_nonzero(covered) / public_dist_matrix.shape[1]

def average_too_close_neighbors(dist_matrix: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    For each threshold, compute the average number of other stops
    within that threshold distance for each stop.
    
    Returns an array of shape (len(thresholds),)
    """
    n = dist_matrix.shape[0]
    if n < 2:
        return np.zeros_like(thresholds, dtype=float)

    # Exclude self-distance
    dm = dist_matrix.copy()
    np.fill_diagonal(dm, np.inf)

    avg_counts = []
    for t in thresholds:
        # Count how many stops are within threshold for each stop
        counts_per_stop = np.sum(dm <= t, axis=1)
        avg_counts.append(counts_per_stop.mean())
    return np.array(avg_counts)



# ---- Styling Helper ----

def style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(left=False, bottom=False)  # remove tick marks


# ---- Main Analysis ----

def analyze_all(data: Dict[str, gpd.GeoDataFrame], thresholds: np.ndarray, fig_dir: str):
    datasets = [
        ('bus stops inglewood', data['bus stops inglewood'], data['public points inglewood']),
        ('bus stops dtla', data['bus stops dtla'], data['public points dtla']),
        ('major transit stops dtla', data['major transit stops dtla'], data['public points dtla']),
    ]

    nearest_list = []
    pair_percent_list = []
    public_service_percent_list = []
    avg_too_close_list = []  # <-- new
    names = []

    for idx, (name, gdf, public_points) in enumerate(datasets):
        n = len(gdf)
        print(f"Processing {name} with {n} stops")
        if n == 0:
            continue

        dist_matrix = _compute_distance_matrix(gdf.geometry)
        nearest_list.append(nearest_neighbor_distances(dist_matrix))

        pair_percent = [count_pairs_below_threshold(dist_matrix, t) for t in thresholds]
        pair_percent_list.append(pair_percent)

        public_dist_matrix = _compute_public_dist_matrix(gdf, public_points)
        public_percent = [count_public_services_covered(public_dist_matrix, t) for t in thresholds]
        public_service_percent_list.append(public_percent)

        # ---- New: Average too-close stops per stop ----
        avg_too_close = average_too_close_neighbors(dist_matrix, thresholds)
        avg_too_close_list.append(avg_too_close)

        names.append(name)

    # ---- Combined Nearest-Neighbor Histograms ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, nearest in enumerate(nearest_list):
        axes[i].hist(nearest, bins=50, color=PASTEL_COLORS[i], edgecolor='none')  # no outlines
        axes[i].set_title(names[i])
        axes[i].set_xlabel('Nearest Neighbor Distance (m)')
        axes[i].set_ylabel('Number of Stops')
        style_axes(axes[i])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'nearest_neighbors_combined.png'), dpi=150)
    plt.close()

    # ---- Stop Pair Percentage ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, pair_percent in enumerate(pair_percent_list):
        ax.plot(thresholds, pair_percent, marker='o', color=PASTEL_COLORS[i], label=names[i])
    ax.set_xlabel('Spacing Threshold (m)')
    ax.set_ylabel('% of Stop Pairs < Threshold')
    ax.set_title('Percentage of Too Close Stop Pairs vs Threshold')
    ax.set_xlim([0, 1010])
    ax.set_ylim([0, 15])
    style_axes(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'stop_pairs_percentages.png'), dpi=150)
    plt.close()

    # ---- Public Services Coverage Percentage ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, public_percent in enumerate(public_service_percent_list):
        ax.plot(thresholds, public_percent, marker='o', color=PASTEL_COLORS[i], label=names[i])
    ax.set_xlabel('Distance Threshold to Public Service (m)')
    ax.set_ylabel('% of Public Services Covered')
    ax.set_title('Public Service Coverage vs Threshold')
    ax.set_xlim([0, 800])
    ax.set_ylim([0, 105])
    style_axes(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'public_service_coverage.png'), dpi=150)
    plt.close()

    # ---- New: Average Too-Close Stops per Stop ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, avg_counts in enumerate(avg_too_close_list):
        ax.plot(thresholds, avg_counts, marker='o', color=PASTEL_COLORS[i], label=names[i])
    ax.set_xlabel('Spacing Threshold (m)')
    ax.set_ylabel('Average # of Too-Close Stops per Stop')
    ax.set_title('Average Number of Too-Close Stops vs Threshold')
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, None])
    style_axes(ax)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'average_too_close_stops.png'), dpi=150)
    plt.close()



# ---- CLI ----

def main(args):
    out_dir = args.output_dir
    ensure_dir(out_dir)

    fig_dir = os.path.join(out_dir, 'figures')
    ensure_dir(fig_dir)

    data = load_datasets()
    thresholds = np.arange(0, 1001, 10)

    analyze_all(data, thresholds, fig_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transit stop spacing and public service analysis')
    parser.add_argument('--output_dir', default='results_transit_analysis', type=str)
    args = parser.parse_args()
    main(args)

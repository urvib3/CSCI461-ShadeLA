import numpy as np
import geopandas as gpd
import pandas as pd
from typing import Dict, List, Tuple
from MILP.utils import _compute_distance_matrix, _compute_public_dist_matrix, _compute_public_gain, select_random

# Greedy + Baselines and evaluation utilities for shade placement
# Assumes input GeoDataFrames are in a projected CRS in meters (e.g., EPSG:3857)

#greedy by heat + socioeconomic score; no spacing, no public access
def select_top_by_heat_socio(candidate_points: gpd.GeoDataFrame, k: int) -> gpd.GeoDataFrame:
    if 'heat_socio_layer' not in candidate_points.columns:
        # Fallback if column doesn't exist
        hs = candidate_points['heat_layer'].fillna(0) + candidate_points['socioeconomic_layer'].fillna(0)
        candidates = candidate_points.copy()
        candidates['heat_socio_layer'] = hs
    else:
        candidates = candidate_points
    k = min(k, len(candidates))
    return candidates.sort_values('heat_socio_layer', ascending=False).head(k)

#greedy by heat + socioeconomic score; no spacing, no public access
def select_top_by_heat_socio_public(
        candidate_points: gpd.GeoDataFrame, 
        public_points: np.ndarray, 
        public_service_threshold: float, 
        k: int
    ) -> gpd.GeoDataFrame:
    
    # Extract the heat socio layer
    if 'heat_socio_layer' not in candidate_points.columns:
        # Fallback if column doesn't exist
        hs = candidate_points['heat_layer'].fillna(0) + candidate_points['socioeconomic_layer'].fillna(0)
        candidates = candidate_points.copy()
        candidates['heat_socio_layer'] = hs
    else:
        candidates = candidate_points
    heat_socio_values = candidate_points.heat_layer
    heat_socio_norm = (heat_socio_values - heat_socio_values.min()) / (heat_socio_values.max() - heat_socio_values.min() + 1e-8)
        
    # Extract the public norm layer
    public_dists = _compute_public_dist_matrix(candidate_points, public_points) if len(public_points) > 0 else np.zeros((n, 0))
    public_gain = _compute_public_gain(public_dists, public_service_threshold)
    # Normalize public gain
    if np.max(public_gain) > 0:
        public_gain_norm = (public_gain - np.min(public_gain)) / (np.max(public_gain) - np.min(public_gain) + 1e-8)
    else:
        public_gain_norm = np.zeros_like(public_gain)

    # Combine the two normalized layers
    k = min(k, len(candidates))
    candidates['heat_socio_public_layer'] = heat_socio_norm + public_gain_norm
    return candidates.sort_values('heat_socio_public_layer', ascending=False).head(k)

#greedy by heat + socioeconomic score; no spacing, no public access
def select_top_by_public(
    candidate_points: gpd.GeoDataFrame,  
    public_points: np.ndarray, 
    public_service_threshold: float, 
    k: int,
) -> gpd.GeoDataFrame:

    n = len(candidate_points)
    if n == 0 or k <= 0:
        return candidate_points.iloc[[]]
    
    public_dists = _compute_public_dist_matrix(candidate_points, public_points) if len(public_points) > 0 else np.zeros((n, 0))
    public_gain = _compute_public_gain(public_dists, public_service_threshold)
    # Normalize public gain
    if np.max(public_gain) > 0:
        public_gain_norm = (public_gain - np.min(public_gain)) / (np.max(public_gain) - np.min(public_gain) + 1e-8)
    else:
        public_gain_norm = np.zeros_like(public_gain)

    # ---- Select top-k candidates by public gain ----
    top_idx = np.argsort(-public_gain_norm)[:k]   # descending sort
    return candidate_points.iloc[top_idx]


#greedy by heat; no spacing, no public access, no socioeconomic layer
def select_top_by_heat(candidates: gpd.GeoDataFrame, k: int) -> gpd.GeoDataFrame:
    if 'heat_layer' not in candidates.columns:
        raise NotImplementedError("You need to add a 'heat_layer' column to candidate_points to use this baseline.")
    k = min(k, len(candidates))
    return candidates.sort_values('heat_layer', ascending=False).head(k)

#greedy by socio layer; no spacing, no public access, no heat layer
def select_top_by_socio(candidates: gpd.GeoDataFrame, k: int) -> gpd.GeoDataFrame:
    if 'socioeconomic_layer' not in candidates.columns:
        raise NotImplementedError("You need to add a 'socioeconomic_layer' column to candidate_points to use this baseline.")
    k = min(k, len(candidates))
    return candidates.sort_values('socioeconomic_layer', ascending=False).head(k)

def select_greedy(
    candidate_points: gpd.GeoDataFrame,
    public_points: gpd.GeoDataFrame,
    k: int,
    spacing_threshold: float = 500.0,
    public_service_threshold: float = 300.0,
    weights: Dict[str, float] = None,
) -> gpd.GeoDataFrame:
    """Greedy heuristic approximating the MILP objective.

    Score(i | S) = w_pub * norm(public_gain[i]) + w_heat * norm(heat[i]) + w_socio * norm(socio[i])
                   - w_space * sum_{j in S} max(0, 1 - d(i,j)/spacing_threshold)
    """
    if weights is None:
        # Inspired by MILP coefficients; tune as needed
        weights = {
            'public': 0.1,
            'heat': 0.1,
            'socio': 0.1,
            'spacing': 1.0,
        }

    n = len(candidate_points)
    if n == 0 or k <= 0:
        return candidate_points.iloc[[]]

    # Normalize features
    heat = candidate_points['heat_layer'].astype(float).fillna(0.0)
    socio = candidate_points['socioeconomic_layer'].astype(float).fillna(0.0)
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    socio_norm = (socio - socio.min()) / (socio.max() - socio.min() + 1e-8)

    # Distances
    dist_mat = _compute_distance_matrix(candidate_points.geometry)
    public_dists = _compute_public_dist_matrix(candidate_points, public_points) if len(public_points) > 0 else np.zeros((n, 0))
    public_gain = _compute_public_gain(public_dists, public_service_threshold)
    # Normalize public gain
    if np.max(public_gain) > 0:
        public_gain_norm = (public_gain - np.min(public_gain)) / (np.max(public_gain) - np.min(public_gain) + 1e-8)
    else:
        public_gain_norm = np.zeros_like(public_gain)

    selected: List[int] = []
    remaining: set = set(range(n))

    while len(selected) < min(k, n) and remaining:
        best_i = None
        best_score = -1e18
        for i in list(remaining):
            base = (
                weights['public'] * float(public_gain_norm[i])
                + weights['heat'] * float(heat_norm.iloc[i])
                + weights['socio'] * float(socio_norm.iloc[i])
            )
            if selected:
                # spacing penalty: sum of (1 - d/threshold) clipped at [0, 1]
                dists = dist_mat[i, selected]
                penalties = np.maximum(0.0, 1.0 - (dists / (spacing_threshold + 1e-8)))
                spacing_pen = weights['spacing'] * float(np.sum(penalties))
            else:
                spacing_pen = 0.0
            score = base - spacing_pen
            if score > best_score:
                best_score = score
                best_i = i
        # Add the winner
        selected.append(best_i)
        remaining.remove(best_i)

    return candidate_points.iloc[selected]

def select_greedy_heat_only(
    candidate_points: gpd.GeoDataFrame,
    k: int,
    spacing_threshold: float = 500.0,
    spacing_weight: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Greedy heuristic optimizing only heat, with optional spacing penalty.
    """
    if 'heat_layer' not in candidate_points.columns:
        raise ValueError("candidate_points must have a 'heat_layer' column.")

    n = len(candidate_points)
    if n == 0 or k <= 0:
        return candidate_points.iloc[[]]

    # Normalize heat to [0, 1]
    heat = candidate_points['heat_layer'].astype(float).fillna(0.0)
    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # Precompute distances between candidates
    dist_mat = _compute_distance_matrix(candidate_points.geometry)

    selected: List[int] = []
    remaining: set = set(range(n))

    while len(selected) < min(k, n) and remaining:
        best_i = None
        best_score = -1e18

        for i in list(remaining):
            # Benefit: normalized heat only
            base = float(heat_norm.iloc[i])

            # Spacing penalty relative to already-selected sites
            if selected:
                dists = dist_mat[i, selected]
                penalties = np.maximum(0.0, 1.0 - (dists / (spacing_threshold + 1e-8)))
                spacing_pen = spacing_weight * float(np.sum(penalties))
            else:
                spacing_pen = 0.0

            score = base - spacing_pen

            if score > best_score:
                best_score = score
                best_i = i

        selected.append(best_i)
        remaining.remove(best_i)

    return candidate_points.iloc[selected]
import geopandas as gpd
from shapely.geometry import Point
import argparse
import geopandas as gpd
import numpy as np
from typing import Dict

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_only_major_transit_stops", 
                        action="store_true",
                        default=False)

    parser.add_argument("--limit_scope_dtla", 
                        action="store_true",
                        default=False)

    parser.add_argument("--limit_scope_inglewood", 
                        action="store_true",
                        default=False)

    parser.add_argument("--k_values", 
                        nargs="+", 
                        type=int, 
                        default=[10, 20, 50])

    parser.add_argument("--walking_threshold", 
                        type=int, 
                        default=300)

    parser.add_argument("--complex_polygon_intersection", 
                        action="store_true",
                        default=False)

    return parser.parse_args()


def circle_weighted_join(points_gdf, polygons_gdf, value_col, radius):
    """
    Assigns weighted polygon values to points by intersecting each point's
    circular coverage area with polygons and computing an area-weighted average.
    
    points_gdf: GeoDataFrame of points
    polygons_gdf: GeoDataFrame of polygons
    value_col: column in polygons_gdf containing the numeric value
    radius: circle radius (in CRS units)
    """

    results = []

    for idx, point in points_gdf.iterrows():
        # Create the coverage circle
        circle = point.geometry.buffer(radius)

        # Find polygons that intersect this circle
        intersects = polygons_gdf[polygons_gdf.intersects(circle)]

        if intersects.empty:
            results.append(None)
            continue

        # Compute area-weighted values
        weighted_sum = 0
        total_area = 0

        for _, poly in intersects.iterrows():
            overlap = poly.geometry.intersection(circle)
            overlap_area = overlap.area

            weighted_sum += poly[value_col] * overlap_area
            total_area += overlap_area

        results.append(weighted_sum / total_area if total_area > 0 else None)

    # Return a copy of original points with new weighted column
    output = points_gdf.copy()
    output[value_col] = results
    return output

def _compute_distance_matrix(geom: gpd.GeoSeries) -> np.ndarray:
    # Always project to meters before computing distances
    if geom.crs != "EPSG:3857":
        geom = geom.to_crs(3857)

    n = len(geom)
    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        gi = geom.iloc[i]
        for j in range(i + 1, n):
            d = gi.distance(geom.iloc[j])
            dist[i, j] = d
            dist[j, i] = d

    return dist


def _compute_public_dist_matrix(candidates: gpd.GeoDataFrame, public_points: gpd.GeoDataFrame) -> np.ndarray:
    # Ensure both GeoDataFrames are in EPSG:3857
    candidates = candidates.to_crs(epsg=3857)
    public_points = public_points.to_crs(epsg=3857)
    
    n = len(candidates)
    p = len(public_points)
    public_dists = np.zeros((n, p), dtype=float)
    
    for i in range(n):
        gi = candidates.geometry.iloc[i]
        for j in range(p):
            public_dists[i, j] = gi.distance(public_points.geometry.iloc[j])
    
    return public_dists


def _compute_public_gain(public_dists: np.ndarray, public_service_threshold: float, weighting: float = 0.2) -> np.ndarray:
    """Distance-weighted access to public facilities (same shape as in MILP objective).
    Sum over facilities within threshold of (1 - (d/threshold * weighting)).
    """
    n = public_dists.shape[0]
    gain = np.zeros(n, dtype=float)
    if n == 0:
        return gain
    for i in range(n):
        di = public_dists[i]
        mask = di < public_service_threshold
        if np.any(mask):
            gain[i] = float(np.sum(1 - (di[mask] / public_service_threshold * weighting)))
        else:
            gain[i] = 0.0
    return gain


def select_random(candidate_points: gpd.GeoDataFrame, k: int, seed: int = 0) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(seed)
    n = len(candidate_points)
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    return candidate_points.iloc[np.sort(idx)]

def compute_metrics(
    selected: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    public_points: gpd.GeoDataFrame,
    spacing_threshold: float = 500.0,
    public_service_threshold: float = 300.0,
    public_weighting: float = 0.2,
    ucla_shade_heat = None,
    ucla_shade_socio = None,
) -> Dict[str, float]:
    
    # Ensure that shade heat and socio layers are given for shade circle intersection with heat/socio layers
    if ucla_shade_heat is None or ucla_shade_socio is None:
        raise ValueError("UCLA SHADE heat and socioeconomic layers must be provided.")
    
    # Safety if empty
    if len(selected) == 0:
        return {
            'count': 0,
            'sum_heat_socio': 0.0,
            'sum_heat': 0.0,
            'sum_socio': 0.0,
            'top_decile_hits': 0,
            'top_decile_total': int(np.ceil(0.1 * len(candidates))) if len(candidates) > 0 else 0,
            'public_access_score': 0.0,
            'close_pairs': 0,
            'area_coverage': 0,
            'heat_coverage': 0,
            'socio_coverage': 0,
        }
    
    # Recalculate the heat and socioeconomic layers of selected points using complex polygon intersection
    selected = selected.copy()
    selected.drop(columns=['heat_layer', 'socioeconomic_layer'], errors='ignore', inplace=True)
    selected = circle_weighted_join(selected, ucla_shade_heat, 'heat_layer', radius=50)
    selected = circle_weighted_join(selected, ucla_shade_socio, 'socioeconomic_layer', radius=50)
    
    # Sum heat and socio layers
    sum_heat = float(selected['heat_layer'].fillna(0).sum())
    sum_socio = float(selected['socioeconomic_layer'].fillna(0).sum())

    # Sum heat + socio
    sum_heat_socio = float((selected['heat_layer'].fillna(0) + selected['socioeconomic_layer'].fillna(0)).sum())

    # Top decile coverage
    if 'heat_socio_layer' in candidates.columns:
        hs_all = candidates['heat_socio_layer'].fillna(0)
    else:
        hs_all = (candidates['heat_layer'].fillna(0) + candidates['socioeconomic_layer'].fillna(0))
    k_dec = max(1, int(np.ceil(0.1 * len(candidates))))
    top_decile_idx = hs_all.sort_values(ascending=False).head(k_dec).index
    hits = len(set(selected.index).intersection(set(top_decile_idx)))

    # Public access distance-weighted score
    public_dists = _compute_public_dist_matrix(candidates, public_points) if len(public_points) > 0 else np.zeros((len(candidates), 0))
    public_gain_all = _compute_public_gain(public_dists, public_service_threshold, weighting=public_weighting)
    public_access_score = float(public_gain_all[selected.index].sum())

    # Spacing: number of close pairs in the selected set
    sel_geom = selected.geometry.reset_index(drop=True)
    dist_sel = _compute_distance_matrix(sel_geom)
    triu = np.triu_indices(len(selected), k=1)
    close_pairs = int(np.sum(dist_sel[triu] < spacing_threshold))

    # Extract polygon with all intersecting shade coverage circles
    coverage_circles = selected.geometry.buffer(spacing_threshold)
    combined_coverage = coverage_circles.unary_union
    coverage_polygon = gpd.GeoDataFrame(geometry=[combined_coverage], crs=selected.crs)
    coverage_polygon = coverage_polygon.to_crs(ucla_shade_heat.crs)

    # Area Coverage
    area_coverage = coverage_polygon.area.sum()

    # Heat Coverage
    heat_intersection = gpd.overlay(coverage_polygon, ucla_shade_heat, how='intersection')
    heat_coverage = (heat_intersection['heat_layer'] * heat_intersection.area).sum() if not heat_intersection.empty else 0.0

    # Socio Coverage
    socio_intersection = gpd.overlay(coverage_polygon, ucla_shade_socio, how='intersection')
    socio_coverage = (socio_intersection['socioeconomic_layer'] * socio_intersection.area).sum() if not socio_intersection.empty else 0.0

    return {
        'count': int(len(selected)),
        'sum_heat': float(sum_heat),
        'sum_socio': float(sum_socio),
        'sum_heat_socio': float(sum_heat_socio),
        'top_decile_hits': int(hits),
        'top_decile_total': int(k_dec),
        'public_access_score': float(public_access_score),
        'close_pairs': int(close_pairs),
        'area_coverage': float(area_coverage),
        'heat_coverage': float(heat_coverage),
        'socio_coverage': float(socio_coverage),
    }


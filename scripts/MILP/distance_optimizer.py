import geopandas as gpd
import numpy as np
import itertools
import sys
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD

def optimize_shade_placement(candidate_points, public_points, max_shades=15, spacing_threshold=300, public_service_threshold=300, use_spacing=True, use_public=True, use_heat=True, use_socioeconomic=True):
    """
    MILP to select shade locations:
    - maximize coverage near public buildings (schools, hospitals, food)
    - maximize spacing between shades
    """

    n = len(candidate_points)
    p = len(public_points)

    print("n: ", n, " p: ", p)
    sys.stdout.flush()

    # binary variables for shade selection
    x = LpVariable.dicts("x", range(n), cat=LpBinary)

    # pairwise "both selected" variables for distance penalty
    y = {}
    for i, j in itertools.combinations(range(n), 2):
        y[(i, j)] = LpVariable(f"y_{i}_{j}", cat=LpBinary)

    # precompute distances (in meters)
    dist_matrix = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d = candidate_points.geometry.iloc[i].distance(candidate_points.geometry.iloc[j])
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

    # distances from candidates to nearest public site
    public_dists = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            public_dists[i, j] = candidate_points.geometry.iloc[i].distance(public_points.geometry.iloc[j])

    # calculates how valuable each bus stop is when considering proximity to public services
    # encourages both threshold coverage of + closeness to public services
    # adding the 1 encourages general coverage and the the subtraction term penalizes shades that are relatively far from the public services they cover
    public_dist_coverage = np.zeros(n)
    public_service_distance_weighting = 0.2     # A higher value means that we care a lot about how close the service is to the public facility. A lower value means that we just care if it is covered under the public_facility_threshold.
    for i in range(n): 
        public_dist_coverage[i] = sum(
            (1 - (d/public_service_threshold*public_service_distance_weighting))
            for d in public_dists[i] 
            if d < public_service_threshold
        )

    # --- PRINT STATISTICS ---
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]
    dist_stats = (
        f"Candidate distances stats (upper tri) — min: {upper_tri.min():.2f} m, "
        f"max: {upper_tri.max():.2f} m, "
        f"mean: {upper_tri.mean():.2f} m, "
        f"median: {np.median(upper_tri):.2f} m\n"
    )

    public_stats = (
        f"Public distances stats — min: {public_dists.min():.2f} m, "
        f"max: {public_dists.max():.2f} m, "
        f"mean: {public_dists.mean():.2f} m, "
        f"median: {np.median(public_dists):.2f} m\n"
    )

    print("dist_stats: ", dist_stats, "\npublic_stats: ", public_stats)
    sys.stdout.flush()

    # --- OBJECTIVE ---
    # encourage spacing (maximize distance between selected pairs)
    spacing_term = lpSum([
        -1 + (dist_matrix[i, j] / spacing_threshold)
        for (i, j) in y.keys() if y[(i, j)] and dist_matrix[i, j] < spacing_threshold
    ])
  
    

    # --- CONSTRUCT MODEL ---
    
    model = LpProblem("Shade_Placement", LpMaximize)

    # CONSTRAINTS
    # linearize y_ij = x_i AND x_j
    for (i, j) in y.keys():
        model += y[(i, j)] <= x[i]
        model += y[(i, j)] <= x[j]
        model += y[(i, j)] >= x[i] + x[j] - 1
    # max shade constraint
    model += lpSum([x[i] for i in range(n)]) == max_shades

    # OBJECTIVES
    objective = 0
    if use_spacing: 
        objective += 1.0 * spacing_term

    if use_public:         # encourage closeness to public buildings (minimize distance)
        # normalize terms to [0,1]
        public_dist_coverage = (public_dist_coverage - np.min(public_dist_coverage)) / (np.max(public_dist_coverage) - np.min(public_dist_coverage) + 1e-8)
        assert np.min(public_dist_coverage) >= 0 and np.max(public_dist_coverage) <= 1, \
        f"Normalization error: min={np.min(public_dist_coverage)}, max={np.max(public_dist_coverage)}"

        # add public proximity term
        public_proximity_term = lpSum([
            x[i] * public_dist_coverage[i] 
            for i in range(len(candidate_points))
        ])
        objective += 0.1 * public_proximity_term

    if use_heat:            # encourage shades in areas with high heat indexes
        # normalize terms to [0,1]
        heat_values = candidate_points.heat_layer
        heat_values = (heat_values - heat_values.min()) / (heat_values.max() - heat_values.min() + 1e-8)
        assert np.min(heat_values) >= 0 and np.max(heat_values) <= 1, \
            f"Heat normalization error: min={np.min(heat_values)}, max={np.max(heat_values)}"

        # add heat term
        heat_term = lpSum([
            x[i] * heat_values.iloc[i] 
            for i in range(len(candidate_points))
        ])
        objective += 0.02 * heat_term

    if use_socioeconomic:   # encourage shades in areas with low socioeconomic status
        # normalize terms to [0,1]
        socio_values = candidate_points.socioeconomic_layer
        socio_values = (socio_values - socio_values.min()) / (socio_values.max() - socio_values.min() + 1e-8)
        assert np.min(socio_values) >= 0 and np.max(socio_values) <= 1, \
            f"Socioeconomic normalization error: min={np.min(socio_values)}, max={np.max(socio_values)}"

        # add socioeconomic term
        socio_term = lpSum([
            x[i] * socio_values.iloc[i] 
            for i in range(len(candidate_points))
        ])

        objective += 0.02 * socio_term
    model += objective

    # --- SOLVE ---
    model.solve(PULP_CBC_CMD(msg=True))

    # Selected shades
    selected_idx = [i for i in range(n) if x[i].value() == 1]

    # Calculate success metrics
    # Count of shade pairs >= spacing_threshold
    count_close_pairs = 0
    for i, j in itertools.combinations(selected_idx, 2):
        if dist_matrix[i, j] < spacing_threshold:
            count_close_pairs += 1

    # Count of public facilities within threshold
    covered_facilities = 0
    for i in selected_idx:
        covered_facilities += np.sum(public_dists[i] < public_service_threshold)

    print(f"Shade pairs closer than threshold: {count_close_pairs}")
    print(f"Public facilities covered by selected shades: {covered_facilities}")

    return candidate_points.iloc[selected_idx]

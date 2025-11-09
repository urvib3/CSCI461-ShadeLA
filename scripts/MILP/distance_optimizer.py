import geopandas as gpd
import numpy as np
import itertools
import sys
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD

def optimize_shade_placement(candidate_points, public_points, max_shades=15, spacing_threshold=300, public_service_threshold=300, use_spacing=True, use_public=True, use_heat=False, use_socioeconomic=False):
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

    # encourage closeness to public buildings (minimize distance)
    public_service_distance_weighting = 0.2     # A higher value means that we care a lot about how close the service is to the public facility. A lower value means that we just care if it is covered under the public_facility_threshold.
    public_proximity_term = lpSum([
        x[i] * (1 - (d/public_service_threshold*public_service_distance_weighting))
        for i in range(n)
        for d in public_dists[i] 
        if d < public_service_threshold
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
    if use_public: 
        objective += 3.0 * public_proximity_term
    model += objective
    #Prioritize accessibility for community facilities = 3, we can lower it

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

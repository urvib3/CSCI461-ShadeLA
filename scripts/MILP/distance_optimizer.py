import geopandas as gpd
import numpy as np
import itertools
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary

def optimize_shade_placement(candidate_points, public_points, max_shades=15, threshold=300):
    """
    MILP to select shade locations:
    - maximize coverage near public buildings (schools, hospitals, food)
    - maximize spacing between shades
    """

    n = len(candidate_points)
    model = LpProblem("Shade_Placement", LpMaximize)

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
    public_dists = np.zeros(n)
    for i in range(n):
        public_dists[i] = candidate_points.geometry.iloc[i].distance(public_points.unary_union)

    # --- CONSTRAINTS ---
    # linearize y_ij = x_i AND x_j
    for (i, j) in y.keys():
        model += y[(i, j)] <= x[i]
        model += y[(i, j)] <= x[j]
        model += y[(i, j)] >= x[i] + x[j] - 1

    # --- OBJECTIVE ---
    # encourage spacing (maximize distance between selected pairs)
    spacing_term = lpSum([
        dist_matrix[i, j] * y[(i, j)]
        for (i, j) in y.keys() if dist_matrix[i, j] < threshold
    ])

    # encourage closeness to public buildings (minimize distance)
    proximity_term = lpSum([
        (1 / (public_dists[i] + 1)) * x[i]
        for i in range(n)
    ])

    # combine objectives (weights can be tuned)
    model += 1.0 * spacing_term + 3.0 * proximity_term
    #Prioritize accessibility for community facilities = 3, we can lower it

    # --- CONSTRAINTS ---
    model += lpSum([x[i] for i in range(n)]) <= max_shades

    # --- SOLVE ---
    model.solve()

    selected_idx = [i for i in range(n) if x[i].value() == 1]
    return candidate_points.iloc[selected_idx]

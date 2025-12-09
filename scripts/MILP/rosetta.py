import os
import re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# CONFIGURATION
# ---------------------------------------------
FILES = [
    "shade_comparison_metrics_Buses_Inglewood_10_150.txt",
    "shade_comparison_metrics_Buses_Inglewood_10_200.txt",
    "shade_comparison_metrics_Buses_Inglewood_10_300.txt",
    "shade_comparison_metrics_Buses_Inglewood_20_150.txt",
    "shade_comparison_metrics_Buses_Inglewood_20_200.txt",
    "shade_comparison_metrics_Buses_Inglewood_20_300.txt",
    "shade_comparison_metrics_Buses_Inglewood_50_150.txt",
    "shade_comparison_metrics_Buses_Inglewood_50_200.txt",
    "shade_comparison_metrics_Buses_Inglewood_50_300.txt",
    "shade_comparison_metrics_Major_Transit_DTLA_10_150.txt",
    "shade_comparison_metrics_Major_Transit_DTLA_10_200.txt",
    "shade_comparison_metrics_Major_Transit_DTLA_10_300.txt",
    "shade_comparison_metrics_Major_Transit_DTLA_20_150.txt",
    "shade_comparison_metrics_Major_Transit_DTLA_20_200.txt",
    "shade_comparison_metrics_Major_Transit_DTLA_20_300.txt",
    "shade_comparison_metrics_Major_Transit_DTLA_50_150.txt",
    "shade_comparison_metrics_Major_Transit_DTLA_50_200.txt",
    "shade_comparison_metrics_Major_Transit_DTLA_50_300.txt",
]
RESULT_DIR = "results_cpi/"
METRICS_DIR = os.path.join(RESULT_DIR, "metrics/")

for i in range(len(FILES)):
    FILES[i] = os.path.join(METRICS_DIR, FILES[i])

METHODS_TO_EXTRACT = [
    "MILP-Heat",
    "MILP-Socio",
    "MILP-Public",
    "MILP-HSP",
    "MILP",
    "Greedy",
]

METRIC_KEYS = ["heat_sum", "socio_sum", "public_access", "area_coverage"]


# ---------------------------------------------
# PARSE ONE FILE
# ---------------------------------------------
def parse_metrics_file(filepath):
    """Parse a metrics text file into dictionary {method: metric_dict}."""
    results = {}

    with open(filepath, "r") as f:
        for line in f:
            for method in METHODS_TO_EXTRACT:
                if line.startswith(method):
                    # Extract key=value pairs
                    pairs = re.findall(r"(\w+)=([\d\.]+)", line)
                    metrics = {k: float(v) for k, v in pairs if k in METRIC_KEYS}
                    results[method] = metrics

    return results


# ---------------------------------------------
# NORMALIZATION
# ---------------------------------------------
def normalize_metrics(method_results):
    """
    Normalize each metric across all methods to [0, 1].
    method_results: dict {method: {metric: value}}
    """
    normed = {m: {} for m in method_results}

    for metric in METRIC_KEYS:
        vals = [method_results[m][metric] for m in method_results]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx > mn else 1.0

        for method in method_results:
            normed[method][metric] = (method_results[method][metric] - mn) / rng

    return normed


# ---------------------------------------------
# ROSETTA (RADAR) PLOT
# ---------------------------------------------
def plot_rosetta(normed_metrics, out_path):
    labels = METRIC_KEYS
    N = len(labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close circle

    pastel_colors = {
        "MILP-Heat": "#c39bd3",     # pastel purple
        "MILP-Socio": "#7fc9e3",    # pastel blue
        "MILP-Public": "#f7a7a6",   # pastel coral/pink
        "MILP-HSP": "#ffd1a4",      # pastel peach/orange
        "MILP": "#a8e6cf",          # pastel mint/green
        "Greedy": "#ffb3de",        # pastel magenta
    }


    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)

    ax.set_yticklabels([])

    for method, metrics in normed_metrics.items():
        values = list(metrics.values())
        values += values[:1]
        ax.plot(angles, values, color=pastel_colors[method], linewidth=2, label=method)
        ax.fill(angles, values, color=pastel_colors[method], alpha=0.25)

    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=11)
    plt.title("Rosetta Metrics Comparison", fontsize=18, pad=20)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------
# HELPERS
# ---------------------------------------------
def extract_metadata_from_filename(path):
    """
    Example filename:
        shade_comparison_metrics_Buses_Inglewood_10_150.txt
    Returns (TransitType, City, NumShades, WalkingThreshold)
    """
    base = os.path.basename(path)
    parts = base.replace(".txt", "").split("_")

    transit_type = parts[3]
    city = parts[4]
    num_shades = parts[5]
    walking_threshold = parts[6]

    return transit_type, city, num_shades, walking_threshold


# ---------------------------------------------
# MAIN
# ---------------------------------------------
def main():
    if not FILES:
        print("No metric files provided in FILES list.")
        return

    for file in FILES:
        print(f"Processing: {file}")

        metrics = parse_metrics_file(file)
        normed = normalize_metrics(metrics)

        transit, city, num_shades, walk = extract_metadata_from_filename(file)

        out_path = (
            os.path.join(RESULT_DIR, f"rosettas/rosetta_{transit}_{city}_{num_shades}_{walk}.png")
        )

        plot_rosetta(normed, out_path)
        print(f"Saved rosetta plot → {out_path}\n")


if __name__ == "__main__":
    main()

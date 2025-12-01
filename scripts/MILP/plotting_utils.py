import matplotlib.pyplot as plt
import contextily as cx

def plot_milp_with_baselines(
    milp_heat,
    milp_socio,
    milp_public,
    milp_hsp,   # heat + socio + public
    milp,
    topk_heat,
    topk_socio,
    topk_public,
    topk_hsp,
    greedy,
    processed_shade_stops,
    public_points,
    out_file_path,
):
    """
    Produce a 2x5 visualization of shade placement strategies across MILP and Top-k + Greedy baselines.
    """

    # --- Plot configuration (consistent markers & sizes) ---
    STYLE = dict(
        milp=dict(color="purple", marker="*", size=140),
        topk=dict(color="blue", marker="v", size=120),
        greedy=dict(color="green", marker="*", size=140),
    )

    # --- Helper: draw processed heat/socio layers ---
    def plot_background(ax, mode):
        if mode == "heat":
            processed_shade_stops.plot(
                ax=ax, column="heat_layer", cmap="YlOrRd", markersize=18, alpha=0.7
            )
        elif mode == "socio":
            processed_shade_stops.plot(
                ax=ax, column="socioeconomic_layer", cmap="YlGnBu", markersize=18, alpha=0.7
            )
        elif mode == "public":
            processed_shade_stops.plot(
                ax=ax, color='yellow', markersize=18, alpha=0.7
            )
            public_points.plot(ax=ax, color="gray", markersize=20, alpha=0.4)
        elif mode == "full":  # heat + socio + public
            processed_shade_stops.plot(
                ax=ax, column="heat_socio_layer", cmap="Purples", markersize=18, alpha=0.6
            )
            public_points.plot(ax=ax, color="gray", markersize=20, alpha=0.35)
        elif mode == "none":
            pass

    # --- Helper: draw shade placements ---
    def plot_points(ax, gdf, label, style_key):
        st = STYLE[style_key]
        gdf.plot(
            ax=ax,
            color=st["color"],
            marker=st["marker"],
            markersize=st["size"],
            alpha=0.75,
            label=label,
        )

    # --- Set up figure ---
    fig, axes = plt.subplots(2, 5, figsize=(40, 12))
    fig.suptitle("MILP, Top-k, and Greedy Shade Placement Comparisons", fontsize=22)

    # --- Define the 10 plots (2x5) ---
    PLOTS = [
        # Row 1 = MILP variants
        (milp_heat,     "MILP - Heat Layer",              "heat",  "milp"),
        (milp_socio,    "MILP - Socioeconomic Layer",     "socio", "milp"),
        (milp_public,   "MILP - Public Proximity",        "public","milp"),
        (milp_hsp,      "MILP - Heat+Socio+Public",       "full",  "milp"),
        (milp,          "MILP",                           "full",  "milp"),

        # Row 2 = Top-k + Greedy variants
        (topk_heat,     "Top-k Heat Layer",               "heat",  "topk"),
        (topk_socio,    "Top-k Socioeconomic Layer",      "socio", "topk"),
        (topk_public,   "Top-k Public Proximity",         "public","topk"),
        (topk_hsp,      "Top-k Heat+Socio+Public",        "full",  "topk"),
        (greedy,        "Greedy",                         "full",  "greedy"),
    ]

    # --- Render each subplot ---
    for ax, (gdf, title, bg_mode, style_key) in zip(axes.ravel(), PLOTS):
        plot_background(ax, bg_mode)
        plot_points(ax, gdf, title, style_key)

        cx.add_basemap(ax, source=cx.providers.CartoDB.PositronNoLabels)
        ax.set_title(title, fontsize=15)
        ax.set_axis_off()
        ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()

    # --- Save the figure ---
    plt.savefig(out_file_path, dpi=300, bbox_inches="tight")
    print(f"Saved MILP vs Baselines plot to: {out_file_path}")

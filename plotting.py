"""Figures of the notebook that need more than a few lines: pie maps of spot compositions, improvement maps."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge


def pie_map(ax, xy, proportions, colors, radius):
    """Draw one pie per spot: xy (n, 2) centres, proportions (n, K) rows summing to 1 (all-zero rows are skipped),
    colors: list of K colours, radius in data units."""
    wedges, facecolors = [], []
    for (x, y), p in zip(xy, proportions):
        if p.sum() <= 0:
            continue
        start = 90.0
        for k in np.nonzero(p)[0]:
            span = 360.0 * p[k] / p.sum()
            wedges.append(Wedge((x, y), radius, start - span, start))
            facecolors.append(colors[k])
            start -= span
    collection = PatchCollection(wedges, facecolors=facecolors, edgecolors="none", rasterized=True)
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def composition_maps(xy, panels, cell_types, palette, radius, window_half=300.0, title=None):
    """One pie map per panel ({name: (n, K) proportions}) on the same spots `xy`, plus an inset zoom on the densest region.

    Spots whose proportions sum to 0 are omitted. The inset window is centred on the spot with the most spots around it.
    """
    colors = [palette[ct] for ct in cell_types]
    n_around = np.array([len(hits) for hits in cKDTree(xy).query_ball_point(xy, r=window_half)])
    cx, cy = xy[n_around.argmax()]
    height = 6.6 * np.ptp(xy[:, 1]) / np.ptp(xy[:, 0]) + 1.4          # panel height follows the section's aspect ratio
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), height), dpi=130)
    for ax, (name, P) in zip(np.atleast_1d(axes), panels.items()):
        pie_map(ax, xy, P, colors, radius)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        ax.set_title(name, fontsize=11)
        axins = ax.inset_axes([0.66, 0.02, 0.33, 0.33])
        win = (np.abs(xy[:, 0] - cx) < window_half) & (np.abs(xy[:, 1] - cy) < window_half)
        pie_map(axins, xy[win], P[win], colors, radius)
        axins.set_xlim(cx - window_half, cx + window_half); axins.set_ylim(cy + window_half, cy - window_half)
        axins.set_aspect("equal"); axins.set_xticks([]); axins.set_yticks([])
        ax.indicate_inset_zoom(axins, edgecolor="black")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6, color=palette[ct], label=ct) for ct in cell_types]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False, fontsize=9, title="cell type")
    if title:
        fig.suptitle(title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def improvement_maps(xy, per, method, baseline, titles):
    """Where `method` beats `baseline`: one panel per metric, green = better, red = worse (spots evaluated in `per`)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6), dpi=110)
    for ax, k in zip(axes, ("pcc", "rmse", "jsd")):
        x, y = per[method][k], per[baseline][k]
        ok = np.isfinite(x) & np.isfinite(y)
        better = ((x > y) if k == "pcc" else (x < y)) & ok
        ax.scatter(xy[ok & ~better, 0], xy[ok & ~better, 1], s=5, c="#d9534f", linewidths=0, rasterized=True, label=f"{baseline} better")
        ax.scatter(xy[better, 0], xy[better, 1], s=5, c="#2ca25f", linewidths=0, rasterized=True, label=f"{method} better")
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        ax.set_title(f"{titles[k]}: {method} better on {better[ok].mean():.0%} of spots", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, markerscale=3, fontsize=9, frameon=False, loc="center left", bbox_to_anchor=(0.99, 0.5))
    fig.tight_layout()
    return fig

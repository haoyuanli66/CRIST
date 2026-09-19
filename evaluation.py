"""Step 7 of CRIST: per-spot metrics and the benchmark-style box plots (as in the paper's figure scripts).

Per-spot definitions: Pearson correlation across cell types, RMSE = sqrt(mean over types of the squared
error) and Jensen-Shannon divergence with the natural logarithm, every method compared on the spots that
contain at least one annotated cell. Box plots: filled boxes, black medians, 1.5-IQR whiskers, no fliers.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

COLORS = {"Cell2location": "#4C78A8", "CRIST": "#59A14F"}
TITLES = {"pcc": "PCC ↑", "rmse": "RMSE ↓", "jsd": "JSD ↓"}
TICK = {"pcc": 0.25, "rmse": 0.10, "jsd": 0.25}


def normalize_celltype_name(name):
    return name.lower().replace(" ", "_").replace("-", "_")


def to_proportions(df, spot_ids, cell_types):
    """(spots x cell types) proportions from a Cell2location abundance table or a CRIST proportion table."""
    lookup = {normalize_celltype_name(c): c for c in df.columns}
    missing = [t for t in cell_types if normalize_celltype_name(t) not in lookup]
    if missing:
        raise ValueError(f"missing cell types {missing}")
    rows = df.reindex(spot_ids)
    if rows.isna().all(axis=1).any():
        raise ValueError(f"{int(rows.isna().all(axis=1).sum())} spots have no prediction")
    arr = rows[[lookup[normalize_celltype_name(t)] for t in cell_types]].to_numpy(dtype=float)
    arr = np.clip(np.nan_to_num(arr, nan=0.0), 0, None)
    return arr / (arr.sum(axis=1, keepdims=True) + 1e-12)


def _kl(p, q, eps=1e-12):
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def per_spot_metrics(G, P):
    """G, P: (spots x types) proportions -> dict of per-spot PCC, RMSE and JSD vectors."""
    pcc = np.full(len(G), np.nan)
    for k in range(len(G)):
        if G[k].std() > 0 and P[k].std() > 0:
            pcc[k] = np.corrcoef(G[k], P[k])[0, 1]
    rmse = np.sqrt(np.mean((G - P) ** 2, axis=1))
    jsd = np.zeros(len(G))
    for k in range(len(G)):
        p, t = P[k] / (P[k].sum() + 1e-12), G[k] / (G[k].sum() + 1e-12)
        m = 0.5 * (p + t)
        jsd[k] = 0.5 * _kl(t, m) + 0.5 * _kl(p, m)
    return {"pcc": pcc, "rmse": rmse, "jsd": jsd}


def compare(gt_counts, predictions, spot_ids, cell_types):
    """Per-spot metrics of every method on the spots with >= 1 annotated cell. predictions: {method: DataFrame}.

    Returns (per: {method: {pcc, rmse, jsd arrays}}, summary: mean per-spot metric per method, valid: mask of the evaluated spots).
    """
    gt_counts = np.asarray(gt_counts, dtype=float)
    valid = gt_counts.sum(axis=1) > 0
    G = (gt_counts / (gt_counts.sum(axis=1, keepdims=True) + 1e-12))[valid]
    per = {m: per_spot_metrics(G, to_proportions(df, spot_ids, cell_types)[valid]) for m, df in predictions.items()}
    summary = pd.DataFrame({m: {TITLES[k]: np.nanmean(v[k]) for k in ("pcc", "rmse", "jsd")} for m, v in per.items()}).T
    print(f"{int(valid.sum()):,} spots with annotated cells; mean per-spot metrics:")
    return per, summary, valid


def significance(per, method, baseline):
    """Paired one-sided Wilcoxon signed-rank test over spots that `method` beats `baseline` on each metric."""
    rows = []
    for k in ("pcc", "rmse", "jsd"):
        x, y = per[method][k], per[baseline][k]
        ok = np.isfinite(x) & np.isfinite(y)
        better = (x[ok] > y[ok]) if k == "pcc" else (x[ok] < y[ok])
        rows.append({"metric": TITLES[k], f"{baseline} mean": y[ok].mean(), f"{method} mean": x[ok].mean(),
                     f"{method} better (fraction of spots)": better.mean(),
                     "paired Wilcoxon p (one-sided)": wilcoxon(x[ok], y[ok], alternative="greater" if k == "pcc" else "less").pvalue})
    return pd.DataFrame(rows).set_index("metric")


def _whisker_extent(arrays):
    lo, hi = [], []
    for v in arrays:
        q1, q3 = np.percentile(v, [25, 75])
        inside = v[(v >= q1 - 1.5 * (q3 - q1)) & (v <= q3 + 1.5 * (q3 - q1))]
        lo.append(inside.min())
        hi.append(inside.max())
    return min(lo), max(hi)


def _panel_limits(arrays, metric):
    step = TICK[metric]
    wmin, wmax = _whisker_extent(arrays)
    if metric == "pcc":
        ymin, ymax = max(-1.0, np.floor(wmin / step) * step), min(1.0, np.ceil(wmax / step) * step)
    else:
        ymin, ymax = (0.0 if wmin >= 0 else np.floor(wmin / step) * step), max(step, np.ceil(wmax / step) * step)
    if ymax <= ymin:
        ymax = ymin + step
    return ymin, ymax, np.arange(ymin, ymax + step / 2, step)


def boxplots(per, title, methods=None):
    """Three panels (PCC, RMSE, JSD) with one box per method, in the style of the paper's benchmark figure."""
    methods = methods or list(per)
    colors = [COLORS.get(m, "#BAB0AC") for m in methods]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
    fig, axes = plt.subplots(1, 3, figsize=(3.6 * 3, 4.2))
    for ax, metric in zip(axes, ("pcc", "rmse", "jsd")):
        arrays = [per[m][metric][np.isfinite(per[m][metric])] for m in methods]
        ymin, ymax, ticks = _panel_limits(arrays, metric)
        bp = ax.boxplot(arrays, tick_labels=methods, patch_artist=True, widths=0.6, showfliers=False,
                        boxprops=dict(linewidth=1.2, edgecolor="black"), medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(color="black", linewidth=1.2), capprops=dict(color="black", linewidth=1.2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.72)
        for cap in bp["caps"]:
            cap.set_clip_on(False)
        ax.set_ylim(ymin - max((ymax - ymin) * 0.03, 0.01) if ymin >= 0 else ymin, ymax)
        ax.set_yticks(ticks)
        ax.set_xlim(0.4, len(arrays) + 0.6)
        ax.tick_params(labelsize=11, direction="out", length=4)
        ax.xaxis.set_tick_params(labelsize=13)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("black")
            ax.spines[side].set_linewidth(1.0)
        ax.set_title(TITLES[metric], fontsize=16, pad=12)
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.5)
    return fig

"""Step 0 of CRIST: cell-type labels for an annotated high-resolution ST sample, from a single-cell atlas with TACCO.

Three input layouts give a cell-level expression matrix (raw counts) with cell positions:
  xenium        10x Xenium outs (cell_feature_matrix.h5 + cells.parquet): the vendor's segmented cells
  hd-segmented  Visium HD with Space Ranger cell segmentation (segmented_outputs/filtered_feature_cell_matrix.h5
                + cell_segmentations.geojson): one profile per segmented cell, centroid of its polygon
  hd-bin2cell   Visium HD without cell segmentation: the CellViT nuclei of the H&E image (step 2) are rasterised
                into a label image, every nucleus label is expanded by up to `max_bin_distance` 2 um bins to a cell
                footprint (bin2cell's expand_labels rule: nearest nucleus, ties broken in expression PCA space),
                and the bins of each footprint are summed to a cell profile. The vendored ./bin2cell package
                (v0.3.3) reads the binned outputs; its expand_labels and bin_to_cell are re-implemented below
                (insert_labels, expand_labels, bins_to_cells) with batched KD-tree queries and per-batch PCA
                tie-breaking, exactly as in the pipeline that produced the paper's Visium HD labels
TACCO then annotates the cells against the atlas: one reference profile per atlas class
(tc.preprocessing.construct_reference_profiles), tc.tl.annotate with max_annotation = 1;
obs['cell_type_tacco'] is the class with the largest weight (NaN when TACCO assigns none) and
obs['cell_type_confidence'] that weight.

TACCO and bin2cell need their own conda envs (TACCO_ENV, BIN2CELL_ENV below), so this file is both a module the
notebook imports and the command-line script those envs execute:
  python annotation.py xenium       --sample_dir ... --atlas ... --labels_key ... --out ...
  python annotation.py hd-segmented --segmented_dir ... --atlas ... --labels_key ... --out ...
  python annotation.py hd-bin2cell  --visium_hd_path ... --he_image ... --cellvit_dir ... --out_dir ...   (bin2cell env)
  python annotation.py tacco        --cells <cell-level h5ad> --atlas ... --labels_key ... --out ...      (TACCO env)
"""
import argparse
import os
import gc
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import scipy.spatial

HERE = Path(__file__).resolve().parent
TACCO_ENV = "tacco_env"                      # conda env with tacco 0.4.0 (python 3.10, scanpy, anndata)
BIN2CELL_ENV = "visium_hd_annotation"        # conda env with the imports of the vendored ./bin2cell: stardist, opencv, scikit-image, seaborn, tifffile, scanpy
ENV_PYTHON = {env: Path(sys.prefix).parent / env / "bin" / "python" for env in (TACCO_ENV, BIN2CELL_ENV)}
RESULT_KEY = "cell_type_tacco"


# =============================================================================
# Cell-level expression matrices
# =============================================================================

def convert_xenium(sample_dir):
    """Xenium outs -> cell-level AnnData: raw counts in X and layers['counts'], cells.parquet as obs, centroids as obsm['spatial'] (um)."""
    sample_dir = Path(sample_dir)
    adata = sc.read_10x_h5(str(sample_dir / "cell_feature_matrix.h5"))       # 'Gene Expression' features only
    adata.var_names_make_unique()
    cells = pd.read_parquet(sample_dir / "cells.parquet")
    cells.index = cells["cell_id"].astype(str)
    assert list(adata.obs_names) == cells.index[: adata.n_obs].tolist(), "cell order differs between the matrix and cells.parquet"
    adata.obs = cells.loc[adata.obs_names].copy()
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values.astype(np.float32)
    adata.layers["counts"] = adata.X.copy()
    print(f"{sample_dir.name}: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


def load_hd_segmented(segmented_dir):
    """Space Ranger segmented outputs -> cell-level AnnData with polygon centroids (full-resolution pixels) as obsm['spatial']."""
    segmented_dir = Path(segmented_dir)
    adata = sc.read_10x_h5(str(segmented_dir / "filtered_feature_cell_matrix.h5"))
    adata.var_names_make_unique()
    with open(segmented_dir / "cell_segmentations.geojson") as f:
        geo = json.load(f)
    centroids = {feat["properties"]["cell_id"]: np.array(feat["geometry"]["coordinates"][0]).mean(axis=0) for feat in geo["features"]}
    spatial = np.zeros((adata.n_obs, 2), dtype=np.float32)
    matched = 0
    for i, name in enumerate(adata.obs_names):                  # cellid_000000001-1 -> 1
        cid = int(name.split("_")[1].split("-")[0])
        if cid in centroids:
            spatial[i] = centroids[cid]
            matched += 1
    adata.obsm["spatial"] = spatial
    adata.layers["counts"] = adata.X.copy()
    print(f"{segmented_dir.parent.name}: {adata.n_obs:,} cells x {adata.n_vars:,} genes, {matched:,} centroids matched")
    return adata


# ---- Visium HD without segmentation: CellViT nuclei + bin2cell ----------------------------------------------------

def cellvit_label_image(cells, image_shape, chunk_height=4096):
    """Rasterise the CellViT nucleus contours into a sparse (H, W) label image (1-based cell ids)."""
    import cv2
    H, W = image_shape
    rows_all, cols_all, data_all = [], [], []
    for y_start in range(0, H, chunk_height):
        y_end = min(y_start + chunk_height, H)
        chunk = np.zeros((y_end - y_start, W), dtype=np.int32)
        for cell_id_0, cell in enumerate(cells):
            y_min, y_max = cell["bbox"][0][0], cell["bbox"][1][0]          # bbox = [[y_min, x_min], [y_max, x_max]]
            if y_max < y_start or y_min >= y_end:
                continue
            contour = np.array([[pt[0], pt[1] - y_start] for pt in cell["contour"]], dtype=np.int32)
            cv2.fillPoly(chunk, [contour], color=cell_id_0 + 1)
        ys, xs = np.nonzero(chunk)
        if len(ys):
            rows_all.append(ys + y_start)
            cols_all.append(xs)
            data_all.append(chunk[ys, xs])
    labels = scipy.sparse.csr_matrix((np.concatenate(data_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
                                     shape=(H, W), dtype=np.int32)
    print(f"label image {H} x {W}: {labels.nnz:,} nucleus pixels, {len(cells):,} nuclei")
    return labels


def insert_labels(adata, labels, key="labels_cellvit"):
    """Give every bin the label of the nucleus pixel under its centre (bins are placed by obsm['spatial'], full-res pixels)."""
    coords = adata.obsm["spatial"]
    rows, cols = np.round(coords[:, 1]).astype(int), np.round(coords[:, 0]).astype(int)
    adata.obs[key] = 0
    inside = (rows >= 0) & (rows < labels.shape[0]) & (cols >= 0) & (cols < labels.shape[1])
    adata.obs.loc[inside, key] = np.asarray(labels[rows[inside], cols[inside]]).flatten()
    print(f"bins under a nucleus: {(adata.obs[key] > 0).sum():,} of {adata.n_obs:,}")


def expand_labels(adata, key="labels_cellvit", expanded_key="labels_cellvit_expanded", max_bin_distance=2, k=4,
                  query_batch_size=500_000, pca_batch_size=50_000):
    """bin2cell's expand_labels (algorithm 'max_bin_distance', subset_pca=True), batched to bound memory.

    Every unlabelled bin within `max_bin_distance` (array units) of a labelled bin takes the nearest label; when
    several labels are equally near, the one whose bin is closest in the PCA space of log1p expression wins.
    """
    adata.obs[expanded_key] = adata.obs[key].values.copy()
    coords = adata.obs[["array_row", "array_col"]].values
    labels = adata.obs[key].values
    object_mask = labels != 0
    reference_inds, query_inds = np.arange(adata.shape[0])[object_mask], np.arange(adata.shape[0])[~object_mask]
    if len(query_inds) == 0 or len(reference_inds) == 0:
        return
    tree = scipy.spatial.cKDTree(coords[object_mask, :])
    label_distance = np.ones((labels.max() + 1,)) * max_bin_distance
    amb_query, amb_hits, amb_is_hit, amb_calls = [], [], [], []
    for start in range(0, len(query_inds), query_batch_size):
        b_query = query_inds[start:start + query_batch_size]
        dists, hits = tree.query(x=coords[b_query, :], k=k, workers=-1)
        hits = reference_inds[hits]
        calls = labels[hits]
        dists[dists > label_distance[calls]] = 1000
        min_per_bin = np.min(dists, axis=1)[:, None]
        is_hit = (dists == min_per_bin) & (min_per_bin < 1000)
        clear = np.sum(is_hit, axis=1) == 1
        if clear.any():
            adata.obs.loc[adata.obs_names[b_query[clear]], expanded_key] = calls[clear, np.argmin(dists[clear, :], axis=1)]
        ambiguous = np.sum(is_hit, axis=1) > 1
        if ambiguous.any():
            amb_query.append(b_query[ambiguous]); amb_hits.append(hits[ambiguous]); amb_is_hit.append(is_hit[ambiguous]); amb_calls.append(calls[ambiguous])
        gc.collect()
    if amb_query:
        amb_query, amb_hits = np.concatenate(amb_query), np.concatenate(amb_hits, axis=0)
        amb_is_hit, amb_calls = np.concatenate(amb_is_hit, axis=0), np.concatenate(amb_calls, axis=0)
        for start in range(0, len(amb_query), pca_batch_size):
            q, h = amb_query[start:start + pca_batch_size], amb_hits[start:start + pca_batch_size]
            m, c = amb_is_hit[start:start + pca_batch_size], amb_calls[start:start + pca_batch_size]
            smol = np.unique(np.concatenate([h.flatten(), q]))
            X = adata.X[smol, :]
            X = X.toarray() if scipy.sparse.issparse(X) else X
            pca = sc.pp.pca(np.log1p(X))
            to_pca = np.zeros(adata.shape[0], dtype=np.int64)
            to_pca[smol] = np.arange(len(smol))
            eucl = np.linalg.norm(pca[to_pca[h], :] - pca[to_pca[q], :][:, None, :], axis=2)
            eucl[~m] = 1000
            adata.obs.loc[adata.obs_names[q], expanded_key] = c[np.arange(len(c)), np.argmin(eucl, axis=1)]
            gc.collect()
    print(f"bins after expansion: {(adata.obs[expanded_key] > 0).sum():,} (was {(adata.obs[key] > 0).sum():,})")


def bins_to_cells(adata, key="labels_cellvit_expanded", gene_batch_size=2000):
    """Sum the bins of every label into one cell profile; cell position = mean position of its bins (bin2cell's bin_to_cell)."""
    sub = adata[adata.obs[key] != 0].copy()
    labels = sub.obs[key].astype(int).values
    unique = np.sort(np.unique(labels))
    idx = {lab: i for i, lab in enumerate(unique)}
    cell_to_bin = scipy.sparse.csr_matrix((np.ones(sub.n_obs, dtype=np.float32), ([idx[l] for l in labels], np.arange(sub.n_obs))),
                                          shape=(len(unique), sub.n_obs))
    blocks = []
    for g in range(0, sub.n_vars, gene_batch_size):
        X = sub.X[:, g:g + gene_batch_size]
        blocks.append(cell_to_bin.dot(X if scipy.sparse.issparse(X) else scipy.sparse.csr_matrix(X)).tocsc())
    cells = ad.AnnData(scipy.sparse.hstack(blocks, format="csr"), var=sub.var.copy())
    cells.obs_names = [str(l) for l in unique]
    cells.obs["object_id"] = unique.tolist()
    bin_count = np.asarray(cell_to_bin.sum(axis=1)).flatten()
    mean = scipy.sparse.diags(1 / bin_count).dot(cell_to_bin)
    cells.obs["bin_count"] = bin_count.astype(int)
    cells.obs["array_row"], cells.obs["array_col"] = mean.dot(sub.obs["array_row"].values), mean.dot(sub.obs["array_col"].values)
    cells.obsm["spatial"] = mean.dot(sub.obsm["spatial"])
    if "spatial" in sub.uns:
        cells.uns["spatial"] = sub.uns["spatial"]
        library = list(cells.uns["spatial"].keys())[0]
        cells.uns["spatial"][library]["scalefactors"]["spot_diameter_fullres"] *= np.sqrt(np.mean(bin_count))
    cells.layers["counts"] = cells.X.copy()
    print(f"{cells.n_obs:,} cells, {np.mean(bin_count):.1f} bins per cell on average")
    return cells


def hd_bin2cell(visium_hd_path, he_image, cellvit_dir, out_dir, bin_size="2um", max_bin_distance=2):
    """CellViT nuclei + bin2cell -> cell-level AnnData for a Visium HD sample without segmentation (writes out_dir/cells.h5ad)."""
    sys.path.insert(0, str(HERE))
    import bin2cell as b2c                       # vendored copy (needs stardist, opencv, scikit-image)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(cellvit_dir) / "cells.json") as f:
        detection = json.load(f)
    with Image.open(he_image) as img:
        width, height = img.size
    labels = cellvit_label_image(detection["cells"], (height, width))
    scipy.sparse.save_npz(out_dir / "cellvit_labels.npz", labels)
    adata = b2c.read_visium(str(Path(visium_hd_path) / "binned_outputs" / f"square_{bin_size.rstrip('um').zfill(3)}um"))
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_counts=1)
    print(f"Visium HD: {adata.n_obs:,} bins x {adata.n_vars:,} genes")
    insert_labels(adata, labels)
    expand_labels(adata, max_bin_distance=max_bin_distance)
    cells = bins_to_cells(adata)
    cells.write_h5ad(out_dir / "cells.h5ad")
    return out_dir / "cells.h5ad"


# =============================================================================
# TACCO annotation
# =============================================================================

def run_tacco(cells, atlas_h5ad, labels_key, max_annotation=1, seed=0):
    """Annotate cell-level counts against the atlas classes in obs[labels_key]; adds obs['cell_type_tacco'] and obs['cell_type_confidence'].

    max_annotation: classes TACCO may assign per cell (1 = one hard label; the weights are then one-hot).
    seed: numpy seed set before TACCO, as in the original runs (TACCO's optimal-transport annotation is deterministic).
    Cells without any count in a gene shared with the atlas receive no class and stay NaN.
    """
    import tacco as tc
    atlas = ad.read_h5ad(atlas_h5ad)
    atlas.var_names_make_unique()
    cells = cells.copy()
    cells.X = cells.layers["counts"].copy() if "counts" in cells.layers else cells.X
    shared = len(set(cells.var_names) & set(atlas.var_names))
    print(f"cells {cells.shape}, atlas {atlas.shape}, shared genes {shared}")
    np.random.seed(seed)
    tc.preprocessing.construct_reference_profiles(atlas, annotation_key=labels_key)
    if "reference" in atlas.varm:
        atlas.varm["reference"] = np.array(atlas.varm["reference"])
    tc.tl.annotate(cells, atlas, labels_key, result_key=RESULT_KEY, max_annotation=max_annotation)
    weights = cells.obsm[RESULT_KEY]                                   # one column per atlas class; all NaN when TACCO assigns none
    annotated = weights.notna().any(axis=1)
    cells.obs[RESULT_KEY] = pd.Series(np.nan, index=cells.obs_names, dtype=object)
    cells.obs.loc[annotated, RESULT_KEY] = weights.loc[annotated].idxmax(axis=1)
    cells.obs["cell_type_confidence"] = weights.max(axis=1)
    print(cells.obs[RESULT_KEY].value_counts(dropna=False).to_string())
    return cells


def manual_correction(annotated_h5ad, mapping):
    """Manual correction of TACCO labels for classes that cannot occur in a sample: {TACCO class: corrected class}.

    Applied to obs['cell_type_tacco'] in place; the uncorrected TACCO call is kept in obs['cell_type_tacco_raw']
    (re-running the correction starts again from that column).
    """
    cells = ad.read_h5ad(annotated_h5ad)
    raw = cells.obs[RESULT_KEY + "_raw"] if RESULT_KEY + "_raw" in cells.obs else cells.obs[RESULT_KEY].copy()
    cells.obs[RESULT_KEY + "_raw"] = raw
    cells.obs[RESULT_KEY] = raw.replace(mapping)
    for src, dst in mapping.items():
        print(f"{Path(annotated_h5ad).stem}: {int((raw == src).sum()):,} {src} cells corrected to {dst}")
    cells.write_h5ad(annotated_h5ad)
    return Path(annotated_h5ad)


# =============================================================================
# Notebook entry points (run the command line in the env that has the tool)
# =============================================================================

def _run(env, args):
    python = ENV_PYTHON[env]
    if not python.exists():
        raise FileNotFoundError(f"conda env '{env}' not found at {python.parents[1]} (expected next to the notebook's env "
                                f"{sys.prefix}); create it or edit ENV_PYTHON at the top of annotation.py")
    cmd = [str(python), str(HERE / "annotation.py")] + [str(a) for a in args]
    print("$ " + " ".join(cmd), flush=True)
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}   # the notebook's inline backend is unknown to other envs
    proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**env, "MPLBACKEND": "Agg"})
    for line in proc.stdout:
        print(line.rstrip().split("\r")[-1], flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"annotation.py {args[0]} failed with exit code {proc.returncode}")


def _tacco_args(labels_key, max_annotation, seed):
    return ["--labels_key", labels_key, "--max_annotation", max_annotation, "--seed", seed]


def annotate_xenium(sample_dir, atlas_h5ad, out, labels_key, max_annotation=1, seed=0):
    """Xenium outs -> annotated cell-level h5ad at `out` (reused if it exists)."""
    if Path(out).exists():
        print(f"{Path(sample_dir).name}: using existing {out}")
        return Path(out)
    _run(TACCO_ENV, ["xenium", "--sample_dir", sample_dir, "--atlas", atlas_h5ad, "--out", out]
         + _tacco_args(labels_key, max_annotation, seed))
    return Path(out)


def annotate_hd_segmented(segmented_dir, atlas_h5ad, out, labels_key, max_annotation=1, seed=0):
    """Space Ranger segmented outputs -> annotated cell-level h5ad at `out` (reused if it exists)."""
    if Path(out).exists():
        print(f"using existing {out}")
        return Path(out)
    _run(TACCO_ENV, ["hd-segmented", "--segmented_dir", segmented_dir, "--atlas", atlas_h5ad, "--out", out]
         + _tacco_args(labels_key, max_annotation, seed))
    return Path(out)


def annotate_hd_bin2cell(visium_hd_path, he_image, cellvit_dir, atlas_h5ad, out_dir, labels_key, bin_size="2um",
                         max_bin_distance=2, max_annotation=1, seed=0):
    """Visium HD without segmentation: bin2cell cells (bin2cell env) then TACCO (TACCO env); returns out_dir/cells_annotated.h5ad."""
    out_dir = Path(out_dir)
    if (out_dir / "cells_annotated.h5ad").exists():
        print(f"using existing {out_dir / 'cells_annotated.h5ad'}")
        return out_dir / "cells_annotated.h5ad"
    if not (out_dir / "cells.h5ad").exists():
        _run(BIN2CELL_ENV, ["hd-bin2cell", "--visium_hd_path", visium_hd_path, "--he_image", he_image, "--cellvit_dir", cellvit_dir,
                            "--out_dir", out_dir, "--bin_size", bin_size, "--max_bin_distance", max_bin_distance])
    _run(TACCO_ENV, ["tacco", "--cells", out_dir / "cells.h5ad", "--atlas", atlas_h5ad, "--out", out_dir / "cells_annotated.h5ad"]
         + _tacco_args(labels_key, max_annotation, seed))
    return out_dir / "cells_annotated.h5ad"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("xenium", "hd-segmented", "tacco"):
        p = sub.add_parser(name)
        if name == "xenium":
            p.add_argument("--sample_dir", required=True, help="Xenium outs: cell_feature_matrix.h5 + cells.parquet")
        elif name == "hd-segmented":
            p.add_argument("--segmented_dir", required=True, help="Space Ranger segmented_outputs folder")
        else:
            p.add_argument("--cells", required=True, help="cell-level h5ad with raw counts (e.g. from hd-bin2cell)")
        p.add_argument("--atlas", required=True, help="single-cell atlas h5ad (raw counts)")
        p.add_argument("--labels_key", required=True, help="obs column of the atlas with the cell-type classes")
        p.add_argument("--out", required=True, help="annotated cell-level h5ad to write")
        p.add_argument("--max_annotation", type=int, default=1, help="classes TACCO may assign per cell (default 1: one hard label)")
        p.add_argument("--seed", type=int, default=0, help="numpy seed set before TACCO (default 0)")
    p = sub.add_parser("hd-bin2cell")
    p.add_argument("--visium_hd_path", required=True, help="Space Ranger outs with binned_outputs/square_002um")
    p.add_argument("--he_image", required=True, help="full-resolution H&E image the CellViT nuclei refer to")
    p.add_argument("--cellvit_dir", required=True, help="CellViT cell_detection folder (cells.json)")
    p.add_argument("--out_dir", required=True, help="folder for cellvit_labels.npz and cells.h5ad")
    p.add_argument("--bin_size", default="2um", help="Visium HD bin size, binned_outputs/square_<bin size> (default 2um)")
    p.add_argument("--max_bin_distance", type=int, default=2, help="bin2cell expansion radius in bins (default 2)")
    args = parser.parse_args()

    if args.command == "hd-bin2cell":
        hd_bin2cell(args.visium_hd_path, args.he_image, args.cellvit_dir, args.out_dir, args.bin_size, args.max_bin_distance)
        return
    if args.command == "xenium":
        cells = convert_xenium(args.sample_dir)
    elif args.command == "hd-segmented":
        cells = load_hd_segmented(args.segmented_dir)
    else:
        cells = ad.read_h5ad(args.cells)
    cells = run_tacco(cells, args.atlas, args.labels_key, args.max_annotation, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cells.write_h5ad(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

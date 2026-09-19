"""Step 1 of CRIST: aggregate an annotated high-resolution ST sample into pseudo-spots.

A high-resolution sample (10x Xenium) is a directory holding
  transcripts.parquet   one row per detected transcript: x_location, y_location (um), feature_name
  cell_boundaries.csv   segmentation-polygon vertices per cell: cell_id, vertex_x, vertex_y (um)
plus the annotated cell-level AnnData written by step 0 (annotation.py): obsm['spatial'] (um) and
obs['cell_type_tacco'] (atlas-based label from TACCO; NaN for cells TACCO could not label).

The tissue is covered with a regular hexagonal lattice of Visium-sized spots (55 um
centre-to-centre pitch, i.e. inradius 27.5 um). Every transcript and every segmented cell
is assigned to the nearest lattice centre (Voronoi assignment). The result is a
pseudo-Visium AnnData:
  X                     spot x feature transcript counts (int32); features are all Xenium
                        feature names, i.e. the panel genes plus the control codewords
  obs['n_<cell type>']  number of labelled cells of each type in the spot (exact composition label)
  obs['total_counts'], obs['n_genes'], obs['total_cells'], obs['in_tissue']
  obsm['spatial']       spot centres (um)
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

SPOT_DIAMETER_UM = 55.0                # Visium capture-spot diameter (lattice inradius 27.5 um)
HEX_PITCH_UM = SPOT_DIAMETER_UM        # centre-to-centre distance: spots are tangent, no gaps
MIN_COUNTS_IN_TISSUE = 10              # keep a lattice spot if it has >= this many transcripts or >= 1 labelled cell
CELL_TYPE_KEY = "cell_type_tacco"      # obs column of the step-0 annotation with the atlas-based cell label
CONTROL_PREFIXES = ("NegControl", "BLANK", "Unassigned", "Deprecated")   # Xenium control codewords


def create_hex_grid(x_min, x_max, y_min, y_max, pitch=HEX_PITCH_UM):
    """Hexagonal lattice of spot centres covering the box [x_min, x_max] x [y_min, y_max].

    Rows are pitch * sqrt(3) / 2 apart and every second row is shifted by half a pitch.
    Returns an (n_spots, 2) float64 array of (x, y) centres in the unit of the inputs (um).
    """
    row_spacing = pitch * np.sqrt(3) / 2
    xs = np.arange(x_min, x_max + pitch, pitch)
    ys = np.arange(y_min, y_max + row_spacing, row_spacing)
    centres = []
    for row, y in enumerate(ys):
        x_offset = pitch / 2 if row % 2 == 1 else 0.0
        for x in xs:
            cx = x + x_offset
            if cx <= x_max + pitch:
                centres.append([cx, y])
    return np.asarray(centres, dtype=np.float64)


def load_high_res_sample(sample_dir, annotation):
    """Load the per-cell labels (step-0 AnnData at `annotation`) and the segmented-cell positions of one Xenium sample.

    Returns a dict with
      'cells'        DataFrame indexed by cell_id: x, y (um, mean of the boundary-polygon vertices)
                     and cell_type (NaN when the cell has no label)
      'cell_xy_bbox' (x_min, x_max, y_min, y_max) of the annotated cells, used to size the lattice
      'transcripts'  path to transcripts.parquet
    """
    sample_dir = Path(sample_dir)
    annot = ad.read_h5ad(annotation, backed="r")
    cell_type = annot.obs[CELL_TYPE_KEY].astype(object)
    xy = np.asarray(annot.obsm["spatial"], dtype=np.float64)[:, :2]
    annot.file.close()

    vertices = pd.read_csv(sample_dir / "cell_boundaries.csv", usecols=["cell_id", "vertex_x", "vertex_y"])
    cells = vertices.groupby("cell_id")[["vertex_x", "vertex_y"]].mean()
    cells.columns = ["x", "y"]
    cells.index = cells.index.astype(str)
    cells["cell_type"] = cell_type.reindex(cells.index)    # cells absent from the annotation get NaN
    if cells["cell_type"].notna().sum() == 0:
        raise ValueError(f"{sample_dir.name}: no cell of cell_boundaries.csv matches a labelled cell "
                         f"in {annotation} (different cell_id formats?)")
    return {
        "cells": cells,
        "cell_xy_bbox": (xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()),
        "transcripts": sample_dir / "transcripts.parquet",
    }


def aggregate_transcripts(transcripts_path, spot_xy, batch_size=1_000_000):
    """Count transcripts per (spot, feature): every transcript goes to its nearest spot centre.

    Returns (int32 array of shape (n_spots, n_features), sorted feature names).
    """
    parquet = pq.ParquetFile(transcripts_path)
    features = set()
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["feature_name"]):
        features.update(batch.column("feature_name").cast(pa.string()).unique().to_pylist())
    features = sorted(features)
    feature_to_idx = {f: i for i, f in enumerate(features)}

    tree = cKDTree(spot_xy)
    counts = np.zeros((len(spot_xy), len(features)), dtype=np.int32)
    for batch in parquet.iter_batches(batch_size=batch_size,
                                      columns=["x_location", "y_location", "feature_name"]):
        xy = np.column_stack([batch.column("x_location").to_numpy(), batch.column("y_location").to_numpy()])
        _, spot_idx = tree.query(xy)
        feat_idx = batch.column("feature_name").cast(pa.string()).to_pandas().map(feature_to_idx).to_numpy()
        np.add.at(counts, (spot_idx, feat_idx), 1)
    return counts, features


def count_cell_types(cells, spot_xy, cell_types):
    """Assign every segmented cell to its nearest spot centre and count the labels per spot.

    Unlabelled cells (NaN) are ignored. Returns a DataFrame of shape (n_spots, len(cell_types)).
    """
    unknown = set(cells["cell_type"].dropna().unique()) - set(cell_types)
    if unknown:
        raise ValueError(f"labels {sorted(unknown)} are not in cell_types")
    tree = cKDTree(spot_xy)
    _, spot_idx = tree.query(cells[["x", "y"]].to_numpy())
    type_idx = pd.Categorical(cells["cell_type"], categories=cell_types).codes   # -1 = unlabelled
    keep = type_idx >= 0
    counts = np.zeros((len(spot_xy), len(cell_types)), dtype=np.int64)
    np.add.at(counts, (spot_idx[keep], type_idx[keep]), 1)
    return pd.DataFrame(counts, columns=list(cell_types))


def build_pseudo_spots(sample_dir, annotation, cell_types):
    """Aggregate one annotated high-resolution sample into a pseudo-Visium AnnData.

    annotation: the cell-level AnnData written by step 0 for this sample.
    cell_types: ordered list of cell-type names; one obs['n_<type>'] column is written per name.
                Pass the same list for every sample so that all samples share one label space
                (types absent from a sample simply get an all-zero column).
    """
    sample_dir = Path(sample_dir)
    sample = load_high_res_sample(sample_dir, annotation)
    cells = sample["cells"]

    x_min, x_max, y_min, y_max = sample["cell_xy_bbox"]
    spot_xy = create_hex_grid(x_min, x_max, y_min, y_max)

    counts, features = aggregate_transcripts(sample["transcripts"], spot_xy)
    type_counts = count_cell_types(cells, spot_xy, cell_types)

    total_counts = counts.sum(axis=1)
    total_cells = type_counts.sum(axis=1).to_numpy()
    in_tissue = (total_counts >= MIN_COUNTS_IN_TISSUE) | (total_cells >= 1)

    adata = ad.AnnData(X=counts[in_tissue])
    adata.var_names = features
    adata.obs_names = [f"spot_{i}" for i in range(int(in_tissue.sum()))]
    adata.obsm["spatial"] = spot_xy[in_tissue]
    for ct in cell_types:
        adata.obs[f"n_{ct}"] = type_counts.loc[in_tissue, ct].to_numpy()
    adata.obs["total_counts"] = total_counts[in_tissue]
    adata.obs["n_genes"] = (counts[in_tissue] > 0).sum(axis=1)
    adata.obs["total_cells"] = total_cells[in_tissue]
    adata.obs["in_tissue"] = True
    adata.uns["pseudo_spots"] = {
        "sample": sample_dir.name,
        "spot_diameter_um": SPOT_DIAMETER_UM,
        "hex_pitch_um": HEX_PITCH_UM,
        "n_segmented_cells": int(len(cells)),
        "cell_types": list(cell_types),
    }
    return adata


def summarize(adata):
    """One-row summary of a pseudo-spot AnnData (for display in the notebook)."""
    obs = adata.obs
    return {
        "spots": adata.n_obs,
        "features": adata.n_vars,
        "control-codeword features": int(sum(v.startswith(CONTROL_PREFIXES) for v in adata.var_names)),
        "segmented cells": adata.uns["pseudo_spots"]["n_segmented_cells"],
        "labelled cells": int(obs["total_cells"].sum()),
        "labelled cells / spot (mean)": round(float(obs["total_cells"].mean()), 2),
        "spots without labelled cells": int((obs["total_cells"] == 0).sum()),
        "transcripts / spot (median)": float(obs["total_counts"].median()),
        "features / spot (median)": float(obs["n_genes"].median()),
    }

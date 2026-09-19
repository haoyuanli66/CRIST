"""Step 4 of CRIST: pseudo-labels for the target sample from Cell2location (v0.1.4, scvi-tools 1.3.0).

Same two-stage procedure as the benchmark script run_cell2location.py:
  1. reference signatures: negative-binomial regression on the single-cell atlas restricted to the
     genes of the target panel (filter_genes defaults of the Cell2location tutorial, 400 epochs)
  2. spatial mapping: Cell2location on all target spots in full batches (30,000 epochs,
     N_cells_per_location = 5, detection_alpha = 20); the posterior is summarised from 1,000 samples
     and the 5 % quantile of the cell abundance (q05_cell_abundance_w_sf) is exported.
The exported abundances are converted to proportions where they are used (adaptation, evaluation).
"""
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import cell2location
    from cell2location.models import Cell2location, RegressionModel
    from cell2location.utils.filtering import filter_genes
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)   # silence the trainer's device banner
warnings.filterwarnings("ignore", module="lightning")


def build_reference(atlas_h5ad, target_genes, labels_key):
    """Atlas cells x (atlas genes that the target panel measures), raw integer counts, sorted gene order."""
    atlas = sc.read_h5ad(atlas_h5ad)
    genes = sorted(set(atlas.var_names) & set(target_genes))
    ref = atlas[:, genes].copy()
    ref.var_names = ref.var_names.astype(str)
    ref.var_names_make_unique()
    ref.X = ref.X.astype(int)
    print(f"Reference: {ref.n_obs:,} cells x {ref.n_vars} genes, {ref.obs[labels_key].nunique()} classes in obs['{labels_key}']")
    return ref


def estimate_signatures(adata_ref, ref_dir, labels_key, hp):
    """Per-cell-type reference expression (genes x types). Cached in ref_dir/sc.h5ad."""
    ref_dir = Path(ref_dir)
    cached = ref_dir / "sc.h5ad"
    if cached.exists():
        print(f"Using existing reference signatures {cached}")
        adata_ref = sc.read_h5ad(cached)
    else:
        selected = filter_genes(adata_ref, cell_count_cutoff=hp["cell_count_cutoff"],
                                cell_percentage_cutoff2=hp["cell_percentage_cutoff2"], nonz_mean_cutoff=hp["nonz_mean_cutoff"])
        adata_ref = adata_ref[:, selected].copy()
        print(f"{adata_ref.n_vars} genes kept by filter_genes")
        RegressionModel.setup_anndata(adata=adata_ref, batch_key=None, labels_key=labels_key, categorical_covariate_keys=None)
        model = RegressionModel(adata_ref)
        model.train(max_epochs=hp["reference_epochs"], enable_progress_bar=False)
        adata_ref = model.export_posterior(adata_ref, sample_kwargs={"num_samples": hp["posterior_samples"], "batch_size": 2500})
        ref_dir.mkdir(parents=True, exist_ok=True)
        adata_ref.write_h5ad(cached)
    factors = adata_ref.uns["mod"]["factor_names"]
    inf_aver = adata_ref.varm["means_per_cluster_mu_fg"][[f"means_per_cluster_mu_fg_{f}" for f in factors]].copy()
    inf_aver.columns = factors
    return inf_aver


def prepare_st(adata_pseudo):
    """Target spots for Cell2location: raw transcript counts, mitochondrial genes set aside."""
    adata = sc.AnnData(X=np.asarray(adata_pseudo.X), obs=adata_pseudo.obs[[]].copy(), var=adata_pseudo.var[[]].copy())
    adata.var_names_make_unique()
    adata.var["MT_gene"] = [g.startswith("MT-") for g in adata.var_names]
    adata.obsm["MT"] = np.asarray(adata[:, adata.var["MT_gene"].values].X)
    return adata[:, ~adata.var["MT_gene"].values].copy()


def run_cell2location(adata_st, inf_aver, save_dir, hp, model_class=None, **model_kwargs):
    """Fit Cell2location (optionally a subclass with extra constructor arguments) and export q05 abundances.

    Returns the spots x cell types DataFrame that is also written to save_dir/deconv.csv (cached).
    """
    save_dir = Path(save_dir)
    deconv_path = save_dir / "deconv.csv"
    if deconv_path.exists():
        print(f"Using existing {deconv_path}")
        return pd.read_csv(deconv_path, index_col=0)
    shared = np.intersect1d(adata_st.var_names, inf_aver.index)
    adata = adata_st[:, shared].copy()
    signatures = inf_aver.loc[shared, :].copy()
    print(f"{adata.n_obs:,} spots x {adata.n_vars} shared genes, {signatures.shape[1]} cell types")
    Cell2location.setup_anndata(adata=adata, batch_key=None)
    model = Cell2location(adata, cell_state_df=signatures, N_cells_per_location=hp["n_cells_per_location"],
                          detection_alpha=hp["detection_alpha"], **({"model_class": model_class} if model_class else {}),
                          **model_kwargs)
    model.train(max_epochs=hp["mapping_epochs"], batch_size=None, train_size=1, enable_progress_bar=False)
    adata = model.export_posterior(adata, sample_kwargs={"num_samples": hp["posterior_samples"], "batch_size": adata.n_obs})
    result = pd.DataFrame(np.asarray(adata.obsm["q05_cell_abundance_w_sf"]), index=adata.obs_names,
                          columns=list(adata.uns["mod"]["factor_names"]))
    save_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(deconv_path)
    print(f"Saved {deconv_path}")
    return result

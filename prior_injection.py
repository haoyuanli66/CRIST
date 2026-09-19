"""Step 6 of CRIST: inject the transferred composition into Cell2location's abundance prior.

cell2location_presence_module.py (unchanged from the benchmark) subclasses Cell2location's Pyro model so
that the prior mean of the cell abundance w_sf at spot s becomes
    mu_sf + lambda * n_s * p_sf
where mu_sf is the model's own factorised prior mean, n_s its latent number of cells at the spot,
p_sf the CRIST proportion of cell type f at spot s and lambda the injection strength (presence_trust).
lambda = 0 recovers the unmodified model; everything else (signatures, detection efficiency, background,
likelihood, training settings) is identical to the pseudo-label run of step 4.
"""
import numpy as np

from cell2location_presence_module import Cell2locationWithPresencePrior
from evaluation import normalize_celltype_name
from pseudo_labels import run_cell2location


def align_prior(proportions_df, spot_ids, factor_names):
    """(spots x factors) array of CRIST proportions in Cell2location's factor order; unmatched entries are 0."""
    lookup = {normalize_celltype_name(c): c for c in proportions_df.columns}
    aligned = np.zeros((len(spot_ids), len(factor_names)), dtype=np.float32)
    rows = proportions_df.reindex(spot_ids)
    matched = 0
    for j, factor in enumerate(factor_names):
        key = normalize_celltype_name(factor)
        if key in lookup:
            aligned[:, j] = np.nan_to_num(rows[lookup[key]].to_numpy(dtype=np.float32))
            matched += 1
    print(f"Prior aligned for {matched}/{len(factor_names)} cell types, values in [{aligned.min():.3f}, {aligned.max():.3f}]")
    return aligned


def inject_prior(adata_st, inf_aver, proportions_df, save_dir, hp):
    """Cell2location with the CRIST prior (strength hp['injection_strength']); returns q05 abundances."""
    shared = np.intersect1d(adata_st.var_names, inf_aver.index)
    factor_names = list(inf_aver.loc[shared, :].columns)
    prior = align_prior(proportions_df, list(adata_st.obs_names), factor_names)
    return run_cell2location(adata_st, inf_aver, save_dir, hp, model_class=Cell2locationWithPresencePrior,
                             presence_scores=prior, presence_trust=hp["injection_strength"])

"""
Cell2location Pyro module with proportion prior.

Injects model-predicted proportions into the Gamma prior of w_sf
as an additive offset, preserving c2l's original factorization prior:

w_sf ~ Gamma(shape, rate) where:
  - Without prior:  prior_mean = w_sf_mu
  - With prior:     prior_mean = w_sf_mu + n_s_cells * proportion * trust

trust=0 recovers standard c2l. Larger trust nudges w_sf toward
the model's predicted composition without overriding the likelihood.
"""

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from scvi import REGISTRY_KEYS
from scvi.nn import one_hot

from cell2location.models._cell2location_module import (
    LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel,
)


class Cell2locationWithPresencePrior(
    LocationModelLinearDependentWMultiExperimentLocationBackgroundNormLevelGeneAlphaPyroModel
):
    """
    Cell2location model with presence scores encoded into w_sf's Gamma prior.

    Instead of pyro.factor() penalty, directly modifies the Gamma prior on w_sf
    so that its mean is shifted towards the presence scores.

    Parameters
    ----------
    presence_scores : np.ndarray, shape (n_obs, n_factors)
        Predicted presence scores from the deconvolution model.
        Negative values will be clipped to 0.
    presence_trust : float
        How much to trust the presence scores relative to the
        factorization prior (z_sr @ x_fr). Higher = stronger prior.
        Analogous to N_cells_mean_var_ratio in the original model.
    """

    def __init__(self, *args, presence_scores=None, presence_trust=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        if presence_scores is not None:
            ps = np.array(presence_scores, dtype=np.float32)
            ps = np.clip(ps, 0.0, None)
            self.register_buffer("presence_scores", torch.tensor(ps))
            self.has_presence = True
        else:
            self.has_presence = False

        self.register_buffer("presence_trust", torch.tensor(float(presence_trust)))

    def forward(self, x_data, idx, batch_index):
        """
        Full forward pass. Identical to parent except w_sf's Gamma prior
        mean is shifted by presence scores.
        """
        obs2sample = one_hot(batch_index, self.n_batch)
        obs_plate = self.create_plates(x_data, idx, batch_index)

        # =====================Gene expression level scaling m_g======================= #
        m_g_mean = pyro.sample(
            "m_g_mean",
            dist.Gamma(
                self.m_g_mu_mean_var_ratio_hyp * self.m_g_mu_hyp,
                self.m_g_mu_mean_var_ratio_hyp,
            )
            .expand([1, 1])
            .to_event(2),
        )

        m_g_alpha_e_inv = pyro.sample(
            "m_g_alpha_e_inv",
            dist.Exponential(self.m_g_alpha_hyp_mean).expand([1, 1]).to_event(2),
        )
        m_g_alpha_e = self.ones / m_g_alpha_e_inv.pow(2)

        m_g = pyro.sample(
            "m_g",
            dist.Gamma(m_g_alpha_e, m_g_alpha_e / m_g_mean).expand([1, self.n_vars]).to_event(2),
        )

        # =====================Cell abundances w_sf======================= #
        with obs_plate as ind:
            k = "n_s_cells_per_location"
            n_s_cells_per_location = pyro.sample(
                k,
                dist.Gamma(
                    self.N_cells_per_location * self.N_cells_mean_var_ratio,
                    self.N_cells_mean_var_ratio,
                ),
            )
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=n_s_cells_per_location,
                )

            k = "b_s_groups_per_location"
            b_s_groups_per_location = pyro.sample(
                k,
                dist.Gamma(self.B_groups_per_location, self.ones),
            )
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=b_s_groups_per_location,
                )

        # cell group loadings
        shape = self.ones_1_n_groups * b_s_groups_per_location / self.n_groups_tensor
        rate = self.ones_1_n_groups / (n_s_cells_per_location / b_s_groups_per_location)
        with obs_plate as ind:
            k = "z_sr_groups_factors"
            z_sr_groups_factors = pyro.sample(
                k,
                dist.Gamma(shape, rate),
            )
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=z_sr_groups_factors,
                )

        k_r_factors_per_groups = pyro.sample(
            "k_r_factors_per_groups",
            dist.Gamma(self.factors_per_groups, self.ones).expand([self.n_groups, 1]).to_event(2),
        )

        c2f_shape = k_r_factors_per_groups / self.n_factors_tensor

        x_fr_group2fact = pyro.sample(
            "x_fr_group2fact",
            dist.Gamma(c2f_shape, k_r_factors_per_groups).expand([self.n_groups, self.n_factors]).to_event(2),
        )

        with obs_plate as ind:
            # Original factorization prior mean for w_sf
            w_sf_mu = z_sr_groups_factors @ x_fr_group2fact

            # =================== PRESENCE PRIOR (MODIFY GAMMA PARAMS) =================== #
            if self.has_presence:
                pres_batch = self.presence_scores[ind]  # (batch, n_factors)
                # Additive form: w_sf_mu + n_cells * proportion * trust
                # trust=0 → pure c2l; trust>0 → nudge toward model predictions
                # n_s_cells_per_location scales proportion to abundance space
                w_sf_prior_mean = w_sf_mu + n_s_cells_per_location * pres_batch * self.presence_trust
                w_sf_prior_mean = w_sf_prior_mean + 1e-10  # ensure positive
            else:
                w_sf_prior_mean = w_sf_mu
            # ============================================================================= #

            k = "w_sf"
            w_sf = pyro.sample(
                k,
                dist.Gamma(
                    w_sf_prior_mean * self.w_sf_mean_var_ratio_tensor,
                    self.w_sf_mean_var_ratio_tensor,
                ),
            )
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=w_sf,
                )

        # =====================Location-specific detection efficiency ======================= #
        detection_mean_y_e = pyro.sample(
            "detection_mean_y_e",
            dist.Gamma(
                self.ones * self.detection_mean_hyp_prior_alpha,
                self.ones * self.detection_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )
        detection_hyp_prior_alpha = pyro.deterministic(
            "detection_hyp_prior_alpha",
            self.ones_n_batch_1 * self.detection_hyp_prior_alpha,
        )

        beta = (obs2sample @ detection_hyp_prior_alpha) / (obs2sample @ detection_mean_y_e)
        with obs_plate:
            k = "detection_y_s"
            detection_y_s = pyro.sample(
                k,
                dist.Gamma(obs2sample @ detection_hyp_prior_alpha, beta),
            )
            if (
                self.training_wo_observed
                and not self.training_wo_initial
                and getattr(self, f"init_val_{k}", None) is not None
            ):
                pyro.sample(
                    k + "_initial",
                    dist.Gamma(
                        self.init_alpha_tt,
                        self.init_alpha_tt / getattr(self, f"init_val_{k}")[ind],
                    ),
                    obs=detection_y_s,
                )

        # =====================Gene-specific additive component ======================= #
        s_g_gene_add_alpha_hyp = pyro.sample(
            "s_g_gene_add_alpha_hyp",
            dist.Gamma(self.ones * self.alpha_g_phi_hyp_prior_alpha, self.ones * self.alpha_g_phi_hyp_prior_beta),
        )
        s_g_gene_add_mean = pyro.sample(
            "s_g_gene_add_mean",
            dist.Gamma(
                self.gene_add_mean_hyp_prior_alpha,
                self.gene_add_mean_hyp_prior_beta,
            )
            .expand([self.n_batch, 1])
            .to_event(2),
        )
        s_g_gene_add_alpha_e_inv = pyro.sample(
            "s_g_gene_add_alpha_e_inv",
            dist.Exponential(s_g_gene_add_alpha_hyp).expand([self.n_batch, 1]).to_event(2),
        )
        s_g_gene_add_alpha_e = self.ones / s_g_gene_add_alpha_e_inv.pow(2)

        s_g_gene_add = pyro.sample(
            "s_g_gene_add",
            dist.Gamma(s_g_gene_add_alpha_e, s_g_gene_add_alpha_e / s_g_gene_add_mean)
            .expand([self.n_batch, self.n_vars])
            .to_event(2),
        )

        # =====================Gene-specific overdispersion ======================= #
        alpha_g_phi_hyp = pyro.sample(
            "alpha_g_phi_hyp",
            dist.Gamma(self.ones * self.alpha_g_phi_hyp_prior_alpha, self.ones * self.alpha_g_phi_hyp_prior_beta),
        )
        alpha_g_inverse = pyro.sample(
            "alpha_g_inverse",
            dist.Exponential(alpha_g_phi_hyp).expand([self.n_batch, self.n_vars]).to_event(2),
        )

        # =====================Expected expression ======================= #
        if not self.training_wo_observed:
            mu = ((w_sf @ self.cell_state) * m_g + (obs2sample @ s_g_gene_add)) * detection_y_s
            alpha = obs2sample @ (self.ones / alpha_g_inverse.pow(2))

            if self.dropout_p != 0:
                x_data = self.dropout(x_data)
            with obs_plate:
                pyro.sample(
                    "data_target",
                    dist.GammaPoisson(concentration=alpha, rate=alpha / mu),
                    obs=x_data,
                )

        # =====================Compute mRNA count from each factor in locations ======================= #
        with obs_plate:
            mRNA = w_sf * (self.cell_state * m_g).sum(-1)
            pyro.deterministic("u_sf_mRNA_factors", mRNA)

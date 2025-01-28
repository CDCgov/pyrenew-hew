import jax.numpy as jnp
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from numpyro.infer.reparam import LocScaleReparam
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

i0_first_obs_n_rv = DistributionalVariable(
    "i0_first_obs_n_rv",
    dist.Beta(1, 10),
)

initialization_rate_rv = DistributionalVariable(
    "rate", dist.Normal(0, 0.01), reparam=LocScaleReparam(0)
)

r_logmean = jnp.log(1.2)
r_logsd = jnp.log(jnp.sqrt(2))

log_r_mu_intercept_rv = DistributionalVariable(
    "log_r_mu_intercept_rv", dist.Normal(r_logmean, r_logsd)
)

eta_sd_rv = DistributionalVariable("eta_sd", dist.TruncatedNormal(0.15, 0.05, low=0))

autoreg_rt_rv = DistributionalVariable("autoreg_rt", dist.Beta(2, 40))


inf_feedback_strength_rv = TransformedVariable(
    "inf_feedback",
    DistributionalVariable(
        "inf_feedback_raw",
        dist.LogNormal(jnp.log(50), jnp.log(1.5)),
    ),
    transforms=transformation.AffineTransform(loc=0, scale=-1),
)
# Could be reparameterized?

# low confidence logit-Normal
p_ed_visit_mean_rv = DistributionalVariable(
    "p_ed_visit_mean",
    dist.Normal(
        transformation.SigmoidTransform().inv(0.005),
        0.3,
    ),
)  # logit scale

# low confidence logit-Normal with same mode as IEDR
ihr_rv = TransformedVariable(
    "ihr",
    DistributionalVariable(
        "logit_ihr",
        dist.Normal(
            transformation.SigmoidTransform().inv(0.005),
            0.3,
        ),
    ),
    transforms=transformation.SigmoidTransform(),
)


p_ed_visit_w_sd_rv = DistributionalVariable(
    "p_ed_visit_w_sd_sd", dist.TruncatedNormal(0, 0.01, low=0)
)


autoreg_p_ed_visit_rv = DistributionalVariable(
    "autoreg_p_ed_visit_rv", dist.Beta(1, 100)
)

ed_visit_wday_effect_rv = TransformedVariable(
    "ed_visit_wday_effect",
    DistributionalVariable(
        "ed_visit_wday_effect_raw",
        dist.Dirichlet(jnp.array([5, 5, 5, 5, 5, 5, 5])),
    ),
    transformation.AffineTransform(loc=0, scale=7),
)

# low confidence with a mode at equivalence and
# plausiblity of 2x or 1/2 the rate
ihr_rel_iedr_rv = DistributionalVariable(
    "ihr_rel_iedr", dist.LogNormal(0, jnp.log(jnp.sqrt(2)))
)

# Based on looking at some historical posteriors.
ed_neg_bin_concentration_rv = DistributionalVariable(
    "ed_visit_neg_bin_concentration", dist.LogNormal(4, 1)
)

# more diffuse than ED visit, same mean
hosp_admit_neg_bin_concentration_rv = DistributionalVariable(
    "hosp_admit_neg_bin_concentration", dist.LogNormal(4, 2)
)

t_peak_rv = DistributionalVariable("t_peak", dist.TruncatedNormal(5, 1, low=0))

duration_shed_after_peak_rv = DistributionalVariable(
    "durtion_shed_after_peak", dist.TruncatedNormal(12, 3, low=0)
)

log10_genome_per_inf_ind_rv = DistributionalVariable(
    "log10_genome_per_inf_ind", dist.Normal(12, 2)
)

mode_sigma_ww_site_rv = DistributionalVariable(
    "mode_sigma_ww_site",
    dist.TruncatedNormal(1, 1, low=0),
)

sd_log_sigma_ww_site_rv = DistributionalVariable(
    "sd_log_sigma_ww_site", dist.TruncatedNormal(0, 0.693, low=0)
)

mode_sd_ww_site_rv = DistributionalVariable(
    "mode_sd_ww_site", dist.TruncatedNormal(0, 0.25, low=0)
)

# model constants related to wastewater obs process
ww_ml_produced_per_day = 227000
max_shed_interval = 26

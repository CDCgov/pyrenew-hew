import jax.numpy as jnp
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from numpyro.infer.reparam import LocScaleReparam
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

# many of these should probably be different depending
# on if we are modeling flu
# or covid

i0_first_obs_n_rv = DistributionalVariable(
    "i0_first_obs_n_rv",
    dist.Beta(1, 10),
)

initialization_rate_rv = DistributionalVariable(
    "rate", dist.Normal(0, 0.01), reparam=LocScaleReparam(0)
)

r_logmean = jnp.log(1)
r_logsd = jnp.log(jnp.sqrt(2))

log_r_mu_intercept_rv = DistributionalVariable(
    "log_r_mu_intercept_rv", dist.Normal(r_logmean, r_logsd)
)

eta_sd_rv = DistributionalVariable(
    "eta_sd", dist.TruncatedNormal(0.04, 0.02, low=0)
)

autoreg_rt_rv = DistributionalVariable("autoreg_rt", dist.Beta(2, 40))


inf_feedback_strength_rv = TransformedVariable(
    "inf_feedback",
    DistributionalVariable(
        "inf_feedback_raw",
        dist.LogNormal(jnp.log(50), jnp.log(2)),
    ),
    transforms=transformation.AffineTransform(loc=0, scale=-1),
)
# Could be reparameterized?

# Note: multiplied by 1/2 from hosp model
# this actually represents ed visits
p_hosp_mean_rv = DistributionalVariable(
    "p_hosp_mean",
    dist.Normal(
        transformation.SigmoidTransform().inv(0.005),
        0.3,
    ),
)  # logit scale


p_hosp_w_sd_rv = DistributionalVariable(
    "p_hosp_w_sd_sd", dist.TruncatedNormal(0, 0.01, low=0)
)


autoreg_p_hosp_rv = DistributionalVariable("autoreg_p_hosp", dist.Beta(1, 100))

hosp_wday_effect_rv = TransformedVariable(
    "hosp_wday_effect",
    DistributionalVariable(
        "hosp_wday_effect_raw",
        dist.Dirichlet(jnp.array([5, 5, 5, 5, 5, 5, 5])),
    ),
    transformation.AffineTransform(loc=0, scale=7),
)

# Based on looking at some historical posteriors.
phi_rv = DistributionalVariable("phi", dist.LogNormal(6, 1))

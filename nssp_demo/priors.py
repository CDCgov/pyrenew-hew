import jax.numpy as jnp
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

from pyrenew_hew.utils import convert_to_logmean_log_sd

# Note: this could be off by an order of magnitude because it has not been
# updates from the regular hospitalization model.
i0_first_obs_n_rv = DistributionalVariable(
    "i0_first_obs_n_rv",
    dist.Beta(1.0015, 5.9985),
)

initialization_rate_rv = DistributionalVariable(
    "rate",
    dist.TruncatedNormal(
        loc=0,
        scale=0.01,
        low=-1,
        high=1,
    ),
)
# could reasonably switch to non-Truncated

r_logmean, r_logsd = convert_to_logmean_log_sd(1, 1)
log_r_mu_intercept_rv = DistributionalVariable(
    "log_r_mu_intercept_rv", dist.Normal(r_logmean, r_logsd)
)

eta_sd_rv = DistributionalVariable(
    "eta_sd", dist.TruncatedNormal(0, 0.01, low=0)
)

autoreg_rt_rv = DistributionalVariable("autoreg_rt", dist.Beta(2, 40))


inf_feedback_strength_rv = TransformedVariable(
    "inf_feedback",
    DistributionalVariable(
        "inf_feedback_raw",
        dist.LogNormal(6.37408, 0.4),
    ),
    transforms=transformation.AffineTransform(loc=0, scale=-1),
)
# Could be reparameterized?

p_hosp_mean_rv = DistributionalVariable(
    "p_hosp_mean",
    dist.Normal(
        transformation.SigmoidTransform().inv(0.01),
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
# depends on the state
# inf_to_hosp_rv

phi_rv = TransformedVariable(
    "phi",
    DistributionalVariable(
        "inv_sqrt_phi",
        dist.TruncatedNormal(
            loc=0.1,
            scale=0.1414214,
            low=1 / jnp.sqrt(5000),
        ),
    ),
    transforms=transformation.PowerTransform(-2),
)


uot = 55

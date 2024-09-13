import jax.numpy as jnp
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from pyrenew.deterministic import DeterministicVariable
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

from pyrenew_covid_wastewater.utils import convert_to_logmean_log_sd

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

# depends on the state
# generation_interval_pmf_rv = DeterministicVariable(


infection_feedback_pmf_rv = DeterministicVariable(
    "infection_feedback_pmf",
    jnp.array(
        [
            0.161701189933765,
            0.320525743089203,
            0.242198071982593,
            0.134825252524032,
            0.0689141939998525,
            0.0346219683116734,
            0.017497710736154,
            0.00908172017279556,
            0.00483656086299504,
            0.00260732346885217,
            0.00143298046642562,
            0.00082002579123121,
            0.0004729600977183,
            0.000284420637980485,
            0.000179877924728358,
        ]
    ),
)

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

inf_to_hosp_rv = DeterministicVariable(
    "inf_to_hosp",
    jnp.array(
        [
            0.0,
            0.00469384736487552,
            0.0145200073436112,
            0.0278627741704387,
            0.0423656492135518,
            0.0558071445014868,
            0.0665713169684116,
            0.0737925805176124,
            0.0772854627892072,
            0.0773666390616176,
            0.0746515449009948,
            0.0698761436052596,
            0.0637663813017696,
            0.0569581929821651,
            0.0499600186601535,
            0.0431457477049282,
            0.0367662806214046,
            0.0309702535668237,
            0.0258273785539499,
            0.0213504646948306,
            0.0175141661880584,
            0.0142698211023571,
            0.0115565159519833,
            0.00930888979824423,
            0.00746229206759214,
            0.00595605679409682,
            0.00473519993107751,
            0.00375117728281842,
            0.00296198928038098,
            0.00233187862772459,
            0.00183079868293457,
            0.00143377454057296,
            0.00107076258525208,
            0.000773006742366448,
            0.000539573690886396,
            0.000364177599116743,
            0.000237727628685579,
            0.000150157714457011,
            9.18283319498657e-05,
            5.44079947589853e-05,
            3.12548818921464e-05,
            1.74202619730274e-05,
            9.42698047424712e-06,
            4.95614149002087e-06,
            2.53275674485913e-06,
            1.25854819834554e-06,
            6.08116579596933e-07,
            2.85572858589747e-07,
            1.30129404249734e-07,
            5.73280599448306e-08,
            2.4219376577964e-08,
            9.6316861194457e-09,
            3.43804936850951e-09,
            9.34806280366887e-10,
            0.0,
        ]
    ),
)


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


uot = len(inf_to_hosp_rv())

# state_pop
# depends on state

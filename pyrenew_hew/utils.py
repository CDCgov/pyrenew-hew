import jax.numpy as jnp


def convert_to_logmean_log_sd(mean, sd):
    logmean = jnp.log(
        jnp.power(mean, 2) / jnp.sqrt(jnp.power(sd, 2) + jnp.power(mean, 2))
    )
    logsd = jnp.sqrt(jnp.log(1 + (jnp.power(sd, 2) / jnp.power(mean, 2))))
    return logmean, logsd


def get_vl_trajectory(tpeak, viral_peak, duration_shedding_after_peak, n):
    growth = viral_peak / tpeak
    wane = viral_peak / duration_shedding_after_peak
    t = jnp.arange(n)
    s = 10 ** jnp.where(
        t <= tpeak, growth * t, viral_peak + wane * (tpeak - t)
    )
    return s / jnp.sum(s)

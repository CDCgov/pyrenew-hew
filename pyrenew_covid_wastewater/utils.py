import jax.numpy as jnp


def convert_to_logmean_log_sd(mean, sd):
    logmean = jnp.log(
        jnp.power(mean, 2) / jnp.sqrt(jnp.power(sd, 2) + jnp.power(mean, 2))
    )
    logsd = jnp.sqrt(jnp.log(1 + (jnp.power(sd, 2) / jnp.power(mean, 2))))
    return logmean, logsd


def get_vl_trajectory(tpeak, viral_peak, duration_shedding, n):
    s = jnp.zeros(n)
    growth = viral_peak / tpeak
    wane = viral_peak / (duration_shedding - tpeak)

    t = jnp.arange(n)
    s = jnp.where(t <= tpeak, jnp.power(10, growth * t), s)

    s = jnp.where(
        t > tpeak, jnp.maximum(0, viral_peak + wane * tpeak - wane * t), s
    )
    s = jnp.where(t > tpeak, jnp.power(10, s), s)

    s = s / jnp.sum(s)
    return s

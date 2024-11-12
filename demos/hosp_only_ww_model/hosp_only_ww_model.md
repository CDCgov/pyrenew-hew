# Replicating Hospital Only Model from ww-inference-model


``` python
import jax
import numpyro
import arviz as az
import pyrenew_hew.plotting as plotting
from pyrenew_hew.hosp_only_ww_model import (
    create_hosp_only_ww_model_from_stan_data,
)

numpyro.set_host_device_count(4)
```

## Background

This tutorial provides a demonstration of our reimplementation of “Model
2” from the [`ww-inference-model`
project](https://github.com/CDCgov/ww-inference-model). The model is
described
[here](https://github.com/CDCgov/ww-inference-model/blob/main/model_definition.md).
Stan code for the model is
[here](https://github.com/CDCgov/ww-inference-model/blob/main/inst/stan/wwinference.stan).

The model we provide is designed to be fully-compatible with the
stan_data generated in the that project. We provide the stan data used
in the `wwinference`
[vignette](https://github.com/CDCgov/ww-inference-model/blob/main/vignettes/wwinference.Rmd)
in the [`ww-inference-model`
project](https://github.com/CDCgov/ww-inference-model). The data is
available in `notebooks/data/fit_hosp_only/stan_data.json`. This data
was generated by running `notebooks/wwinference.Rmd`, which replicates
the original vignette and saves the relevant data. This script also
saves the posterior samples from the model for comparison to our own
model.

## Load Data and Create Priors

We begin by loading the Stan data, converting it the correct inputs for
our model, and definitng the model.

``` python
my_hosp_only_ww_model, data_observed_disease_hospital_admissions = (
    create_hosp_only_ww_model_from_stan_data(
        "data/fit_hosp_only/stan_data.json"
    )
)
```

# Simulate from the model

We check that we can simulate from the prior predictive

``` python
n_forecast_days = 35

prior_predictive = my_hosp_only_ww_model.prior_predictive(
    n_datapoints=len(data_observed_disease_hospital_admissions) + n_forecast_days,
    numpyro_predictive_args={"num_samples": 200},
)
```

# Fit the model

Now we can fit the model to the observed data:

``` python
my_hosp_only_ww_model.run(
    num_warmup=500,
    num_samples=500,
    rng_key=jax.random.key(200),
    data_observed_disease_hospital_admissions=data_observed_disease_hospital_admissions,
    mcmc_args=dict(num_chains=4, progress_bar=False),
    nuts_args=dict(find_heuristic_step_size=True),
)
```

Create the posterior predictive and forecast:

``` python
posterior_predictive = my_hosp_only_ww_model.posterior_predictive(
    n_datapoints=len(data_observed_disease_hospital_admissions) + n_forecast_days
)
```

## Prepare for plotting

``` python
import arviz as az

idata = az.from_numpyro(
    my_hosp_only_ww_model.mcmc,
    posterior_predictive=posterior_predictive,
    prior=prior_predictive,
)
```

## Plot Predictive Distributions

``` python
plotting.plot_predictive(idata, prior=True)
```

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-prior-preditive-output-1.png"
id="plot-prior-preditive" />

``` python
plotting.plot_predictive(idata)
```

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-posterior-preditive-output-1.png"
id="plot-posterior-preditive" />

## Plot all posteriors

``` python
for key in list(idata.posterior.keys()):
    try:
        plotting.plot_posterior(idata, key)
    except Exception as e:
        print(f"An error occurred while plotting {key}: {e}")
```

    An error occurred while plotting autoreg_p_hosp: "No variable named 'autoreg_p_hosp_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting autoreg_rt: "No variable named 'autoreg_rt_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting eta_sd: "No variable named 'eta_sd_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting i0_first_obs_n_rv: "No variable named 'i0_first_obs_n_rv_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting inf_feedback_raw: "No variable named 'inf_feedback_raw_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting inv_sqrt_phi: "No variable named 'inv_sqrt_phi_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting log_r_mu_intercept_rv: "No variable named 'log_r_mu_intercept_rv_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting p_hosp_ar_init: "No variable named 'p_hosp_ar_init_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting p_hosp_mean: "No variable named 'p_hosp_mean_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting p_hosp_w_sd_sd: "No variable named 'p_hosp_w_sd_sd_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting rate: "No variable named 'rate_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"
    An error occurred while plotting rt_init_rate_of_change: "No variable named 'rt_init_rate_of_change_dim_0'. Variables on the dataset include ['chain', 'draw', 'autoreg_p_hosp', 'autoreg_rt', 'eta_sd', ..., 'rtu', 'rtu_weekly_diff_first_diff_ar_process_noise_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered_dim_0', 'rtu_weekly_diff_first_diff_ar_process_noise_decentered']"

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-all-posteriors-output-2.png"
id="plot-all-posteriors-1" />

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-all-posteriors-output-3.png"
id="plot-all-posteriors-2" />

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-all-posteriors-output-4.png"
id="plot-all-posteriors-3" />

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-all-posteriors-output-5.png"
id="plot-all-posteriors-4" />

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-all-posteriors-output-6.png"
id="plot-all-posteriors-5" />

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-all-posteriors-output-7.png"
id="plot-all-posteriors-6" />

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-all-posteriors-output-8.png"
id="plot-all-posteriors-7" />

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-all-posteriors-output-9.png"
id="plot-all-posteriors-8" />

<img
src="hosp_only_ww_model_files/figure-commonmark/plot-all-posteriors-output-10.png"
id="plot-all-posteriors-9" />

## Save for Post-Processing

``` python
idata.to_dataframe().to_csv("data/fit_hosp_only/inference_data.csv", index=False)
```
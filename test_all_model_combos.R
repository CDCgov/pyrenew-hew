library(tidyverse)
library(fs)
library(hewr)
source("~/Documents/GitHub/pyrenew-hew/hewr/R/process_state_forecast.R")
model_batch_dirs <- c(
  path(
    "/Users/damon/Documents/GitHub/pyrenew-hew/pipelines/tests/private_data",
    "covid-19_r_2024-12-21_f_2024-10-22_t_2024-12-20"
  ),
  path(
    "/Users/damon/Documents/GitHub/pyrenew-hew/pipelines/tests/private_data",
    "influenza_r_2024-12-21_f_2024-10-22_t_2024-12-20"
  )
)

model_run_dirs <- dir_ls(path(model_batch_dirs, "model_runs"))

walk(model_run_dirs, \(model_run_dir) {
  process_state_forecast(
    model_run_dir = model_run_dir,
    pyrenew_model_name = "pyrenew_h",
    timeseries_model_name = NULL
  )


  process_state_forecast(
    model_run_dir = model_run_dir,
    pyrenew_model_name = "pyrenew_e",
    timeseries_model_name = "timeseries_e"
  )

  process_state_forecast(
    model_run_dir = model_run_dir,
    pyrenew_model_name = "pyrenew_he",
    timeseries_model_name = "timeseries_e"
  )
})

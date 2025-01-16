library(tidyverse)
library(fs)
model_batch_dir <- path("/Users/damon/Downloads/Archive/covid-19_r_2024-12-21_f_2024-10-22_t_2024-12-20/model_runs/")
model_run_dir <- path(model_batch_dir, "CA")

source("~/Documents/GitHub/pyrenew-hew/hewr/R/process_state_forecast.R")

dir_copy(path(model_run_dir, "pyrenew_e"), path(model_run_dir, "pyrenew_h"))

pyrenew_model_name <- "pyrenew_h"
timeseries_model_name <- "NULL"
process_state_forecast(model_run_dir = model_run_dir,
                       pyrenew_model_name = "pyrenew_h",
                       timeseries_model_name = NULL)

pyrenew_model_name <- "pyrenew_he"
timeseries_model_name <- "timeseries_e"
process_state_forecast(model_run_dir = model_run_dir,
                       pyrenew_model_name = "pyrenew_e",
                       timeseries_model_name = "timeseries_e")

pyrenew_model_name <- "pyrenew_e"
timeseries_model_name <- "timeseries_e"
process_state_forecast(model_run_dir = model_run_dir, |>
                         pyrenew_model_name = "pyrenew_he",
                       timeseries_model_name = "timeseries_e")

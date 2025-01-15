library(tidyverse)
library(fs)
library(hewr)
library(argparser)
p <- arg_parser("Process model batch directories")
p <- add_argument(p, "super_dir", help = "Directory containing model batch directories")
argv <- parse_args(p)
super_dir <- path(argv$super_dir)

source("hewr/R/process_state_forecast.R")
model_run_dirs <- dir_ls(path(dir_ls(super_dir), "model_runs"))

walk(
  model_run_dirs,
  \(model_run_dir) {
    print("processing h data for")
    print(model_run_dir)
    process_state_forecast(
      model_run_dir = model_run_dir,
      pyrenew_model_name = "pyrenew_h",
      timeseries_model_name = NULL
    )
  }
)

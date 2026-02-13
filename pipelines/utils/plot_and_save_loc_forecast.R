script_packages <- c(
  "argparser",
  "cowplot",
  "dplyr",
  "fs",
  "glue",
  "hewr",
  "purrr",
  "tidyr",
  "stringr",
  "lubridate"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


save_forecast_figures <- function(
  model_run_dir,
  n_forecast_days,
  pyrenew_model_name = NA,
  timeseries_model_name = NA,
  model_name = NA
) {
  processed_forecast <- process_loc_forecast(
    model_run_dir = model_run_dir,
    n_forecast_days = n_forecast_days,
    pyrenew_model_name = pyrenew_model_name,
    timeseries_model_name = timeseries_model_name,
    model_name = model_name,
    save = TRUE
  )
}

p <- arg_parser("Generate forecast figures") |>
  add_argument(
    "model_run_dir",
    help = "Directory containing the model data and output.",
  ) |>
  add_argument(
    "--pyrenew-model-name",
    help = "Name of directory containing pyrenew model outputs"
  ) |>
  add_argument(
    "--timeseries-model-name",
    help = "Name of directory containing timeseries model outputs",
  ) |>
  add_argument(
    "--n-forecast-days",
    help = "Number of days to forecast"
  ) |>
  add_argument(
    "--model-name",
    help = "Name of directory with model outputs (auto-detects type)"
  )

argv <- parse_args(p)

model_run_dir <- path(argv$model_run_dir)
n_forecast_days <- as.numeric(argv$n_forecast_days)
pyrenew_model_name <- argv$pyrenew_model_name
timeseries_model_name <- argv$timeseries_model_name
model_name <- argv$model_name

save_forecast_figures(
  model_run_dir,
  n_forecast_days,
  pyrenew_model_name,
  timeseries_model_name,
  model_name
)

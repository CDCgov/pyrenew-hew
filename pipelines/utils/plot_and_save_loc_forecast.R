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
  if (all(is.na(c(pyrenew_model_name, timeseries_model_name, model_name)))) {
    stop(
      "At least one of `pyrenew_model_name`, `timeseries_model_name`, ",
      "or `model_name` must be provided."
    )
  }

  create_file_name <- function(
    model_name,
    .variable,
    resolution,
    aggregated_numerator,
    aggregated_denominator,
    y_transform
  ) {
    glue(
      "{model_name}_",
      "{.variable}_{resolution}",
      "{dplyr::if_else(vctrs::vec_equal(",
      "aggregated_numerator,TRUE, na_equal = TRUE),'_agg_num', '')}",
      "{dplyr::if_else(vctrs::vec_equal(",
      "aggregated_denominator, TRUE, na_equal = TRUE), '_agg_denom', '')}",
      "{y_transforms[y_transform]}"
    ) |>
      str_replace_all("_+", "_")
  }

  # Determine which model name to use (prioritize in order)
  final_model_name <- dplyr::case_when(
    !is.na(model_name) ~ model_name,
    !is.na(pyrenew_model_name) ~ pyrenew_model_name,
    !is.na(timeseries_model_name) ~ timeseries_model_name,
    TRUE ~ NA_character_
  )

  model_dir <- fs::path(model_run_dir, final_model_name)
  figure_dir <- fs::path(model_dir, "figures")
  data_dir <- fs::path(model_dir, "data")
  dir_create(figure_dir)

  parsed_model_run_dir <- parse_model_run_dir_path(model_run_dir)
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

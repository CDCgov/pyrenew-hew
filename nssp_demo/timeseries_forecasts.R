script_packages <- c(
  "dplyr",
  "tidyr",
  "tibble",
  "readr",
  "stringr",
  "fs",
  "fable",
  "jsonlite",
  "argparser",
  "arrow"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})

#' Fit and Forecast Time Series Data
#'
#' This function fits a combination ensemble model to the training data and
#' generates forecast samples for a specified number of days.
#'
#' @param data A data frame containing the time series data. It should have a
#' column named `data_type` to distinguish between training and other data.
#' @param n_forecast_days An integer specifying the number of days to forecast.
#' Default is 28.
#' @param n_samples An integer specifying the number of forecast samples to
#' generate. Default is 2000.
#' @param target_col A string specifying the name of the target column in the
#' data. Default is "ed_visits".
#' @param output_col A string specifying the name of the output column for the
#' forecasted values. Default is "other_ed_visits".
#
#' @return A tibble containing the forecast samples with columns for date,
#' draw number, and forecasted values.
fit_and_forecast <- function(data,
                             n_forecast_days = 28,
                             n_samples = 2000,
                             target_col = "ed_visits",
                             output_col = "other_ed_visits") {
  forecast_horizon <- glue::glue("{n_forecast_days} days")
  target_sym <- rlang::sym(target_col)
  output_sym <- rlang::sym(output_col)
  fit <-
    data |>
    filter(data_type == "train") |>
    model(
      comb_model = combination_ensemble(
        ETS(log(!!target_sym) ~ trend(method = c("N", "M", "A"))),
        ARIMA(log(!!target_sym))
      )
    )

  forecast_samples <- fit |>
    generate(h = forecast_horizon, times = n_samples) |>
    as_tibble() |>
    mutate("{output_col}" := .sim, .draw = as.integer(.rep)) |> # nolint
    select(date, .draw, !!output_sym)

  forecast_samples
}

main <- function(model_run_dir, n_forecast_days = 28, n_samples = 2000) {
  # to do: do this with json data that has dates
  data_path <- path(model_run_dir, "data", ext = "csv")

  target_and_other_data <- read_csv(
    data_path,
    col_types = cols(
      disease = col_character(),
      data_type = col_character(),
      ed_visits = col_double(),
      date = col_date()
    )
  ) |>
    mutate(disease = if_else(
      disease == disease_name_nssp,
      "Disease", disease
    )) |>
    pivot_wider(names_from = disease, values_from = ed_visits) |>
    mutate(Other = Total - Disease) |>
    select(date, ed_visits_target = Disease, ed_visits_other = Other,
        data_type) |>
    as_tsibble(index = date)

  forecast_other <- fit_and_forecast(target_and_other_data, n_forecast_days,
    n_samples, target_col = "ed_visits_other", output_col = "other_ed_visits")
  forecast_baseline <- fit_and_forecast(target_and_other_data, n_forecast_days,
    n_samples, target_col = "ed_visits_target",
    output_col = "baseline_ed_visits")

  save_path_other <- path(model_run_dir, "other_ed_visits_forecast",
    ext = "parquet")
  save_path_baseline <- path(model_run_dir, "baseline_ed_visits_forecast",
    ext = "parquet")
  write_parquet(forecast_other, save_path_other)
  write_parquet(forecast_baseline, save_path_baseline)
}


p <- arg_parser(
  "Forecast other (non-target-disease) ED visits for a given location."
) |>
  add_argument(
    "--model-run-dir",
    help = "Directory containing the model data and output.",
  ) |>
  add_argument(
    "--n-forecast-days",
    help = "Number of days to forecast.",
    default = 28L
  ) |>
  add_argument(
    "--n-samples",
    help = "Number of samples to generate.",
    default = 2000L
  )

argv <- parse_args(p)
model_run_dir <- path(argv$model_run_dir)
n_forecast_days <- argv$n_forecast_days
n_samples <- argv$n_samples

disease_name_nssp_map <- c(
  "covid-19" = "COVID-19",
  "influenza" = "Influenza"
)

base_dir <- path_dir(model_run_dir)

disease_name_raw <- base_dir |>
  path_file() |>
  str_extract("^.+(?=_r_)")

disease_name_nssp <- unname(disease_name_nssp_map[disease_name_raw])

main(model_run_dir, n_forecast_days, n_samples)

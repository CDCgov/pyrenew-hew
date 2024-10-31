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
  "arrow",
  "pak",
  "glue"
)

script_pak_packages <- c(
  "epipredict",
  "epiprocess"
)
##

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})

## Load packages from the cmu-delphi repo if required
purrr::walk(script_pak_packages, \(pkg) {
  if (pkg %in% rownames(installed.packages())) {
    suppressPackageStartupMessages(library(pkg,
      character.only = TRUE
    ))
  } else {
    suppressMessages(pak::pkg_install(glue("cmu-delphi/{pkg}@main")))
    suppressPackageStartupMessages(library(pkg,
      character.only = TRUE
    ))
  }
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
    as_tsibble(index = date) |>
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

#' Generate CDC Flat Forecast
#'
#' This function generates a CDC flat forecast for the given data and returns
#' a data frame containing the forecasted values with columns for quantile
#' levels, reference dates, and target end dates suitable for use with
#' `scoringutils`.
#'
#' @param data A data frame containing the input data.
#' @param target_col A string specifying the column name of the target variable
#' in the data. Default is "ed_visits".
#' @param output_col A string specifying the column name for the output variable
#' in the forecast. Default is "other_ed_visits".
#' @param ... Additional arguments passed to the
#' `epipredict::cdc_baseline_args_list` function.
#' @return A data frame containing the forecasted values with columns for
#' quantile levels, (forecast) dates, and target values
cdc_flat_forecast <- function(data,
                              target_col = "ed_visits_target",
                              output_col = "cdc_flat_ed_visits",
                              ...) {
  output_sym <- rlang::sym(output_col)
  opts <- cdc_baseline_args_list(...)
  # coerce data to epiprocess::epi_df format
  epi_data <- data |>
    filter(data_type == "train") |>
    mutate(geo_value = "us", time_value = date) |>
    as_epi_df()
  # fit the model
  cdc_flat_fit <- cdc_baseline_forecaster(epi_data, target_col, opts)
  # generate forecast
  cdc_flat_forecast <- cdc_flat_fit$predictions |>
    pivot_quantiles_longer(.pred_distn) |>
    mutate("{output_col}" := .pred) |> # nolint
    rename(
      quantile_level = quantile_levels, report_date = forecast_date,
      date = target_date
    ) |>
    select(date, quantile_level, !!output_sym)

  cdc_flat_forecast
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
    select(date,
      ed_visits_target = Disease, ed_visits_other = Other,
      data_type
    )
  ## Time series forecasting
  ## Fit and forecast other (non-target-disease) ED visits using a combination
  ## ensemble model
  forecast_other <- fit_and_forecast(target_and_other_data, n_forecast_days,
    n_samples,
    target_col = "ed_visits_other", output_col = "other_ed_visits"
  )
  ## Fit and forecast baseline number ED visits using a combination ensemble
  # model
  forecast_baseline <- fit_and_forecast(target_and_other_data, n_forecast_days,
    n_samples,
    target_col = "ed_visits_target",
    output_col = "baseline_ed_visits"
  )
  ## Generate CDC flat forecast for the target disease number of ED visits
  forecast_cdc_flat <- cdc_flat_forecast(target_and_other_data,
    target_col = "ed_visits_target",
    output_col = "cdc_flat_ed_visits",
    data_frequency = "1 day",
    aheads = 1:n_forecast_days
  )

  ## Save the forecasted values to parquet files
  save_path_other <- path(model_run_dir, "other_ed_visits_forecast",
    ext = "parquet"
  )
  save_path_baseline <- path(model_run_dir, "baseline_ed_visits_forecast",
    ext = "parquet"
  )
  save_path_cdc_flat <- path(model_run_dir, "cdc_flat_ed_visits_forecast",
    ext = "parquet"
  )

  write_parquet(forecast_other, save_path_other)
  write_parquet(forecast_baseline, save_path_baseline)
  write_parquet(forecast_cdc_flat, save_path_cdc_flat)
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

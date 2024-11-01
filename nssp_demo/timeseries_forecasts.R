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
  "glue",
  "epipredict",
  "epiprocess"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


to_prop_forecast <- function(forecast_disease_count,
                             forecast_other_count,
                             disease_count_col =
                               "baseline_ed_visit_count_forecast",
                             other_count_col =
                               "other_ed_visits",
                             output_col = "prop_disease_ed_visits") {
  result <- dplyr::inner_join(
    forecast_disease_count,
    forecast_other_count,
    by = c(".draw", "date")
  ) |>
    dplyr::mutate(
      !!output_col :=
        .data[[disease_count_col]] /
          (.data[[disease_count_col]] +
            .data[[other_count_col]])
    )

  return(result)
}


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
    mutate(!!output_col := values) |>
    rename(
      quantile_level = quantile_levels, report_date = forecast_date,
      date = target_date
    ) |>
    select(date, quantile_level, all_of(output_col))

  return(cdc_flat_forecast)
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
  forecast_other <- fit_and_forecast(
    target_and_other_data,
    n_forecast_days,
    n_samples,
    target_col = "ed_visits_other",
    output_col = "other_ed_visits"
  )
  forecast_baseline_ts_count <- fit_and_forecast(
    target_and_other_data,
    n_forecast_days,
    n_samples,
    target_col = "ed_visits_target",
    output_col = "baseline_ed_visit_count_forecast"
  )
  ## Generate CDC flat forecast for the target disease number of ED visits
  forecast_baseline_cdc_count <- cdc_flat_forecast(
    target_and_other_data,
    target_col = "ed_visits_target",
    output_col = "baseline_ed_visit_count_forecast",
    data_frequency = "1 day",
    aheads = 1:n_forecast_days
  )

  forecast_baseline_ts_prop <- forecast_baseline_ts_count |>
    to_prop_forecast(forecast_other)

  forecast_baseline_cdc_prop <- cdc_flat_forecast(
    target_and_other_data |>
      mutate(ed_visits_prop = ed_visits_target /
        (ed_visits_target + ed_visits_other)),
    target_col = "ed_visits_prop",
    output_col = "baseline_ed_visit_prop_forecast",
    data_frequency = "1 day",
    aheads = 1:n_forecast_days
  )

  save_path_other <- path(
    model_run_dir,
    "other_ed_visits_forecast",
    ext = "parquet"
  )
  save_path_baseline_ts_count <- path(
    model_run_dir,
    "baseline_ts_count_ed_visits_forecast",
    ext = "parquet"
  )
  save_path_baseline_ts_prop <- path(
    model_run_dir,
    "baseline_ts_prop_ed_visits_forecast",
    ext = "parquet"
  )
  save_path_baseline_cdc_count <- path(
    model_run_dir,
    "baseline_cdc_count_ed_visits_forecast",
    ext = "parquet"
  )
  save_path_baseline_cdc_prop <- path(
    model_run_dir,
    "baseline_cdc_prop_ed_visits_forecast",
    ext = "parquet"
  )

  write_parquet(
    forecast_other,
    save_path_other
  )
  write_parquet(
    forecast_baseline_ts_count,
    save_path_baseline_ts_count
  )
  write_parquet(
    forecast_baseline_ts_prop,
    save_path_baseline_ts_prop
  )
  write_parquet(
    forecast_baseline_cdc_count,
    save_path_baseline_cdc_count
  )
  write_parquet(
    forecast_baseline_cdc_prop,
    save_path_baseline_cdc_prop
  )
}


p <- arg_parser(
  "Forecast other (non-target-disease) ED visits for a given location."
) |>
  add_argument(
    "model-run-dir",
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

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
  "nanoparquet",
  "glue",
  "epipredict",
  "epiprocess",
  "purrr",
  "rlang",
  "glue",
  "hewr"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


to_prop_forecast <- function(
  forecast_disease_count,
  forecast_other_count,
  disease_count_col = "observed_ed_visits",
  other_count_col = "other_ed_visits",
  output_col = "prop_disease_ed_visits"
) {
  result <- inner_join(
    forecast_disease_count,
    forecast_other_count,
    by = join_by(date, .draw)
  ) |>
    mutate(
      !!output_col := .data[[disease_count_col]] /
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
fit_and_forecast_ensemble <- function(
  data,
  n_forecast_days = 28,
  n_samples = 2000,
  target_col = "ed_visits",
  output_col = "other_ed_visits"
) {
  forecast_horizon <- glue("{n_forecast_days} days")
  target_sym <- sym(target_col)
  output_sym <- sym(output_col)

  offset <- 1

  fit <- data |>
    as_tsibble(index = date) |>
    filter(data_type == "train") |>
    model(
      comb_model = combination_ensemble(
        ETS(
          log(!!target_sym + !!offset) ~
            trend(
              method = c("N", "M", "A")
            )
        ),
        ARIMA(log(!!target_sym + !!offset))
      )
    )

  forecast_samples <- fit |>
    generate(h = forecast_horizon, times = n_samples) |>
    as_tibble() |>
    mutate(
      !!output_col := pmax(.data$.sim, 0), # clip values
      .draw = as.integer(.data$.rep)
    ) |>
    select("date", ".draw", all_of(output_col))

  if (any(forecast_samples[[output_col]] < 0)) {
    stop(glue::glue("Negative count forecast for {output_col}"))
  }

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
cdc_flat_forecast <- function(
  data,
  target_col = "ed_visits_target",
  output_col = "cdc_flat_ed_visits",
  ...
) {
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
    epipredict::pivot_quantiles_longer(".pred_distn") |>
    dplyr::rename(
      !!output_col := ".pred_distn_value",
      quantile_level = ".pred_distn_quantile_level",
      report_date = "forecast_date",
      date = "target_date"
    ) |>
    dplyr::select("date", "quantile_level", all_of(output_col))

  return(cdc_flat_forecast)
}

main <- function(
  model_run_dir,
  model_name,
  n_forecast_days = 28,
  n_samples = 2000,
  epiweekly = FALSE
) {
  resolution <- if_else(epiweekly, "epiweekly", "daily")
  prefix <- str_c(resolution, "_")
  aheads_cdc_baseline <- if_else(
    epiweekly,
    ceiling(n_forecast_days / 7),
    n_forecast_days
  )
  base_data_name <- "combined_training_data"
  data_name <- if_else(epiweekly, str_c(prefix, base_data_name), base_data_name)
  data_frequency <- if_else(epiweekly, "1 week", "1 day")

  data_path <- path(model_run_dir, "data", data_name, ext = "tsv")

  # Having a lab_site_index column of NA values errors
  # while using full_join later
  target_and_other_data <- read_tsv(
    data_path,
    col_types = cols(
      date = col_date(),
      geo_value = col_character(),
      disease = col_character(),
      data_type = col_character(),
      .variable = col_character(),
      .value = col_double()
    )
  ) |>
    dplyr::select(-"lab_site_index") |>
    filter(str_ends(.variable, "ed_visits")) |>
    pivot_wider(names_from = ".variable", values_from = ".value")

  geo_value <- target_and_other_data$geo_value[1]
  disease <- target_and_other_data$disease[1]

  ## Time series forecasting
  ## Fit and forecast other (non-target-disease) ED visits using a combination
  ## ensemble model
  ts_ensemble_other_e <- fit_and_forecast_ensemble(
    target_and_other_data,
    n_forecast_days,
    n_samples,
    target_col = "other_ed_visits",
    output_col = "other_ed_visits"
  )

  ts_ensemble_count_e <- fit_and_forecast_ensemble(
    target_and_other_data,
    n_forecast_days,
    n_samples,
    target_col = "observed_ed_visits",
    output_col = "observed_ed_visits"
  )

  ts_ensemble_prop_e <- ts_ensemble_count_e |>
    to_prop_forecast(ts_ensemble_other_e)

  ## Generate CDC flat forecast for the target disease number of ED visits
  baseline_cdc_count_e <- cdc_flat_forecast(
    target_and_other_data,
    target_col = "observed_ed_visits",
    output_col = "observed_ed_visits",
    data_frequency = data_frequency,
    aheads = 1:aheads_cdc_baseline
  )

  baseline_cdc_prop_e <- cdc_flat_forecast(
    target_and_other_data |>
      mutate(
        prop_disease_ed_visits = observed_ed_visits /
          (observed_ed_visits + other_ed_visits)
      ),
    target_col = "prop_disease_ed_visits",
    output_col = "prop_disease_ed_visits",
    data_frequency = data_frequency,
    aheads = 1:aheads_cdc_baseline
  )

  model_dir <- path(model_run_dir, model_name)
  dir_create(model_dir)

  ts_ensemble_forecast_e <-
    ts_ensemble_prop_e |>
    pivot_longer(
      -c("date", ".draw"),
      names_to = ".variable",
      values_to = ".value"
    ) |>
    mutate(
      geo_value = geo_value,
      disease = disease,
      resolution = resolution,
      aggregated_numerator = FALSE,
      aggregated_denominator = if_else(
        str_starts(.variable, "prop_"),
        FALSE,
        NA
      )
    ) |>
    select(
      "date",
      ".draw",
      "geo_value",
      "disease",
      "resolution",
      "aggregated_numerator",
      "aggregated_denominator",
      ".variable",
      ".value"
    )

  baseline_cdc_forecast_e <-
    dplyr::full_join(
      baseline_cdc_count_e,
      baseline_cdc_prop_e,
      by = c("date", "quantile_level")
    ) |>
    pivot_longer(
      -c("date", "quantile_level"),
      names_to = ".variable",
      values_to = ".value"
    ) |>
    mutate(
      geo_value = geo_value,
      disease = disease,
      resolution = resolution,
      aggregated_numerator = FALSE,
      aggregated_denominator = if_else(
        str_starts(.variable, "prop_"),
        FALSE,
        NA
      )
    ) |>
    select(
      "date",
      "geo_value",
      "disease",
      "resolution",
      "aggregated_numerator",
      "aggregated_denominator",
      ".variable",
      "quantile_level",
      ".value"
    )

  to_save <- tribble(
    ~base_name,
    ~value,
    "baseline_cdc_quantiles_e",
    baseline_cdc_forecast_e,
    "ts_ensemble_samples_e",
    ts_ensemble_forecast_e
  ) |>
    mutate(
      save_path = path(
        !!model_dir,
        glue::glue("{prefix}{base_name}"),
        ext = "parquet"
      )
    )

  walk2(
    to_save$value,
    to_save$save_path,
    write_parquet
  )
}


p <- arg_parser(
  "Forecast other (non-target-disease) ED visits for a given location."
) |>
  add_argument(
    "model_run_dir",
    help = "Directory containing the model data and output.",
  ) |>
  add_argument(
    "--model-name",
    help = "Name of model.",
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
model_name <- argv$model_name
n_forecast_days <- argv$n_forecast_days
n_samples <- argv$n_samples

disease_name_nssp_map <- c(
  "covid-19" = "COVID-19",
  "influenza" = "Influenza"
)

disease_name_nssp <- parse_model_run_dir_path(model_run_dir)$disease

# Baseline forecasts on 1 day resolution
main(
  model_run_dir,
  model_name,
  n_forecast_days = n_forecast_days,
  n_samples = n_samples
)
# Baseline forecasts on 1 (epi)week resolution
main(
  model_run_dir,
  model_name,
  n_forecast_days = n_forecast_days,
  n_samples = n_samples,
  epiweekly = TRUE
)

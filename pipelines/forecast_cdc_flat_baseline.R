script_packages <- c(
  "dplyr",
  "tidyr",
  "tibble",
  "readr",
  "stringr",
  "fs",
  "argparser",
  "arrow",
  "glue",
  "epipredict",
  "epiprocess",
  "hewr"
)


## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})

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
  epiweekly = FALSE
) {
  aheads_cdc_baseline <- if_else(
    epiweekly,
    ceiling(n_forecast_days / 7),
    n_forecast_days
  )
  data_frequency <- if_else(epiweekly, "1 week", "1 day")

  data_info <- hewr::load_training_data(
    model_run_dir,
    "combined_training_data",
    epiweekly
  )
  target_and_other_data <- data_info$data
  geo_value <- data_info$geo_value
  disease <- data_info$disease
  resolution <- data_info$resolution
  prefix <- data_info$prefix

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

  baseline_cdc_forecast_e <-
    dplyr::full_join(
      baseline_cdc_count_e,
      baseline_cdc_prop_e,
      by = c("date", "quantile_level")
    ) |>
    hewr::format_timeseries_output(
      geo_value = geo_value,
      disease = disease,
      resolution = resolution,
      output_type_id = "quantile_level"
    ) |>
    write_parquet(path(
      model_dir,
      glue::glue("{prefix}baseline_cdc_quantiles_e"),
      ext = "parquet"
    ))
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
  )
argv <- parse_args(p)
model_run_dir <- path(argv$model_run_dir)
model_name <- argv$model_name
n_forecast_days <- argv$n_forecast_days

# Baseline forecasts on 1 day resolution
main(
  model_run_dir,
  model_name,
  n_forecast_days = n_forecast_days
)
# Baseline forecasts on 1 (epi)week resolution
main(
  model_run_dir,
  model_name,
  n_forecast_days = n_forecast_days,
  epiweekly = TRUE
)

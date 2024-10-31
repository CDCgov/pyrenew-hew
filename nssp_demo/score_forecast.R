library(dplyr)
library(scoringutils)
library(arrow)

#' Join Forecast and Actual Data
#'
#' This function reads forecast data from a Parquet file and actual data from a
#' TSV file, then joins them using a specified key.
#'
#' @param forecast_source A character string specifying the path to the
#' directory containing Parquet file(s) containing forecast data.
#' @param data_path A character string specifying the path to the TSV file
#' containing actual/truth data.
#' @param join_key A character vector specifying the key(s) to join the forecast
#' and actual data on. Default is NULL.
#' @param ... Additional arguments passed to `arrow::open_dataset`.
#'
#' @return A data frame resulting from the left join of the forecast and actual
#' data.
#' @export
join_forecast_and_data <- function(
    forecast_source, data_path, join_key = NULL,
    ...) {
  predictions <- arrow::read_parquet(forecast_source, ...)
  actual_data <- readr::read_tsv(data_path) |> rename(true_value = value)
  joined_data <- dplyr::left_join(predictions, actual_data, by = join_key)
  return(joined_data)
}


#' Score Forecasts
#'
#' This function scores forecast data using the `scoringutils` package. It takes
#' in scorable data, that is data which has a joined truth data and forecast
#' data, and scores this.
#'
#' This function aims at scoring _sampled_ forecasts. Care must be taken to
#' select the appropriate columns for the observed and predicted values, as well
#' as the forecast unit. The expected `sample_id` column is `.draw` due to
#' expecting input from a tidybayes format.
#'
#' NB: this function assumes that _log-scale_ scoring is the default. If you
#' want to vary this behaviour, you can splat additional arguments to
#' `scoringutils::transform_forecasts` such as the identity transformation e.g.
#' `fun = identity` with `label = "identity"`.
#'
#' If more than one model is present in the data, in the column `model_col` the
#' function will add relative skill metrics to the output.
#'
#' @param scorable_data A data frame containing the data to be scored.
#' @param forecast_unit A character string specifying the forecast unit.
#' @param observed A character string specifying the column name for observed
#' values.
#' @param predicted A string vector specifying the column name for predicted
#' values.
#' @param sample_id A character string specifying the column name for sample
#' IDs. Default is ".draw".
#' @param model_col A character string specifying the column name for models.
#' @param ... Additional arguments passed to
#' `scoringutils::transform_forecasts`.
#'
#' @return A data frame with scored forecasts and relative skill metrics.
#' @export
score_forecasts <- function(
    scorable_data, forecast_unit, observed, predicted,
    sample_id = ".draw", model_col = "model", ...) {
  scored_data <- scorable_data |>
    collect() |>
    scoringutils::as_forecast_sample(
      forecast_unit = forecast_unit,
      observed = observed,
      predicted = predicted,
      sample_id = sample_id
    ) |>
    scoringutils::transform_forecasts(...) |>
    scoringutils::score()
  # Add relative skill if more than one model is present
  if (n_distinct(scorable_data[[model_col]]) != 1) {
    scored_data <- scoringutils::add_relative_skill(scored_data)
  }
  return(scored_data)
}

base_path <- "nssp_demo/private_data/influenza_r_2024-10-31_f_2024-10-11_t_2024-10-30/MA"
forecast_path <- fs::path(base_path, "forecast_samples.parquet")
truth_path <- fs::path(base_path, "data.tsv")
forecast_date <- lubridate::ymd("2024-10-31")

predictions <- arrow::read_parquet(forecast_path)

actual_data <- readr::read_tsv(truth_path) |>
    rename(true_value = value) |>
    filter(data_type == "test")
joined_data <- dplyr::inner_join(predictions, actual_data,
                                 by = c(disease, date))

max_visits <- actual_data |>
    filter(disease == "Total") |>
    pull(true_value) |>
    max()

scored <- score_forecasts(to_score |>
                filter(disease == "prop_disease_ed_visits") |>
                mutate(model = "pyrenew-hew"),
                forecast_unit=c("date"),
                observed="true_value",
                sample_id="sample_id",
                predicted=".value",
                offset = 1 / max_visits)

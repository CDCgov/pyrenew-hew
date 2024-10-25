library(dplyr)
library(scoringutils)

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
#' @param ... Additional arguments passed to
#' `scoringutils::transform_forecasts`.
#'
#' @return A data frame with scored forecasts and relative skill metrics.
#' @export
score_forecasts <- function(scorable_data, forecast_unit, observed, predicted,
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
    model_sym <- rlang::sym(model_col)
    if (summarize(scored_data,
        all_same = n_distinct(!!model_sym) != 1)$all_same) {
        scored_data <- scoringutils::add_relative_skill(scored_data)
    }
  return(scored_data)
}

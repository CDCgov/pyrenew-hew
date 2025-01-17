#' Read in and score hubverse-format output from `pyrenew-hew`.
#'
#' Expects forecast output created by [to_epiweekly_quantile_table()]
#' and an observed data table with location, date, and value columns.
#' The column names in the observed data table can be configured;
#' defaults are `"location"`, `"reference_date"`, and
#' `"value"`, respectively, to reflecting the format used
#' in pyrenew-hew pipeline scripts such as
#' `pipelines/create_observed_data_table.py`.
#'
#' @param forecast forecasts, as a hubverse-format
#' [`tibble`][tibble::tibble()] produced by
#' [to_epiweekly_quantile_table()], with columns
#' `target_end_date`, `value`, and `horizon`
#' @param observed observations, as a [`tibble`][tibble::tibble()].
#' @param horizons Horizons to score. Default `c(0, 1)`
#' @param transform transformation passed as the
#' `fun` argument to [scoringutils::transform_forecasts()].
#' Default [scoringutils::log_shift()].
#' @param offset Offset for the transformation, passed to
#' [scoringutils::transform_forecasts()]. Default 1.
#' @param append_transformed When calling
#' [scoringutils::transform_forecasts()], append
#' the transformed scale forecasts to the base scale forecasts
#' or keep only the transformed scale forecasts? Passed as the
#' `append` argument to [scoringutils::transform_forecasts()].
#' Boolean, default `FALSE` (keep only transformed scale).
#' @param offset Offset for the transform passed to
#' [scoringutils::transform_forecasts()].
#' @param observed_value_column Name of the column containing
#' observed values in the `observed` table, as a string.
#' Default `"value"`
#' @param observed_location_column Name of the column containing
#' location values in the `observed` table, as a string.
#' Default `"location"`
#' @param observed_date_column Name of the column containing
#' date values in the `observed` table, as a string.
#' Default `"reference_date"`
#' @param quantile_tol Round quantile level values to this many
#' decimal places, to avoid problems with floating point number
#' comparisons. Passed as the `digits` argument to [base::round()].
#' Default 10.
#' @param ... Other keyword arguments passed to
#' [scoringutils::transform_forecasts()].
#' @return A table of scores, as the output of
#' [scoringutils::score()], filtered to include only the
#' transformed_scale.
#' @export
score_hubverse <- function(forecast,
                           observed,
                           horizons = c(0, 1),
                           transform = scoringutils::log_shift,
                           append_transformed = FALSE,
                           offset = 1,
                           observed_value_column = "value",
                           observed_location_column = "location",
                           observed_date_column = "reference_date",
                           quantile_tol = 10,
                           ...) {
  obs <- observed |>
    dplyr::select(
      location = !!observed_location_column,
      target_end_date = !!observed_date_column,
      observed = !!observed_value_column
    )

  to_score <- forecast |>
    dplyr::filter(.data$horizon %in% !!horizons) |>
    dplyr::inner_join(obs,
      by = c(
        "location",
        "target_end_date"
      )
    ) |>
    dplyr::mutate(output_type_id = output_type_id |>
      as.numeric() |>
      round(digits = !!quantile_tol)) |>
    scoringutils::as_forecast_quantile(
      predicted = "value",
      observed = "observed",
      quantile_level = "output_type_id"
    ) |>
    scoringutils::transform_forecasts(
      fun = transform,
      append = append_transformed,
      offset = offset,
      ...
    )

  interval_coverage_95 <- purrr::partial(
    scoringutils::interval_coverage,
    interval_range = 95
  )

  scored <- to_score |>
    scoringutils::score(
      metrics = c(
        scoringutils::get_metrics(to_score),
        interval_coverage_95 = interval_coverage_95
      )
    )

  return(scored)
}

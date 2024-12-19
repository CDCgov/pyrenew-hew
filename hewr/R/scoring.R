#' Default scoring function for `pyrenew-hew`.
#'
#' Simple wrapper of [scoringutils::transform_forecasts()]
#' and [scoringutils::score()] with reasonable defaults for
#' `pyrenew-hew` production.
#'
#' @param scoreable_table Table to score, as the output
#' of [forecasttools::quantile_table_to_scoreable()].
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
#' @param ... Other keyword arguments passed to
#' [scoringutils::transform_forecasts()].
#' @return A table of scores, as the output of
#' [scoringutils::score()], filtered to include only the
#' transformed_scale.
#' @export
score_hewr <- function(scoreable_table,
                       transform = scoringutils::log_shift,
                       append_transformed = FALSE,
                       offset = 1,
                       ...) {
  to_score <- scoreable_table |>
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

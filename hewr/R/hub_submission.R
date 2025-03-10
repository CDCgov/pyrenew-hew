#' Downsample a large hubverse table to a table
#' that contains only one particular model and only the
#' columns expected in a typical single model submission
#' to a hubverse-format Hub.
#'
#' Useful for creating Hub submissions.
#'
#' @param df A large format hubverse table, as the output
#' of [to_epiweekly_quantile_table()]
#' @param model Name of the model for which to get hubverse
#' submittable quantiles
#' @param targets Name(s) of the targets(s) for which to get
#' submittable quantiles. If `NULL`, get all included targets
#' from that model.
#' @return The quantiles as a submittable hubverse table for the
#' given model.
#' @export
to_single_model_hubverse_table <- function(df,
                                           model,
                                           targets = NULL) {
  result <- df |>
    dplyr::filter(
      .data$model == !!model,
      forecasttools::nullable_comparison(
        .data$.target, "%in%", !!targets
      ),
      .data$output_type == "quantile"
    ) |>
    dplyr::select(
      "reference_date",
      "target",
      "horizon",
      "target_end_date",
      "location",
      "output_type",
      "output_type_id",
      "value"
    )

  return(result)
}

#' Find multiple forecast options for a given target and location
#' in a hubverse table.
#'
#' @param df Hubverse table to examine.
#' @param model_id_columns Columns that serve as a primary
#' key for individual model runs.
#' @export
find_multiple_options <- function(df,
                                  model_run_id_columns) {
  multi <- df |>
    dplyr::distinct(
      .data$target,
      .data$location,
      dplyr::all_of(model_id_columns)
    ) |>
    dplyr::group_by(.data$target, .data$location) |>
    dplyr::filter(dplyr::n() > 1) |>
    dplyr::arrange(
      .data$target,
      .data$location
    )

  return(multi)
}

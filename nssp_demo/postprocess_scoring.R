script_packages <- c(
  "dplyr",
  "scoringutils",
  "lubridate"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})

#' Summarise Scoring Table using quantile scores
#'
#' This function takes a scoring table and summarises it by calculating both
#' relative and absolute Weighted Interval Scores (WIS) for each model.
#' The relative WIS is computed by comparing each model to a baseline model
#' ("cdc_baseline"), while the absolute WIS, median absolute error (MAE),
#' and interval coverages (50% and 90%) are directly summarised from the
#' scoring table.
#'
#' @param score_table A data frame containing the scoring table with quantile
#' scores.
#' @param scale A character string specifying the scale to filter the quantile
#' scores. Default is "natural".
#'
#' @return A data frame with summarised scores for each model, including
#' relative WIS, absolute WIS, MAE, and interval coverages (50% and 90%).
#'
#' @examples
#' # Assuming `score_table` is a data frame with the necessary structure:
#' summarised_scores <- summarised_scoring_table(score_table, scale = "natural")
#' print(summarised_scores)
summarised_scoring_table <- function(score_table, scale = "natural") {
  rel_wis <- score_table$quantile_scores |>
    filter(scale == !!scale) |>
    get_pairwise_comparisons(baseline = "cdc_baseline") |>
    group_by(model) |>
    summarise(rel_wis = mean(wis_scaled_relative_skill))

    abs_wis <- score_table$quantile_scores |>
        filter(scale == !!scale) |>
        summarise_scores(by = "model") |>
        select(model, abs_wis = wis, mae = ae_median, interval_coverage_50,
            interval_coverage_90)

    summarised_scores <- left_join(abs_wis, rel_wis, by = "model")
    return(summarised_scores)
}

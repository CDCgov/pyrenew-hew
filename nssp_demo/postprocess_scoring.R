script_packages <- c(
  "dplyr",
  "scoringutils",
  "lubridate",
  "ggplot2"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})

epiweek_to_date <- function(epiweek, epiyear) {
  # Create date for January 1st of the epiyear
  jan1 <- as.Date(paste0(epiyear, "-01-01"))
  # Calculate days to add (epiweeks start on Sunday)
  days_to_add <- (epiweek - 1) * 7
  # Add days and adjust to previous Sunday
  date <- jan1 + days_to_add
  date <- date - lubridate::wday(date, week_start = 7)
  return(date)
}

#' Summarise Scoring Table using quantile scores
#'
#' This function takes a scoring table and summarises it by calculating both
#' relative and absolute Weighted Interval Scores (WIS) for each model.
#' The relative WIS is computed by comparing each model to a baseline model
#' ("cdc_baseline"), while the absolute WIS, median absolute error (MAE),
#' and interval coverages (50% and 90%) are directly summarised from the
#' scoring table.
#'
#' @param score_table A scoring object containing the scoring table with
#' quantile scores.
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
    select(model,
      abs_wis = wis, mae = ae_median, interval_coverage_50,
      interval_coverage_90
    )

  summarised_scores <- left_join(abs_wis, rel_wis, by = "model")
  return(summarised_scores)
}

#' @title Epiweekly Scoring Plot
#' This function generates a line plot of the weighted interval scores (WIS) for
#' different models over epiweeks.
#' @param quantile_scores A scoringutils output table
#' containing the unsummarised quantile scores.
#' @param scale A character string specifying the scale
#' to filter the quantile
#' scores. Default is "natural".
#' @return A ggplot object representing the epiweekly scoring plot.
epiweekly_scoring_plot <- function(quantile_scores, scale = "natural") {
  epiweekly_score_fig <- quantile_scores |>
    filter(scale == !!scale) |>
    mutate(epiweek = epiweek(date), epiyear = epiyear(date)) |>
    get_pairwise_comparisons(
      by = c("epiweek", "epiyear"),
      baseline = "cdc_baseline"
    ) |>
    mutate(epidate = epiweek_to_date(epiweek, epiyear)) |>
    group_by(model, epidate) |>
    summarise(
      wis = mean(wis_scaled_relative_skill),
      .groups = "drop"
    ) |>
    as_tibble() |>
    ggplot(aes(x = epidate, y = wis, color = model)) +
    geom_line() +
    geom_point() + # Add points to the line plot
    labs(
      title = "Epiweekly Scoring by Model",
      x = "Epiweek start dates",
      y = "Relative Weighted Interval Score (WIS)"
    ) +
    theme_minimal()

  return(epiweekly_score_fig)
}

location_rel_wis_plot <- function(location, quantile_scores, ...) {
  return(epiweekly_scoring_plot(
    quantile_scores |>
      dplyr::filter(location == !!location),
    ...
  ) + ggtitle(
    glue::glue("Relative WIS over time for {location}")
  ))
}

#' Save a list of plots as a PDF, with a
#' grid of `nrow` by `ncol` plots per page
#'
#' @param list_of_plots list of plots to save to PDF
#' @param save_path path to which to save the plots
#' @param nrow Number of rows of plots per page
#' (passed to [gridExtra::marrangeGrob()])
#' Default `1`.
#' @param ncol Number of columns of plots per page
#' (passed to [gridExtra::marrangeGrob()]).
#' Default `1`.
#' @param width page width in device units (passed to
#' [ggplot2::ggsave()]). Default `8.5`.
#' @param height page height in device units (passed to
#' [ggplot2::ggsave()]). Default `11`.
#' @return `TRUE` on success.
#' @export
plots_to_pdf <- function(list_of_plots,
                         save_path,
                         nrow = 1,
                         ncol = 1,
                         width = 8.5,
                         height = 11) {
  if (!stringr::str_ends(
    save_path, ".pdf"
  )) {
    cli::cli_abort("Filepath must end with `.pdf`")
  }
  cli::cli_inform("Saving plots to {save_path}")
  ggplot2::ggsave(
    filename = save_path,
    plot = gridExtra::marrangeGrob(list_of_plots,
      nrow = nrow,
      ncol = ncol
    ),
    width = width,
    height = height
  )
  return(TRUE)
}


main <- function(path_to_scores) {
  scores <- readRDS(path_to_scores)

  quantile_scores <- scores$quantile_scores
  locations <- unique(quantile_scores$location) |>
    purrr::set_names()
  overall_fig <- epiweekly_scoring_plot(
    quantile_scores,
    scale = "log"
  )

  location_figs <- purrr::map(locations,
    location_rel_wis_plot,
    quantile_scores = quantile_scores,
    scale = "log"
  )

  ggsave("ws_by_time_overall.pdf", overall_fig)

  plots_to_pdf(location_figs,
    "wis_by_time_and_location.pdf",
    width = 7,
    height = 5
  )
}

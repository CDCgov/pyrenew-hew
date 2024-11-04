script_packages <- c(
  "dplyr",
  "scoringutils",
  "lubridate",
  "ggplot2",
  "argparser"
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
#' @param qunatile_scores A scoring object containing the scoring table with
#' quantile scores.
#' @param scale A character string specifying the scale to filter the quantile
#' scores. Default is "natural".
#'
#' @return A data frame with summarised scores for each model, including
#' relative WIS, absolute WIS, MAE, and interval coverages (50% and 90%).
#'
#' @examples
#' # Assuming `quantile_scores` is a data frame with the necessary structure:
#' summarised_scores <- summarised_scoring_table(
#'   quantile_scores,
#'   scale = "natural"
#' )
#' print(summarised_scores)
summarised_scoring_table <- function(quantile_scores,
                                     scale = "natural") {
  rel_wis <- quantile_scores |>
    filter(scale == !!scale) |>
    get_pairwise_comparisons(baseline = "cdc_baseline") |>
    group_by(model) |>
    summarise(rel_wis = mean(wis_scaled_relative_skill))

  abs_wis <- quantile_scores |>
    filter(scale == !!scale) |>
    summarise_scores(by = "model") |>
    select(model,
      abs_wis = wis,
      mae = ae_median,
      interval_coverage_50,
      interval_coverage_90
    )

  summarised_scores <- left_join(abs_wis, rel_wis, by = "model")
  return(summarised_scores)
}


location_summary_table <- function(quantile_scores,
                                   scale = "natural") {
  rel_wis <- quantile_scores |>
    filter(scale == !!scale) |>
    get_pairwise_comparisons(
      baseline = "cdc_baseline",
      by = "location"
    ) |>
    group_by(model, location) |>
    summarise(rel_wis = mean(wis_scaled_relative_skill))

  abs_wis <- quantile_scores |>
    filter(scale == !!scale) |>
    summarise_scores(by = c("model", "location")) |>
    select(model,
      location,
      abs_wis = wis,
      mae = ae_median,
      interval_coverage_50,
      interval_coverage_90
    )

  summarised_scores <- left_join(abs_wis, rel_wis,
    by = c("model", "location")
  )
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
    scale_y_continuous(trans = "log10") +
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

location_score_table <- function(location, quantile_scores, ...) {
  return
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

relative_wis_by_location <- function(scores,
                                     baseline_model = "cdc_baseline") {
  scoring_data <- scores |>
    get_pairwise_comparisons(
      by = c("date", "location"),
      baseline = baseline_model
    ) |>
    group_by(model, location) |>
    summarise(
      relative_wis = mean(wis_scaled_relative_skill),
      .groups = "drop"
    ) |>
    filter(model == "pyrenew-hew")

  min_wis <- min(scoring_data$relative_wis)
  max_wis <- max(scoring_data$relative_wis)
  max_overall <- max(1 / min_wis, max_wis)
  theme_minimal()


  fig <- scoring_data |>
    arrange(relative_wis) |>
    mutate(location = factor(location,
      ordered = TRUE,
      levels = location
    )) |>
    ggplot(
      aes(
        y = location,
        x = relative_wis,
        group = model
      )
    ) +
    geom_point(
      shape = 21,
      size = 3,
      fill = "darkblue",
      color = "black"
    ) +
    geom_vline(
      xintercept = 1,
      linetype = "dashed"
    ) +
    scale_x_continuous(trans = "log10") +
    coord_cartesian(xlim = c(1 / max_overall, max_overall)) +
    theme_minimal()

  return(fig)
}


main <- function(path_to_scores,
                 output_directory,
                 output_prefix = "") {
  get_save_path <- function(filename, ext = "pdf") {
    fs::path(output_directory,
      glue::glue("{output_prefix}{filename}"),
      ext = ext
    )
  }

  scores <- readRDS(path_to_scores)

  quantile_scores <- scores$quantile_scores

  locations <- unique(quantile_scores$location) |>
    purrr::set_names()

  message("Plotting relative WIS by forecast date across locations...")

  rel_wis_by_date <- epiweekly_scoring_plot(
    quantile_scores,
    scale = "log"
  )

  rel_wis_by_date_save_path <- get_save_path("relative_wis_by_date")

  message(glue::glue("Saving figure to {rel_wis_by_date_save_path}..."))
  ggsave(rel_wis_by_date_save_path,
    rel_wis_by_date,
    width = 8,
    height = 4
  )


  message("Plotting relative WIS by forecast date and location...")
  rel_wis_by_date_and_location <- purrr::map(locations,
    location_rel_wis_plot,
    quantile_scores = quantile_scores,
    scale = "log"
  )

  rel_wis_by_date_loc_save_path <- get_save_path(
    "relative_wis_by_date_and_location"
  )

  message(
    glue::glue("Saving figure to {rel_wis_by_date_loc_save_path}...")
  )

  plots_to_pdf(rel_wis_by_date_and_location,
    rel_wis_by_date_loc_save_path,
    width = 8,
    height = 4
  )

  message("Plotting WIS components by location for pyrenew-hew...")
  wis_components_by_location <-
    scoringutils::plot_wis(
      quantile_scores |>
        filter(model == "pyrenew-hew"),
      x = "location"
    )
  wis_comp_by_loc_save_path <- get_save_path(
    "wis_components_by_location",
    ext = "png"
  ) # wis components do not save well as vectors
  ggsave(
    wis_comp_by_loc_save_path,
    wis_components_by_location
  )

  wis_components_by_model <-
    scoringutils::plot_wis(quantile_scores,
      x = "model"
    )

  wis_comp_by_model_save_path <- get_save_path(
    "wis_components_by_model",
    ext = "png"
  )
  ggsave(
    wis_comp_by_model_save_path,
    wis_components_by_model
  )

  message("Plotting relative WIS across dates by location")
  rel_wis_by_location <- relative_wis_by_location(
    quantile_scores
  )

  rel_wis_by_location_save_path <- get_save_path(
    "relative_wis_by_location"
  )

  ggsave(rel_wis_by_location_save_path,
    rel_wis_by_location,
    height = 10,
    width = 4
  )

  message("Making tables...")
  table_all <- summarised_scoring_table(
    quantile_scores,
    scale = "log"
  )

  table_all_save_path <- get_save_path(
    "overall_scores",
    ext = "tsv"
  )
  readr::write_tsv(table_all, table_all_save_path)


  table_locs <- location_summary_table(quantile_scores,
    scale = "log"
  )

  table_locs_save_path <- get_save_path(
    "scores_by_location",
    ext = "tsv"
  )
  readr::write_tsv(table_locs, table_locs_save_path)

  message("Done with score postprocessing.")
}

p <- arg_parser |>
  add_argument(
    "path_to_scores",
    help = paste0(
      "Path to a file holding all scores, as an .rds ",
      "file, in the list of scoringutils objects output ",
      "format of collate_score_tables.R"
    )
  ) |>
  add_argument("--output-directory",
    help = paste0(
      "Output directory in which to save the ",
      "generated score plots and tables. ",
      "Default '.', i.e. the current working ",
      "directory"
    ),
    default = "."
  ) |>
  add_argument("--output-prefix",
    help = paste0(
      "Prefix to append to output file names, e.g. ",
      "the name(s) of the target disease(s) ",
      "and/or epidemiological signal(s). ",
      "Default '' (no prefix)"
    ),
    default = ""
  )


argv <- p$parse_args()

main(
  argv$path_to_scores,
  argv$output_directory,
  argv$output_prefix
)

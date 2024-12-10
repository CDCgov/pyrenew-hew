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


#' Summarise Scoring Table using quantile scores
#'
#' This function takes a scoring table and summarises it by calculating both
#' relative and absolute Weighted Interval Scores (WIS) for each model.
#' The relative WIS is computed by comparing each model to a baseline model
#' ("cdc_baseline"), while the absolute WIS, median absolute error (MAE),
#' and interval coverages (50% and 90%) are directly summarised from the
#' scoring table.
#'
#' @param quantile_scores A scoring object containing the scoring table with
#' quantile scores.
#' @param scale A string specifying the scale to filter the quantile
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
                                     scale = "natural",
                                     baseline = "cdc_baseline",
                                     by = NULL) {
  filtered_scores <- quantile_scores |>
    filter(scale == !!scale) |>
    with_horizons()


  summarised_rel <- filtered_scores |>
    get_pairwise_comparisons(
      baseline = baseline,
      by = by
    ) |>
    filter(.data$compare_against == !!baseline) |>
    select(model,
      all_of(by),
      relative_wis =
        "wis_scaled_relative_skill"
    )

  summarised <- filtered_scores |>
    summarise_scores(by = c("model", by)) |>
    select(model,
      all_of(by),
      abs_wis = wis,
      mae = ae_median,
      interval_coverage_50,
      interval_coverage_90,
      interval_coverage_95
    ) |>
    inner_join(summarised_rel,
      by = c("model", by)
    )
  return(summarised)
}


with_horizons <- function(df) {
  return(df |>
    mutate(horizon = floor(as.numeric(.data$date -
      .data$report_date) / 7)))
}


plot_scores_by_date <- function(scores_by_date,
                                date_column = "date",
                                score_column = "relative_wis",
                                model_column = "model",
                                plot_title = "Scores by model over time",
                                xlabel = "Date",
                                ylabel = "Relative WIS") {
  min_score <- min(scores_by_date[[score_column]])
  max_score <- max(scores_by_date[[score_column]])
  max_overall <- max(c(1 / min_score, max_score))
  sym_ylim <- c(1 / max_overall, max_overall)

  score_fig <- scores_by_date |>
    ggplot(aes(
      x = .data[[date_column]],
      y = .data[[score_column]],
      color = .data[[model_column]],
      fill = .data[[model_column]]
    )) +
    geom_line(linewidth = 2) +
    geom_point(
      shape = 21,
      size = 3,
      color = "black"
    ) +
    labs(
      title = plot_title,
      x = xlabel,
      y = ylabel
    ) +
    scale_y_continuous(trans = "log10") +
    theme_minimal() +
    coord_cartesian(ylim = sym_ylim) +
    facet_wrap(~horizon)

  return(score_fig)
}

location_rel_wis_plot <- function(location, scores, ...) {
  return(plot_scores_by_date(
    scores |>
      dplyr::filter(location == !!location),
    ...
  ) + ggtitle(
    glue::glue("Relative WIS over time for {location}")
  ))
}


relative_wis_by_location <- function(summarised_scores,
                                     model = "pyrenew-hew") {
  summarised_scores <- summarised_scores |>
    filter(.data$model == !!model)

  ## compute x limits
  min_wis <- min(summarised_scores$relative_wis)
  max_wis <- max(summarised_scores$relative_wis)
  max_overall <- max(c(1 / min_wis, max_wis))
  sym_xlim <- c(1 / max_overall, max_overall)

  ## compute location order (by 0-week horizon
  ## rel wis
  ordered_locs <- summarised_scores |>
    filter(.data$horizon == min(.data$horizon)) |>
    arrange(.data$relative_wis) |>
    pull("location")

  fig <- summarised_scores |>
    mutate(location = factor(.data$location,
      ordered = TRUE,
      levels = !!ordered_locs
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
    coord_cartesian(xlim = sym_xlim) +
    theme_minimal() +
    facet_wrap(~horizon,
      nrow = 1
    )

  return(fig)
}

coverage_plot <- function(data,
                          coverage_level,
                          date_column = "date") {
  coverage_column <-
    glue::glue("interval_coverage_{100 * coverage_level}")
  return(
    ggplot(
      data = data,
      mapping = aes(
        x = .data[[date_column]],
        y = .data[[coverage_column]]
      )
    ) +
      geom_line(linewidth = 2) +
      geom_point(shape = 21, size = 3, fill = "darkgreen") +
      geom_hline(
        yintercept = coverage_level,
        linewidth = 1.5,
        linetype = "dashed"
      ) +
      facet_wrap(~horizon_name) +
      scale_y_continuous(label = scales::label_percent()) +
      scale_x_date() +
      coord_cartesian(ylim = c(0, 1)) +
      theme_minimal()
  )
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

  scores <- readr::read_rds(path_to_scores)

  quantile_scores <- scores$quantile_scores

  locations <- unique(quantile_scores$location) |>
    purrr::set_names()


  message("Making tables...")
  desired_tables <- c(
    "overall",
    "report_date",
    "horizon",
    "location",
    "report_date__location",
    "report_date__horizon",
    "horizon__location",
    "report_date__horizon__location"
  )

  make_and_save_table <- function(table_name) {
    message(
      "Making table ",
      table_name,
      "..."
    )
    tab_by <- if (table_name == "overall") {
      NULL
    } else {
      stringr::str_split_1(table_name, "__")
    }

    table <- summarised_scoring_table(
      quantile_scores,
      scale = "log",
      by = tab_by
    )

    save_path <- get_save_path(
      glue::glue("table_scores_by_{table_name}"),
      ext = "tsv"
    )
    message(
      "Saving table to ",
      save_path,
      "..."
    )
    readr::write_tsv(table, save_path)

    return(table)
  }

  score_tables <- purrr::map(
    desired_tables |> purrr::set_names(),
    make_and_save_table
  )


  message("Making plots...")
  message("Making coverage plots...")

  for_coverage_plots <- quantile_scores |>
    with_horizons() |>
    summarise_scores(by = c("model", "report_date", "horizon")) |>
    mutate(horizon_name = glue::glue("{horizon} week ahead")) |>
    filter(model == "pyrenew-hew")

  coverage_plots <-
    purrr::map(
      c(0.5, 0.9, 0.95),
      \(x) {
        coverage_plot(for_coverage_plots,
          x,
          date_column = "report_date"
        )
      }
    )
  forecasttools::plots_to_pdf(
    coverage_plots,
    get_save_path(
      "coverage_by_date_and_horizon"
    ),
    width = 8,
    height = 4
  )

  message("Plotting relative WIS by forecast date across locations...")

  rel_wis_by_date <- plot_scores_by_date(
    score_tables$report_date__horizon,
    date_column = "report_date"
  )

  rel_wis_by_date_save_path <- get_save_path("relative_wis_by_date")

  message(glue::glue("Saving figure to {rel_wis_by_date_save_path}..."))
  ggsave(rel_wis_by_date_save_path,
    rel_wis_by_date,
    width = 10,
    height = 8
  )

  message("Plotting relative WIS by horizon, forecast date, location...")
  rel_wis_by_dhl <-
    purrr::map(
      locations,
      \(x) {
        location_rel_wis_plot(
          x,
          scores =
            score_tables$report_date__horizon__location,
          date_column = "report_date"
        )
      }
    )

  forecasttools::plots_to_pdf(
    rel_wis_by_dhl,
    get_save_path(
      "relative_wis_by_date_horizon_location"
    ),
    width = 10,
    height = 8
  )

  message("Plotting WIS components by location for pyrenew-hew...")
  wis_components_by_location <-
    scoringutils::plot_wis(
      quantile_scores |>
        filter(model == "pyrenew-hew"),
      x = "location"
    )

  ggsave(
    get_save_path(
      "wis_components_by_location"
    ),
    wis_components_by_location
  )

  wis_components_by_model <-
    scoringutils::plot_wis(quantile_scores,
      x = "model"
    )

  ggsave(
    get_save_path(
      "wis_components_by_model"
    ),
    wis_components_by_model
  )

  message("Plotting relative WIS across dates by location and horizon")
  rel_wis_by_location_horizon <- relative_wis_by_location(
    score_tables$horizon__location
  )

  ggsave(
    get_save_path(
      "relative_wis_by_location_horizon"
    ),
    rel_wis_by_location_horizon,
    height = 10,
    width = 8
  )
  message("Plotting relative WIS across dates by location and horizon")
  rel_wis_by_location <- relative_wis_by_location(
    score_tables$location |> mutate(horizon = "all horizons")
  )

  ggsave(
    get_save_path(
      "relative_wis_by_location"
    ),
    rel_wis_by_location,
    height = 10,
    width = 4
  )

  message("Done with scoring postprocessing.")
}

p <- arg_parser(paste0(
  "Postprocess a raw score table, creating summary plots ",
  "and tables."
)) |>
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


argv <- parse_args(p)

main(
  argv$path_to_scores,
  argv$output_directory,
  argv$output_prefix
)

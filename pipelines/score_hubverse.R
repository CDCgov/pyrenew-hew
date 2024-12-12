library(argparser)
library(ggplot2)

get_hubverse_table_paths <- function(dir,
                                     disease) {
  path_df <- tibble::tibble(
    path = fs::dir_ls(
      path = dir,
      type = "file",
      glob = glue::glue("*-{disease}-hubverse-table.tsv")
    ),
    disease = disease
  )

  return(path_df)
}


plot_predicted_actual <- function(scoreable_table,
                                  location) {
  to_plot <- scoreable_table |>
    dplyr::filter(
      location == !!location,
      quantile_level %in% c(0.025, 0.5, 0.975)
    ) |>
    tidyr::pivot_wider(
      id_cols = c(
        reference_date,
        target_end_date,
        horizon,
        disease,
        observed
      ),
      names_from = quantile_level,
      names_glue = "q_{quantile_level * 100}",
      values_from = predicted
    )

  plot <- to_plot |>
    ggplot(aes(
      x = target_end_date,
      y = q_50
    )) +
    geom_point(color = "blue") +
    geom_line(
      color = "blue",
      linetype = "dashed"
    ) +
    geom_ribbon(
      aes(
        ymin = q_2.5,
        ymax = q_97.5
      ),
      fill = "blue",
      alpha = 0.5
    ) +
    geom_point(aes(y = observed)) +
    geom_line(aes(y = observed)) +
    facet_wrap(disease ~ horizon) +
    labs(
      title =
        glue::glue("Predictions and observations for {location}"),
      x = "Target date",
      y = "%ED visits"
    ) +
    scale_y_continuous(labels = scales::label_percent()) +
    forecasttools::theme_forecasttools()

  return(plot)
}

plot_predicted_actual_horizons <- function(scoreable_table,
                                           location,
                                           disease) {
  to_plot <- scoreable_table |>
    dplyr::filter(
      location == !!location,
      disease == !!disease
    )

  to_plot_obs <- to_plot |>
    dplyr::filter(
      quantile_level == 0.5,
      horizon == 0
    ) |>
    dplyr::select(target_end_date, observed)

  to_plot_forecast <- to_plot |>
    dplyr::filter(quantile_level %in% c(0.025, 0.5, 0.975)) |>
    tidyr::pivot_wider(
      id_cols = c(
        reference_date,
        target_end_date,
        horizon,
        disease,
        observed
      ),
      names_from = quantile_level,
      names_glue = "q_{quantile_level * 100}",
      values_from = predicted
    )

  plot <- to_plot_forecast |>
    ggplot(aes(
      x = target_end_date,
      y = q_50
    )) +
    geom_point(color = "blue") +
    geom_line(
      color = "blue",
      linetype = "dashed"
    ) +
    geom_ribbon(
      aes(
        ymin = q_2.5,
        ymax = q_97.5
      ),
      fill = "blue",
      alpha = 0.5
    ) +
    geom_point(
      mapping = aes(y = observed),
      data = to_plot_obs
    ) +
    geom_line(
      mapping = aes(y = observed),
      data = to_plot_obs
    ) +
    facet_wrap(~reference_date) +
    labs(
      title =
        glue::glue("Predictions and observations across horizons for {disease} in {location}"),
      x = "Date",
      y = "%ED visits"
    ) +
    scale_y_continuous(labels = scales::label_percent()) +
    forecasttools::theme_forecasttools()

  return(plot)
}

score_and_save <- function(observed_data_path,
                           influenza_table_dir,
                           covid_table_dir,
                           output_dir,
                           last_target_date = NULL,
                           horizons = c(0, 1)) {
  scoring_date <- lubridate::today()

  all_paths <- dplyr::bind_rows(
    get_hubverse_table_paths(
      influenza_table_dir,
      "influenza"
    ),
    get_hubverse_table_paths(
      covid_table_dir,
      "covid-19"
    )
  )

  message(
    "Scoring the following hubverse table files: ",
    all_paths$path,
    "..."
  )

  message(
    "Scoring against observed data from the file: ",
    observed_data_path,
    "..."
  )

  observed_data <- readr::read_tsv(
    observed_data_path,
    show_col_types = FALSE
  )

  if (is.null(last_target_date)) {
    ## if no last target date, use a placeholder
    last_target_date <- lubridate::ymd("9999-01-01")
  }

  read_and_prep_for_scoring <- function(path, disease) {
    disease_short <- dplyr::case_when(
      disease == "covid-19" ~ "covid",
      TRUE ~ disease
    )

    to_score <- readr::read_tsv(
      path,
      show_col_types = FALSE
    ) |>
      dplyr::mutate(disease = !!disease) |>
      dplyr::filter(
        .data$target_end_date <= !!last_target_date,
        .data$horizon %in% !!horizons
      )


    scoreable_table <- if (nrow(to_score) > 0) {
      hewr::to_scoreable_table(
        to_score,
        observed = observed_data,
        observed_value_column =
          glue::glue("prop_{disease_short}"),
        horizons = horizons
      )
    } else {
      NULL
    }

    return(scoreable_table)
  }

  full_scoreable_table <- all_paths |>
    purrr::pmap(read_and_prep_for_scoring) |>
    dplyr::bind_rows()

  full_scores <- hewr::score_hewr(
    full_scoreable_table
  )

  message("Scoring complete.")

  message("Producing summaries and plots...")
  desired_summaries <- list(
    summary_overall = c("horizon", "target"),
    summary_by_epiweek = c(
      "horizon",
      "reference_date",
      "target"
    ),
    summary_by_location = c(
      "horizon",
      "location",
      "target"
    )
  )

  summaries <- desired_summaries |>
    purrr::map(\(x) scoringutils::summarise_scores(full_scores, by = x))


  coverage_figs <- purrr::map(
    c(0.5, 0.95),
    \(x) {
      forecasttools::plot_coverage_by_date(
        full_scores, x
      ) +
        ggplot2::theme_minimal()
    }
  )

  locations <- unique(full_scoreable_table$location)

  pred_actual_figs <- purrr::map(
    locations,
    \(x) plot_predicted_actual(
      full_scoreable_table,
      x
    )
  )

  pred_actual_horizons <- c(
    purrr::map(
      locations,
      \(x) plot_predicted_actual_horizons(
        full_scoreable_table,
        x,
        "covid-19"
      )
    ),
    purrr::map(
      locations,
      \(x) plot_predicted_actual_horizons(
        full_scoreable_table,
        x,
        "influenza"
      )
    )
  )

  make_output_path <- function(output_name,
                               extension) {
    return(fs::path(output_dir,
      glue::glue("{scoring_date}_{output_name}"),
      ext = extension
    ))
  }

  forecasttools::plots_to_pdf(
    coverage_figs,
    make_output_path(
      "coverage",
      "pdf"
    ),
    width = 11,
    height = 8.5
  )
  forecasttools::plots_to_pdf(
    pred_actual_figs,
    make_output_path(
      "predicted_actual",
      "pdf"
    ),
    width = 11,
    height = 8.5
  )

  forecasttools::plots_to_pdf(
    pred_actual_horizons,
    make_output_path(
      "predicted_actual_horizons",
      "pdf"
    ),
    width = 11,
    height = 8.5
  )

  message("Saving summary tables...")
  purrr::walk2(
    summaries,
    names(summaries),
    \(x, y) {
      readr::write_tsv(
        x,
        make_output_path(
          y,
          "tsv"
        )
      )
    }
  )

  message("Done.")
}

p <- arg_parser("Score hubverse tables against observed data.") |>
  add_argument(
    "observed_data_path",
    help = "Path to a .tsv containing observed data."
  ) |>
  add_argument(
    "hubverse_table_dir",
    help = paste0(
      "Directory containing influenza and/or ",
      "COVID-19 forecasts as hubverse tables."
    )
  ) |>
  add_argument(
    "--output-dir",
    help = paste0(
      "Output directory for scores and plots. If not given, ",
      "use the directory in which the script is invoked."
    ),
    default = "."
  ) |>
  add_argument(
    "--last-target-date",
    help = paste0(
      "Only score forecasts up to and including this target ",
      "date, specified in YYYY-MM-DD format. ",
      "If not given, score all available forecasts for ",
      "which there is observed data."
    ),
    default = "9999-01-01"
  ) |>
  add_argument(
    "--horizons",
    help = paste0(
      "Score these horizons, given as a ",
      "whitespace-separated ",
      "string of integers. Default '0 1'."
    ),
    default = "0 1"
  )


argv <- parse_args(p)
score_and_save(
  observed_data_path = argv$observed_data_path,
  influenza_table_dir = argv$hubverse_table_dir,
  covid_table_dir = argv$hubverse_table_dir,
  ## for the CLI, covid and influenza should be
  ## in the same directory
  output_dir = fs::path(argv$output_dir),
  last_target_date = lubridate::ymd(argv$last_target_date),
  horizons = stringr::str_split_1(argv$horizons, " ")
)

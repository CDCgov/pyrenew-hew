library(argparser)
library(ggplot2)
library(ggdist)
library(forecasttools)

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


plot_pred_act_by_horizon <- function(scorable_table,
                                     location) {
  to_plot <- scorable_table |>
    dplyr::filter(
      location == !!location,
      quantile_level %in% c(0.025, 0.25, 0.5, 0.75, 0.975, NA)
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
    geom_pointinterval(
      aes(
        ymin = q_2.5,
        ymax = q_97.5
      ),
      color = "blue",
      alpha = 0.5,
      linewidth = 10,
      point_size = 0
    ) +
    geom_pointinterval(
      aes(
        ymin = q_25,
        ymax = q_75
      ),
      color = "blue",
      point_fill = "darkblue",
      point_color = "black",
      point_alpha = 1,
      alpha = 0.75,
      linewidth = 20,
      point_size = 4,
      shape = 23
    ) +
    geom_line(aes(y = observed),
      size = 1
    ) +
    geom_point(aes(y = observed),
      size = 4,
      shape = 21,
      fill = "darkred",
      color = "black",
      alpha = 1
    ) +
    facet_grid(disease ~ horizon,
      scales = "free_y"
    ) +
    labs(
      title =
        glue::glue("Predictions and observations for {location}"),
      x = "Target date",
      y = "%ED visits"
    ) +
    scale_y_continuous(
      labels = scales::label_percent(),
      transform = "log10"
    ) +
    theme_forecasttools()

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
        .data$target_end_date <= !!last_target_date
      )


    scorable_table <- if (nrow(to_score) > 0) {
      quantile_table_to_scorable(
        to_score,
        observation_table = observed_data,
        obs_value_column =
          glue::glue("prop_{disease_short}"),
        obs_date_column = "reference_date",
        obs_location_column = "location"
      )
    } else {
      NULL
    }

    return(scorable_table)
  }

  full_scorable_table <- all_paths |>
    purrr::pmap(read_and_prep_for_scoring) |>
    dplyr::bind_rows()

  message("Finished reading in forecasts and preparing for scoring.")
  message("Scoring forecasts...")
  full_scores <- hewr::score_hewr(
    full_scorable_table |> dplyr::filter(
      .data$horizon %in% !!horizons
    )
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
    purrr::map(\(x) {
      scoringutils::summarise_scores(full_scores, by = x)
    })


  coverage_figs <- purrr::map(
    c(0.5, 0.95),
    \(x) {
      plot_coverage_by_date(
        full_scores, x
      ) +
        theme_forecasttools()
    }
  )

  locations <- unique(full_scorable_table$location)

  pred_actual_by_horizon <-
    purrr::map(
      locations,
      \(x) {
        plot_pred_act_by_horizon(
          full_scorable_table |>
            dplyr::filter(
              .data$horizon %in% !!horizons
            ),
          x
        )
      }
    )

  pred_act_plot_fn <- function(location, disease) {
    loc_table <- full_scorable_table |>
      dplyr::filter(
        location == !!location,
        disease == !!disease,
        horizon %in% c(-1, !!horizons)
      )

    ## use observed for -1 if no explicit -1 provided
    if (!-1 %in% loc_table$horizon) {
      obs <- loc_table |>
        dplyr::distinct(
          target_end_date, observed, quantile_level,
          .keep_all = TRUE
        ) |>
        dplyr::mutate(
          predicted = .data$observed,
          horizon = -1
        ) |>
        dplyr::select(-"reference_date")
      needed <- loc_table |>
        dplyr::distinct(reference_date) |>
        dplyr::mutate(
          horizon = -1,
          target_end_date = reference_date +
            lubridate::dweeks(.data$horizon)
        )

      synth_minus_one <- dplyr::left_join(needed,
        obs,
        by = c(
          "target_end_date",
          "horizon"
        )
      )
      loc_table <- dplyr::bind_rows(
        synth_minus_one,
        loc_table
      )
    }

    plot <- plot_pred_obs_by_forecast_date(
      loc_table,
      facet_columns = "reference_date"
    ) |>
      suppressMessages()

    return(suppressMessages(plot +
      labs(
        title = glue::glue(
          "Predicted and observed ",
          "{disease} %ED visits in ",
          "{location}."
        ),
        y = "%ED visits"
      ) +
      scale_y_continuous(
        transform = "identity",
        labels = scales::label_percent()
      )))
    ## plot the -1 horizon for pred/obs by forecast date
  }

  pred_act_plot_targets <-
    tidyr::crossing(
      disease = c("covid-19", "influenza"),
      location = locations
    )

  pred_actual_by_date <- purrr::pmap(
    pred_act_plot_targets,
    pred_act_plot_fn
  )

  wis_by_loc <- scoringutils::plot_wis(
    summaries$summary_by_location |>
      dplyr::arrange(horizon, target, wis) |>
      dplyr::mutate(location = factor(
        location,
        levels = unique(
          location
        ),
        ordered = TRUE
      )),
    x = "location"
  ) +
    facet_grid(horizon ~ target)


  make_output_path <- function(output_name,
                               extension) {
    return(fs::path(output_dir,
      glue::glue("{scoring_date}_{output_name}"),
      ext = extension
    ))
  }

  message("Saving WIS component plot...")
  ggsave(
    make_output_path("wis_components", "pdf"),
    wis_by_loc,
    width = 8.5,
    height = 11
  )

  plots_to_pdf(
    coverage_figs,
    make_output_path(
      "coverage",
      "pdf"
    ),
    width = 11,
    height = 8.5
  )
  plots_to_pdf(
    pred_actual_by_horizon,
    make_output_path(
      "predicted_actual_by_horizon",
      "pdf"
    ),
    width = 11,
    height = 8.5
  )

  plots_to_pdf(
    pred_actual_by_date,
    make_output_path(
      "predicted_actual_by_forecast_date",
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

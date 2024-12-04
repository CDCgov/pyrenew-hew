library(argparser)

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


score_and_save <- function(truth_data_path,
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
    truth_data_path,
    "..."
  )

  truth_data <- readr::read_tsv(
    truth_data_path,
    show_col_types = FALSE
  )

  if (is.null(last_target_date)) {
    ## if no last target date, use a placeholder
    last_target_date <- lubridate::ymd("9999-01-01")
  }

  read_and_score <- function(path, disease) {
    disease_short <- dplyr::case_when(
      disease == "covid-19" ~ "covid",
      TRUE ~ disease
    )

    scored <- readr::read_tsv(
      path,
      show_col_types = FALSE
    ) |>
      dplyr::mutate(disease = !!disease) |>
      dplyr::filter(.data$target_end_date <= !!last_target_date) |>
      hewr::score_hubverse(
        observed = truth_data,
        observed_value_column = glue::glue("prop_{disease_short}"),
        horizons = horizons
      )

    return(scored)
  }

  full_scores <- all_paths |>
    purrr::pmap(read_and_score) |>
    dplyr::bind_rows()

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

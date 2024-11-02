library(data.table)
library(argparser)

#' Get all the subdirectories within a parent directory
#' that match the pattern for a forecast run for a
#' given disease and optionally a given report date.
#'
#' @param parent_dir Directory in which to look for forecast subdirectories.
#' @param diseases Names of the diseases to match, as a vector of strings,
#' or a single disease as a string.
#' @return A vector of paths to the forecast subdirectories.
get_all_forecast_dirs <- function(dir_of_forecast_date_dirs,
                                  diseases) {
  # disease names are lowercase by convention
  match_patterns <- stringr::str_c(tolower(diseases), "_r")
  matches_diseases <- \(x) {
    any(stringr::str_starts(x, pattern = match_patterns))
  }

  dirs <- tibble::tibble(
    dir_path =
      fs::dir_ls(dir_of_forecast_date_dirs,
        type = "directory"
      )
  ) |>
    dplyr::mutate(dir_name = fs::path_file(dir_path)) |>
    dplyr::filter(purrr::map_lgl(dir_name, matches_diseases)) |>
    dplyr::pull(dir_path)

  return(dirs)
}

process_loc_date_score_table <- function(model_run_dir) {
  table_path <- fs::path(model_run_dir,
    "score_table",
    ext = "rds"
  )
  location <- fs::path_file(model_run_dir)
  if (!(fs::file_exists(table_path))) {
    warning(glue::glue(
      "No `score_table.rds` found for location ",
      "{location} in directory {model_run_dir}"
    ))
    return(NULL)
  }
  score_table <- readRDS(table_path)
  score_table$quantile_scores$location <- location
  score_table$sample_scores$location <- location
  return(score_table)
}

process_date_score_table <- function(date_fit_dir) {
  table_path <- fs::path(date_fit_dir,
    "score_table",
    ext = "rds"
  )
  table_dir <- fs::path_file(date_fit_dir)

  report_date <- stringr::str_match(
    table_dir,
    "r_(([0-9]|-)+)_f"
  )[2] |>
    lubridate::ymd()

  score_table <- readRDS(table_path)
  score_table$quantile_scores$report_date <- report_date
  score_table$sample_scores$report_date <- report_date
  return(score_table)
}

bind_tables <- function(list_of_table_pairs) {
  sample_metrics <- purrr::map(
    list_of_table_pairs,
    \(x) {
      attr(x$sample_scores, "metrics")
    }
  ) |>
    unlist() |>
    unique()
  quantile_metrics <- purrr::map(
    list_of_table_pairs,
    \(x) {
      attr(x$quantile_scores, "metrics")
    }
  ) |>
    unlist() |>
    unique()

  sample_scores <- purrr::map(
    list_of_table_pairs,
    \(x) {
      x$sample_scores
    }
  ) |>
    data.table::rbindlist(fill = TRUE)

  quantile_scores <- purrr::map(
    list_of_table_pairs,
    \(x) {
      x$quantile_scores
    }
  ) |>
    data.table::rbindlist(fill = TRUE)

  attr(sample_scores, "metrics") <- sample_metrics
  attr(quantile_scores, "metrics") <- quantile_metrics


  return(list(
    sample_scores = sample_scores,
    quantile_scores = quantile_scores
  ))
}

collate_scores_for_date <- function(model_run_dir,
                                    score_file_name = "score_table",
                                    score_file_ext = "rds",
                                    save = FALSE) {
  message(glue::glue("Processing scores from {model_run_dir}..."))
  locations_to_process <- fs::dir_ls(model_run_dir,
    type = "directory"
  )
  location_score_tables <- purrr::map(
    locations_to_process,
    process_loc_date_score_table
  )

  date_score_table <- bind_tables(location_score_tables)

  if (save) {
    save_path <- fs::path(model_run_dir,
      score_file_name,
      ext = score_file_ext
    )
    message(glue::glue("Saving score table to {save_path}..."))
    saveRDS(date_score_table, save_path)
  }
  message(glue::glue("Done processing scores for {model_run_dir}."))
  return(date_score_table)
}


collate_all_score_tables <- function(model_base_dir,
                                     disease,
                                     score_file_name = "score_table",
                                     score_file_ext = "rds",
                                     save = FALSE) {
  date_dirs_to_process <- get_all_forecast_dirs(
    model_base_dir,
    diseases = disease
  )

  # collate scores across locations for each date
  purrr::map(date_dirs_to_process,
    collate_scores_for_date,
    save = save
  )

  # get all dates, annotate, and combine
  message(paste0(
    "Combining date-specific score tables ",
    "to create a full score table..."
  ))
  date_tables <- purrr::map(
    date_dirs_to_process,
    process_date_score_table
  )

  full_score_table <- bind_tables(date_tables)

  if (save) {
    save_path <- fs::path(model_base_dir,
      score_file_name,
      ext = score_file_ext
    )
    message(glue::glue("Saving full score table to {save_path}..."))
    saveRDS(full_score_table, save_path)
  }

  message("Done creating full score table")

  return(full_score_table)
}


p <- arg_parser(
  "Forecast other (non-target-disease) ED visits for a given location."
) |>
  add_argument(
    "model_base_dir",
    help = paste0(
      "Base directory containing subdirectories that represent ",
      "individual forecast dates, each of which in turn has ",
      "subdirectories that represent individual location forecasts."
    )
  ) |>
  add_argument(
    "disease",
    help = paste0(
      "Name of the disease for which to collate scores."
    )
  )

argv <- parse_args(p)

collate_all_score_tables(
  argv$model_base_dir,
  argv$disease,
  score_file_name = glue::glue("{argv$disease}_score_table"),
  score_file_ext = "rds",
  save = TRUE
)

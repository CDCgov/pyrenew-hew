#' Utilities for handling and parsing directory names
#' based on pyrenew-hew pipeline conventions.

disease_map_lower <- list(
  "covid-19" = "COVID-19",
  "influenza" = "Influenza"
)

#' Parse model batch directory name.
#'
#' Parse the name of a model batch directory
#' (i.e. a directory representing a single
#' report date and disease pair, but potentially
#' with fits for multiple locations), returning
#' a named list of quantities of interest.
#'
#' @param model_batch_dir_name Name of the model batch
#' directory (not the full path to it, just the directory
#' base name) to parse.
#' @return A list of quantities: `disease`, `report_date`,
#' `first_training_date`, and `last_training_date`.
#' @export
parse_model_batch_dir_name <- function(model_batch_dir_name) {
  pattern <- "(.+)_r_(.+)_f_(.+)_t_(.+)"

  matches <- stringr::str_match(
    model_batch_dir_name,
    pattern
  )

  if (is.na(matches[1])) {
    stop(
      "Invalid format for model batch directory name; ",
      "could not parse. Expected ",
      "'<disease>_r_<report_date>_f_<first_training_date>_t_",
      "<last_training_date>'."
    )
  }

  return(list(
    disease = disease_map_lower[[matches[2]]],
    report_date = lubridate::ymd(matches[3]),
    first_training_date = lubridate::ymd(matches[4]),
    last_training_date = lubridate::ymd(matches[5])
  ))
}

#' Parse model run directory path.
#'
#' Parse path to a model run directory
#' (i.e. a directory representing a run for a
#' particular location, disease, and reference
#' date, and extract key quantities of interest.
#'
#' @param model_run_dir_path Path to parse.
#' @return A list of parsed attributes:
#' `location`, `disease`, `report_date`,
#' `first_training_date`, and `last_training_date`.
#'
#' @export
parse_model_run_dir_path <- function(model_run_dir_path) {
  batch_dir <- fs::path_dir(model_run_dir_path) |>
    fs::path_file()
  location <- fs::path_file(model_run_dir_path)

  return(c(
    list(location = location),
    parse_model_batch_dir(batch_dir)
  ))
}


#' Get forecast directories.
#'
#' Get all the subdirectories within a parent directory
#' that match the pattern for a forecast run for a
#' given disease and optionally a given report date.
#'
#' @param dir_of_forecast_dirs Directory in which to look for
#' subdirectories representing individual forecast date / pathogen /
#' dataset combinations.
#' @param diseases Names of the diseases to match, as a vector of strings,
#' or a single disease as a string.
#' @return A vector of paths to the forecast subdirectories.
get_all_forecast_dirs <- function(dir_of_forecast_dirs,
                                  diseases) {
  # disease names are lowercase by convention
  match_patterns <- stringr::str_c(tolower(diseases),
    "_r",
    collapse = "|"
  )

  dirs <- tibble::tibble(
    dir_path = fs::dir_ls(
      dir_of_forecast_dirs,
      type = "directory"
    )
  ) |>
    dplyr::filter(str_starts(
      fs::path_file(dir_path),
      match_patterns
    )) |>
    dplyr::pull(dir_path)

  return(dirs)
}

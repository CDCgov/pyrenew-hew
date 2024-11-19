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
#' @param model_batch_dir_path Path to the model batch
#' directory to parse. Will parse only the basename.
#' @return A list of quantities: `disease`, `report_date`,
#' `first_training_date`, and `last_training_date`.
#' @export
parse_model_batch_dir_path <- function(model_batch_dir_path) {
  pattern <- "(.+)_r_(.+)_f_(.+)_t_(.+)"
  model_batch_dir_name <- fs::path_file(model_batch_dir_path)
  matches <- stringr::str_match(
    model_batch_dir_name,
    pattern
  )

  if (any(is.na(matches))) {
    stop(
      "Invalid format for model batch directory name; ",
      "could not parse. Expected ",
      "'<disease>_r_<report_date>_f_<first_training_date>_t_",
      "<last_training_date>'."
    )
  }

  result <- list(
    disease = disease_map_lower[[matches[2]]],
    report_date = lubridate::ymd(matches[3], quiet = TRUE),
    first_training_date = lubridate::ymd(matches[4], quiet = TRUE),
    last_training_date = lubridate::ymd(matches[5], quiet = TRUE)
  )

  if (any(sapply(result, is.null)) || any(sapply(result, is.na))) {
    stop(
      "Could not parse extracted disease and/or date ",
      "values expected 'disease' to be one of 'covid-19' ",
      "and 'influenza' and all dates to be valid dates in ",
      "YYYY-MM-DD format. Got: ",
      glue::glue(
        "disease: {matches[[2]]}, ",
        "report_date: {matches[[3]]}, ",
        "first_training_date: {matches[[4]]}, ",
        "last_training_date: {matches[[5]]}."
      )
    )
  }

  return(result)
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
    location = location,
    parse_model_batch_dir_path(batch_dir)
  ))
}


#' Get forecast directories.
#'
#' Get all the subdirectories within a parent directory
#' that match the pattern for a forecast run for a
#' given disease and optionally a given report date.
#'
#' @param dir_of_batch_dirs Directory in which to look for
#' "model batch" directories, each of which represents an
#' individual forecast date / pathogen / dataset combination.
#' @param diseases Names of the diseases to match, as a vector of strings,
#' or a single disease as a string.
#' @return A vector of paths to the forecast subdirectories.
#' @export
get_all_model_batch_dirs <- function(dir_of_batch_dirs,
                                     diseases) {
  # disease names are lowercase by convention
  match_patterns <- stringr::str_c(tolower(diseases),
    "_r",
    collapse = "|"
  )

  dirs <- tibble::tibble(
    dir_path = fs::dir_ls(
      dir_of_batch_dirs,
      type = "directory"
    )
  ) |>
    dplyr::filter(stringr::str_starts(
      fs::path_file(dir_path),
      match_patterns
    )) |>
    dplyr::pull(dir_path)

  return(dirs)
}

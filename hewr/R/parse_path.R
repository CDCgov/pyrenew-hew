disease_map_lower <- list(
  "covid-19" = "COVID-19",
  "influenza" = "Influenza"
)

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
parse_model_batch_dir <- function(model_batch_dir_name) {
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
parse_model_run_dir <- function(model_run_dir_path) {
  batch_dir <- fs::path_dir(model_run_dir_path) |>
    fs::path_file()
  location <- fs::path_file(model_run_dir_path)

  return(c(
    list(location = location),
    parse_model_batch_dir(batch_dir)
  ))
}

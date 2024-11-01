library(data.table)
library(argparser)

get_all_flu_forecast_dirs <- function(dir_of_forecast_date_dirs) {
  dirs <- tibble::tibble(
    dir_path = fs::dir_ls(dir_of_forecast_date_dirs,
      type = "directory"
    )
  ) |>
    dplyr::mutate(
      dir_name = fs::path_file(dir_path)
    ) |>
    dplyr::filter(stringr::str_starts(
      dir_name,
      "influenza_r"
    )) |>
    dplyr::pull(dir_path)

  return(dirs)
}

process_loc_date_score_table <- function(model_run_dir) {
  table_path <- fs::path(model_run_dir,
    "score_table",
    ext = "rds"
  )
  location <- fs::path_file(model_run_dir)
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
  table_dir <- fs::path_file(data_fit_dir)

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
  sample_table <- purrr::map(
    \(x) {
      x$sample_scores
    },
    list_of_table_pairs
  ) |>
    data_table::rbindlist()

  quantile_table <- purrr::map(
    \(x) {
      x$quantile_scores
    },
    list_of_table_pairs
  ) |>
    data_table::rbindlist()

  return(list(
    sample_table = sample_table,
    quantile_table = quantile_table
  ))
}

process_all_locations <- function(model_base_dir,
                                  score_file_name = "score_table",
                                  score_file_ext = ".rds",
                                  save = TRUE) {
  to_process <- fs::dir_ls(model_base_dir,
    type = "directory"
  )
  location_tables <- purrr::map(
    to_process,
    process_loc_date_score_table
  )

  result <- bind_tables(location_tables)

  if (save) {
    save_path <- fs::path(model_base_dir,
      score_file_name,
      ext = score_file_ext
    )
    message(glue::glue("Saving score table to {save_path}"))
    saveRDS(result, save_path)
  }

  return(result)
}


process_all_dates <- function(dir_of_forecast_date_dirs,
                              score_file_name = "score_table",
                              score_file_ext = ".rds",
                              save = TRUE,
                              verbose = TRUE) {
  to_process <- get_all_flu_forecast_dirs(
    dir_of_forecast_date_dirs
  )

  date_tables <- purrr::map(
    to_process,
    process_date_score_table
  )

  result <- bind_tables(date_tables)

  if (save) {
    save_path <- fs::path(model_base_dir,
      score_file_name,
      ext = score_file_ext
    )
    message(glue::glue("Saving score table to {save_path}"))
    saveRDS(result, save_path)
  }

  return(result)
}


collate_all <- function(dir_of_forecast_date_dirs,
                        save = FALSE) {
  dirs_to_process <- get_all_flu_forecast_dirs(
    dir_of_forecast_date_dirs
  )

  purrr::map(dirs_to_process,
    process_all_locations,
    save = save
  )

  collated <- purrr::map(dir_of_forecast_date_dirs,
    process_all_dates,
    save = save
  )

  return(collated)
}


p <- arg_parser(
  "Forecast other (non-target-disease) ED visits for a given location."
) |>
  add_argument(
    "dir-of-forecast-date-dirs",
    help = paste0(
      "Top-level containing a number of ",
      "sub-directories, each of which ",
      "represents forecasts for a single ",
      "forecast date across locations"
    )
  )

argv <- parse_args(p)

collate_all(argv$dir_of_forecast_date_dirs, save = TRUE)

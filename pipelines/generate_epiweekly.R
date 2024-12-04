script_packages <- c(
  "arrow",
  "argparser",
  "dplyr",
  "forecasttools",
  "fs",
  "readr"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})

#' Convert Daily Data to Epiweekly Data
#'
#' This function reads daily data from a CSV file, converts it to epiweekly
#' data, and writes the resulting epiweekly data to a new CSV file.
#'
#' @param model_run_dir A string specifying the directory containing the model
#'  run data.
#' @param dataname A string specifying the name of the daily data file. Default
#'
#' @param strict A logical value indicating whether to enforce strict inclusion
#' of only full epiweeks. Default is TRUE.
#' @param day_of_week An integer specifying the day of the week to use for the
#' epiweek date. Default is 1 (Monday).
#'
#' @return None. The function writes the epiweekly data to a CSV file in the
#'  specified directory.
convert_daily_to_epiweekly <- function(
    model_run_dir, dataname = "data.csv",
    strict = TRUE, day_of_week = 7) {
  ext <- path_ext(dataname)
  data_basename <- path_ext_remove(dataname)
  if (!ext %in% c("csv", "tsv")) {
    stop("Invalid file extension. Only 'csv' and 'tsv' are allowed.")
  }

  delim <- if (ext == "csv") "," else "\t"
  message(glue::glue("Generating epi-weekly data {model_run_dir}..."))

  data_path <- path(model_run_dir, dataname)

  daily_data <- read_delim(
    data_path,
    delim = delim,
    col_types = cols(
      disease = col_character(),
      data_type = col_character(),
      ed_visits = col_double(),
      date = col_date()
    )
  ) |>
    mutate(.draw = 1)

  epiweekly_data <- daily_data |>
    group_by(disease) |>
    group_modify(~ forecasttools::daily_to_epiweekly(.x,
      value_col = "ed_visits", weekly_value_name = "ed_visits",
      strict = strict
    )) |>
    ungroup() |>
    mutate(date = epiweek_to_date(epiweek, epiyear,
      day_of_week = day_of_week
    )) |>
    select(date, disease, ed_visits) |>
    inner_join(daily_data |> select(date, disease, data_type),
      by = c("date", "disease")
    )
  # epiweek end date determines data_type classification

  output_file <- path(
    model_run_dir,
    glue::glue("epiweekly_{data_basename}"),
    ext = ext
  )

  write_delim(epiweekly_data, output_file, delim = delim)
}

main <- function(model_run_dir) {
  convert_daily_to_epiweekly(model_run_dir, dataname = "data.csv")
  convert_daily_to_epiweekly(model_run_dir, dataname = "eval_data.tsv")
}

# Create a parser
p <- arg_parser("Create epiweekly data") |>
  add_argument(
    "model_run_dir",
    help = "Directory containing the model data and output."
  )

argv <- parse_args(p)

main(argv$model_run_dir)

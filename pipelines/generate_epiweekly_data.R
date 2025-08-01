script_packages <- c(
  "argparser",
  "dplyr",
  "forecasttools",
  "fs",
  "readr",
  "lubridate",
  "stringr"
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
#' @param data_name A string specifying the name of the daily data file. Default
#'
#' @param strict A logical value indicating whether to enforce strict inclusion
#' of only full epiweeks. Default is TRUE.
#' @param day_of_week An integer specifying the day of the week to use for the
#' epiweek date. Default is 1 (Monday).
#'
#' @return None. The function writes the epiweekly data to a CSV file in the
#'  specified directory.
convert_daily_to_epiweekly <- function(
  model_run_dir,
  data_name,
  strict = TRUE,
  day_of_week = 7
) {
  data_path <- path(model_run_dir, "data", data_name)

  daily_data <- read_tsv(
    data_path,
    col_types = cols(
      date = col_date(),
      geo_value = col_character(),
      disease = col_character(),
      data_type = col_character(),
      .variable = col_character(),
      .value = col_double()
    )
  )

  daily_ed_data <- daily_data |>
    filter(str_ends(.variable, "_ed_visits"))

  epiweekly_hosp_data <- daily_data |>
    filter(.variable == "observed_hospital_admissions")

  epiweekly_ed_data <- daily_ed_data |>
    forecasttools::daily_to_epiweekly(
      value_col = ".value",
      weekly_value_name = ".value",
      id_cols = c("geo_value", "disease", "data_type", ".variable"),
      strict = strict
    ) |>
    mutate(
      date = epiweek_to_date(epiweek, epiyear, day_of_week = day_of_week)
    ) |>
    select(date, geo_value, disease, data_type, .variable, .value)

  epiweekly_data <- bind_rows(epiweekly_ed_data, epiweekly_hosp_data) |>
    arrange(date, .variable)

  output_file <- path(
    model_run_dir,
    "data",
    glue::glue("epiweekly_{data_name}")
  )

  write_tsv(epiweekly_data, output_file)
}

main <- function(model_run_dir) {
  convert_daily_to_epiweekly(
    model_run_dir,
    data_name = "combined_training_data.tsv"
  )
}

# Create a parser
p <- arg_parser("Create epiweekly data") |>
  add_argument(
    "model_run_dir",
    help = "Directory containing the model data and output."
  )

argv <- parse_args(p)
main(argv$model_run_dir)

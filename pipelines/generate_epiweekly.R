script_packages <- c(
  "arrow",
  "argparser",
  "dplyr",
  "forecasttools",
  "fs",
  "readr",
  "lubridate"
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
    model_run_dir, dataname = "data.tsv",
    strict = TRUE, day_of_week = 7) {
  ext <- path_ext(dataname)
  data_basename <- path_ext_remove(dataname)
  if (!ext %in% c("csv", "tsv")) {
    stop("Invalid file extension. Only 'csv' and 'tsv' are allowed.")
  }

  delim <- if (ext == "csv") "," else "\t"
  message(glue::glue("Generating epi-weekly data {model_run_dir}..."))

  data_path <- path(model_run_dir, "data", dataname)

  daily_data <- read_delim(
    data_path,
    delim = delim,
    col_types = cols(
      disease = col_character(),
      data_type = col_character(),
      ed_visits = col_double(),
      date = col_date()
    )
  )

  epiweekly_data <- daily_data |>
    forecasttools::daily_to_epiweekly(
      value_col = "ed_visits",
      weekly_value_name = "ed_visits",
      id_cols = c("disease", "geo_value"),
      strict = strict
    ) |>
    mutate(date = epiweek_to_date(epiweek,
      epiyear,
      day_of_week = day_of_week
    )) |>
    select(date, disease, ed_visits, geo_value) |>
    inner_join(daily_data |> select(date, disease, data_type, geo_value),
      by = c("date", "disease", "geo_value")
    )
  # epiweek end date determines data_type classification

  output_file <- path(
    model_run_dir, "data",
    glue::glue("epiweekly_{data_basename}"),
    ext = ext
  )

  write_delim(epiweekly_data, output_file, delim = delim)
}

convert_comb_daily_to_ewkly <- function(
    model_run_dir, dataname = "combined_training_data.tsv",
    strict = FALSE, day_of_week = 7) {
  message(glue::glue("Generating epi-weekly data {model_run_dir}..."))

  data_basename <- path_ext_remove(dataname)
  data_path <- path(model_run_dir, "data", dataname)

  raw_data <- read_tsv(
    data_path,
    col_types = cols(
      date = col_date(),
      geo_value = col_character(),
      disease = col_character(),
      data_type = col_character(),
      value_type = col_character(),
      value = col_double()
    )
  )

  daily_ed_visit_data <- raw_data |>
    filter(value_type == "ed_visits")

  ewkly_hospital_admission_data <- raw_data |>
    filter(value_type == "hospital_admissions") |>
    mutate(
      epiweek = epiweek(date),
      epiyear = epiyear(date)
    )

  # Verify hospital admissions dates are epiweekly
  invalid_dates <-
    ewkly_hospital_admission_data |>
    mutate(implied_date = epiweek_to_date(epiweek,
      epiyear,
      day_of_week = day_of_week
    )) |>
    filter(date != implied_date) |>
    pull(date)

  if (length(invalid_dates) > 0) {
    stop(glue::glue(
      "Invalid dates found in hospital admissions data: ",
      "{paste0(invalid_dates, collapse = ', ')}"
    ))
  }

  epiweekly_ed_visit_data <- daily_ed_visit_data |>
    forecasttools::daily_to_epiweekly(
      value_col = "value",
      weekly_value_name = "value",
      id_cols = c("disease", "geo_value", "data_type", "value_type"),
      strict = strict
    ) |>
    mutate(date = epiweek_to_date(epiweek,
      epiyear,
      day_of_week = day_of_week
    ))

  epiweekly_data <- bind_rows(
    epiweekly_ed_visit_data,
    ewkly_hospital_admission_data
  ) |>
    arrange(date, value_type, disease) |>
    select(date, everything())

  output_file <- path(
    model_run_dir, "data",
    glue::glue("epiweekly_{data_basename}"),
    ext = "tsv"
  )

  write_tsv(epiweekly_data, output_file)
}


convert_if_exists <- function(model_run_dir,
                              dataname) {
  exists <- fs::file_exists(fs::path(
    model_run_dir,
    dataname
  ))
  convert_fn <- ifelse(
    "combined" %in% dataname,
    convert_comb_daily_to_ewkly,
    convert_daily_to_epiweekly
  )

  return(if (exists) {
    convert_fn(model_run_dir,
      dataname = dataname
    )
  } else {
    NULL
  })
}


main <- function(model_run_dir) {
  datanames <- c(
    "data.tsv",
    "eval_data.tsv",
    "combined_training_data.tsv",
    "combined_eval_data.tsv"
  )
  purrr::map(
    datanames,
    \(x) convert_if_exists(model_run_dir, dataname = x)
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

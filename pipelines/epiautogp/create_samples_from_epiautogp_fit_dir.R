library(argparser)
library(tidyverse)
library(glue)

# put this list in hewr
required_columns_e <- c(
  ".chain",
  ".iteration",
  ".draw",
  "date",
  "geo_value",
  "disease",
  ".variable",
  ".value",
  "resolution",
  "aggregated_numerator",
  "aggregated_denominator"
)

create_samples_from_epiautogp_fit_dir <- function(model_fit_dir) {
  model_name <- fs::path_file(model_fit_dir)

  # Determine the frequency and target from model name
  frequency <- if_else(
    str_detect(model_name, "epiweekly"),
    "epiweekly",
    "daily"
  )

  # Determine target letter (h for NHSN, e for NSSP)
  # this is a bit over-engineered. Epiautogp samples could presumably just stored as epiautogp_samples.parquet?
  target_letter <- if (stringr::str_detect(model_name, "nhsn")) {
    "h"
  } else if (stringr::str_detect(model_name, "nssp")) {
    "e"
  } else {
    stop("Cannot determine target type from model name: ", model_name)
  }

  samples_file <- glue("{frequency}_epiautogp_samples_{target_letter}.parquet")

  # Read EpiAutoGP samples
  samples_path <- fs::path(model_fit_dir, samples_file)

  epiautogp_samples <- forecasttools::read_tabular(samples_path)

  # EpiAutoGP output already has all required columns from Julia:
  # date, .value, .draw, .variable, resolution, geo_value, disease
  # Just need to add aggregated_numerator and aggregated_denominator
  samples_tidy <- epiautogp_samples |>
    dplyr::mutate(
      date = lubridate::as_date(.data$date),
      aggregated_numerator = FALSE,
      aggregated_denominator = NA
    ) |>
    dplyr::select(tidyselect::any_of(required_columns_e))

  forecasttools::write_tabular(
    samples_tidy,
    fs::path(
      model_fit_dir,
      "samples",
      ext = "parquet"
    )
  )
}

p <- arg_parser(
  "Create samples file from an EpiAutoGP model fit directory."
) |>
  add_argument(
    "model-fit-dir",
    help = "Directory containing the model data and output.",
  )

argv <- parse_args(p)

create_samples_from_epiautogp_fit_dir(model_fit_dir = argv$model_fit_dir)

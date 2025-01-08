library(argparser)
library(fs)
library(tidyverse)
p <- arg_parser("Process model batch directories")
p <- add_argument(p, "super_dir", help = "Directory containing model batch directories")
argv <- parse_args(p)
super_dir <- path(argv$super_dir)

source("hewr/R/to_epiweekly_quantile_table.R")

save_hubverse_table <- function(model_batch_dir) {
  model_runs_path <- fs::path(model_batch_dir, "model_runs")

  draws_file_name <- "daily_samples"

  locations_to_process <- fs::dir_ls(model_runs_path,
    type = "directory"
  ) |>
    purrr::discard(~ fs::path_file(.x) %in% exclude)

  batch_params <- hewr::parse_model_batch_dir_path(
    model_batch_dir
  )
  report_date <- batch_params$report_date
  disease <- batch_params$disease
  disease_abbr <- dplyr::case_when(
    disease == "Influenza" ~ "flu",
    disease == "COVID-19" ~ "covid",
    TRUE ~ disease
  )

  report_epiweek <- lubridate::epiweek(report_date)
  report_epiyear <- lubridate::epiyear(report_date)
  report_epiweek_end <- forecasttools::epiweek_to_date(
    report_epiweek,
    report_epiyear,
    day_of_week = 7
  )

  hubverse_table <- purrr::map(
    locations_to_process,
    \(x) {
      to_epiweekly_quantiles(
        x,
        report_date = report_date,
        max_lookback_days = 8,
        disease_name = "observed_hospital_admissions",
        draws_file_name = draws_file_name,
        disease_model_name = "pyrenew_h",
        strict = strict
      )
    }
    ## max_lookback_days = 8 ensures we get
    ## the full -1 horizon but do not waste
    ## time quantilizing draws that will not
    ## be included in the final table.
  ) |>
    dplyr::bind_rows() |>
    forecasttools::get_hubverse_table(
      report_epiweek_end,
      target_name =
        glue::glue("wk inc {disease_abbr} prop ed visits")
    ) |>
    dplyr::arrange(
      .data$target,
      .data$output_type,
      .data$location,
      .data$reference_date,
      .data$horizon,
      .data$output_type_id
    )
  hubverse_file_name <- path(glue::glue("{report_date}-{str_to_lower(disease)}-hubverse-table"), ext = "tsv")
  write_tsv(hubverse_table, path(model_batch_dir, hubverse_file_name))
}


exclude <- character(0)

model_batch_dirs <- dir_ls(super_dir)
walk(model_batch_dirs, save_hubverse_table)

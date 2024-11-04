draws_to_quantiles <- function(forecast_dir) {
  draws_path <- fs::path(forecast_dir,
    "forecast_samples",
    ext = "parquet"
  )
  location <- fs::path_file(forecast_dir)
  draws <- arrow::read_parquet(draws_path) |>
    dplyr::filter(disease == "prop_disease_ed_visits")

  epiweekly_quantiles <-
    forecasttools::daily_to_epiweekly(
      date_col = "date",
      value_col = ".value",
      id_cols = ".draw",
      weekly_value_col = "value"
    ) |>
    forecasttools::trajectories_to_quantiles(
      timepoint_cols = c("epiweek", "epiyear"),
      value_col = "value"
    ) |>
    dplyr::mutate(
      location = !!location
    )


  return(epiweekly_quantiles)
}

create_hubverse_table <- function(model_run_dir) {
  locations_to_process <- fs::dir_ls(model_run_dir,
    type = "directory"
  )

  report_date <- stringr::str_match(
    model_run_dir,
    "r_(([0-9]|-)+)_f"
  )[2]

  disease <- dplyr::case_when(
    stringr::str_starts(
      model_run_dir,
      "covid-19"
    ) ~ "covid",
    stringr::str_starts(
      model_run_dir,
      "influenza"
    ) ~ "flu",
    TRUE ~ NA
  )

  hubverse_table <- purrr::map(
    locations_to_process,
    draws_to_quantiles
  ) |>
    dplyr::bind_rows() |>
    forecasttools::get_flusight_table(
      report_date
    ) |>
    dplyr::mutate(
      target =
        glue::glue("wk inc {disease} prop ed visits")
    ) |>
    dplyr::arrange(
      target,
      output_type,
      location,
      reference_date,
      horizon,
      output_type_id
    )

  return(hubverse_table)
}


main <- function(model_run_dir,
                 output_path) {
  creat_hubverse_table(model_run_dir) |>
    readr::write_tsv(output_path)
}


p <- argparser::arg_parser(
  "Create a hubverse table from location specific forecast draws."
) |>
  argparser::add_argument(
    "model_run_dir",
    help = paste0(
      "Directory containing subdirectories that represent ",
      "individual forecast locations, with a directory name ",
      "that indicates the target pathogen and reference date"
    )
  ) |>
  argparser::add_argument(
    output_path,
    help = "path to which to save the table"
  )

argv <- argparser::parse_args(p)

main(
  argv$model_run_dir,
  arvg$output_path
)

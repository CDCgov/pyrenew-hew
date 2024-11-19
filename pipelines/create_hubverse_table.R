draws_to_quantiles <- function(forecast_dir,
                               report_date,
                               max_lookback_days) {
  message(glue::glue("Processing {forecast_dir}..."))
  draws_path <- fs::path(forecast_dir,
    "forecast_samples",
    ext = "parquet"
  )
  location <- fs::path_file(forecast_dir)

  draws <- arrow::read_parquet(draws_path) |>
    dplyr::filter(date >= lubridate::ymd(report_date) -
      lubridate::days(max_lookback_days))

  if (nrow(draws) < 1) {
    return(NULL)
  }

  epiweekly_disease_draws <- draws |>
    dplyr::filter(
      disease == "Disease"
    ) |>
    forecasttools::daily_to_epiweekly(
      date_col = "date",
      value_col = ".value",
      id_cols = ".draw",
      weekly_value_name = "epiweekly_disease",
      strict = TRUE
    )

  epiweekly_total_draws <- draws |>
    dplyr::filter(disease == "Other") |>
    forecasttools::daily_to_epiweekly(
      date_col = "date",
      value_col = ".value",
      id_cols = ".draw",
      weekly_value_name = "epiweekly_total",
      strict = TRUE
    )

  epiweekly_prop_draws <- dplyr::inner_join(
    epiweekly_disease_draws,
    epiweekly_total_draws,
    by = c(
      "epiweek",
      "epiyear",
      ".draw"
    )
  ) |>
    dplyr::mutate(
      epiweekly_proportion =
        epiweekly_disease / epiweekly_total
    )


  epiweekly_quantiles <- epiweekly_prop_draws |>
    forecasttools::trajectories_to_quantiles(
      timepoint_cols = c("epiweek", "epiyear"),
      value_col = "epiweekly_proportion"
    ) |>
    dplyr::mutate(
      location = !!location
    )

  message(glue::glue("Done processing {forecast_dir}"))
  return(epiweekly_quantiles)
}

create_hubverse_table <- function(model_batch_dir,
                                  exclude = NULL) {
  locations_to_process <- fs::dir_ls(model_batch_dir,
    type = "directory"
  )

  if (!is.null(exclude)) {
    locations_to_process <- locations_to_process[
      !(fs::path_file(locations_to_process) %in% exclude)
    ]
  }

  batch_params <- hewr::parse_model_batch_dir_name(model_batch_dir)
  report_date <- batch_params$report_date
  disease <- batch_params$disease

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
      draws_to_quantiles(
        x,
        report_date = report_date,
        max_lookback_days = 8
      )
    }
    ## ensures we get the full -1 horizon but do not
    ## waste time quantilizing draws that will not be
    ## included in the final table.
  ) |>
    dplyr::bind_rows() |>
    forecasttools::get_hubverse_table(
      report_epiweek_end,
      target_name =
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


main <- function(model_batch_dir,
                 output_path,
                 exclude = NULL) {
  create_hubverse_table(model_batch_dir, exclude = exclude) |>
    readr::write_tsv(output_path)
}


p <- argparser::arg_parser(
  "Create a hubverse table from location specific forecast draws."
) |>
  argparser::add_argument(
    "model_batch_dir",
    help = paste0(
      "Directory containing subdirectories that represent ",
      "individual forecast locations, with a directory name ",
      "that indicates the target pathogen and reference date"
    )
  ) |>
  argparser::add_argument(
    "output_path",
    help = "path to which to save the table"
  ) |>
  argparser::add_argument(
    "--exclude",
    help = "locations to exclude, as a whitespace-separated string",
    default = ""
  )
argv <- argparser::parse_args(p)

main(
  argv$model_batch_dir,
  argv$output_path,
  stringr::str_split_1(argv$exclude, " ")
)

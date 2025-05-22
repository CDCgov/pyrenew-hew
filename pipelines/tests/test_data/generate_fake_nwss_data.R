base_dir <- "pipelines/tests/end_to_end_test_output/private_data/covid-19_r_2024-12-21_f_2024-10-22_t_2024-12-20/model_runs/" # nolint
loc_params <- tibble::tibble(
  loc_abb = c("CA", "MT"),
  loc_offset = c(10, 0), # add to lab_site_index for uniqueness
)

get_nwss_data_from_posterior <- function(
  loc_abb,
  loc_offset
) {
  model_run_dir <- fs::path(base_dir, loc_abb)
  model_info <- hewr::parse_model_run_dir_path(model_run_dir)
  pyrenew_model_name <- "pyrenew_hew"
  dat_path <- fs::path(model_run_dir, "data", "data_for_model_fit.json")
  data_for_model_fit <- readr::read_lines(dat_path) |>
    stringr::str_replace_all("-Infinity", "null") |>
    jsonlite::fromJSON()

  first_nhsn_date <- data_for_model_fit$nhsn_training_dates[[1]]
  first_nssp_date <- data_for_model_fit$nssp_training_dates[[1]]
  first_nwss_date <- ifelse(
    !is.null(data_for_model_fit$data_observed_disease_wastewater),
    min(unlist(data_for_model_fit$data_observed_disease_wastewater$date)),
    NA
  )
  nhsn_step_size <- data_for_model_fit$nhsn_step_size

  pyrenew_posterior <-
    arrow::read_parquet(
      fs::path(
        model_run_dir,
        pyrenew_model_name,
        "mcmc_tidy",
        "pyrenew_posterior",
        ext = "parquet"
      )
    )
  var_name <- "site_level_log_ww_conc[group_time_index,lab_site_index]"

  site_info <- tibble::tibble(
    wwtp_id = unique(
      data_for_model_fit$data_observed_disease_wastewater$lab_site_index
    ) +
      loc_offset,
    population_served = unique(
      data_for_model_fit$data_observed_disease_wastewater$site_pop
    ),
    quality_flag = c("no", NA_character_, "n", "n")
  ) |>
    dplyr::mutate(
      lod_sewage = abs(rnorm(dplyr::n(), mean = 5, sd = 1))
    )

  loc_ww_conc_data <- pyrenew_posterior |>
    tidybayes::gather_draws(!!rlang::parse_expr(var_name)) |>
    dplyr::ungroup() |>
    dplyr::mutate(
      date = hewr::group_time_index_to_date(
        group_time_index = .data$group_time_index,
        variable = .data$.variable,
        first_nssp_date = first_nssp_date,
        first_nhsn_date = first_nhsn_date,
        first_nwss_date = first_nwss_date,
        nhsn_step_size = nhsn_step_size
      )
    ) |>
    dplyr::group_by(.variable, lab_site_index, date) |>
    dplyr::summarise(
      .value = mean(.value, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::filter(date <= model_info$last_training_date) |>
    dplyr::sample_n(100) |>
    dplyr::mutate(
      sample_collect_date = date,
      lab_id = lab_site_index + loc_offset,
      wwtp_id = lab_site_index + loc_offset,
      pcr_target_avg_conc = exp(.value),
      sample_location = "wwtp",
      sample_matrix = "raw wastewater",
      pcr_target_units = "copies/l wastewater",
      pcr_target = "sars-cov-2",
      wwtp_jurisdiction = model_info$location
    ) |>
    dplyr::select(-date, -lab_site_index, -.variable, -.value) |>
    dplyr::left_join(site_info, by = "wwtp_id")
}

ww_data <- purrr::pmap_dfr(loc_params, get_nwss_data_from_posterior)
report_date <- "2024-12-21"
ww_dir <- fs::path(
  "pipelines/tests/test_data/nwss_vintages",
  paste0("NWSS-ETL-covid-", report_date)
)

fs::dir_create(ww_dir, recurse = TRUE)
arrow::write_parquet(
  ww_data,
  fs::path(ww_dir, "bronze", ext = "parquet")
)

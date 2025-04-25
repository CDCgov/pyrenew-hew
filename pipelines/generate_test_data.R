script_packages <- c(
  "argparser",
  "nanoparquet",
  "dplyr",
  "fs",
  "hewr",
  "lubridate",
  "tidyr"
)

# load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})

# Dict for converting to short names
disease_short_names <- list(
  "COVID-19/Omicron" = "COVID-19",
  "Influenza" = "Influenza",
  "RSV" = "RSV"
)


#' Create Facility Test Data
#'
#' This function generates test data for a given facility over a specified
#' date range. The test data counts for the target disease are generated using
#' a cosine varying exponential growth process, with Poisson samples, while
#' counts for other diseases are generated using a mean-constant Poisson
#' process.
#'
#' @param facility A number representing the name of the facility.
#' @param start_reference A Date object representing the start date of the
#' reference period.
#' @param end_reference A Date object representing the end date of the reference
#' period.
#' @param geo_value A string representing the geographic value
#' (location) for the simulated data. Default is `"CA"`.
#' @param initial A numeric value representing the initial expected count for
#' the target disease. Default is 10.0.
#' @param mean_other A numeric value representing the mean count for other
#' diseases. Default is 200.0.
#' @param target_disease A character string representing the name of the target
#' disease. Default is `"COVID-19/Omicron"`.
#'
#' @return A tibble containing the generated test data with columns for
#' reference date, report date, geo type, geo value, as of date, run ID,
#' facility, disease, and value.
create_facility_test_data <- function(
    facility,
    start_reference,
    end_reference,
    geo_value = "CA",
    initial = 10.0,
    mean_other = 200.0,
    target_disease = "COVID-19/Omicron") {
  reference_dates <- seq(start_reference, end_reference, by = "day")
  rt <- 0.25 *
    cos(
      2 *
        pi *
        as.numeric(difftime(reference_dates, start_reference, units = "days")) /
        180
    )
  yt <- generate_exp_growth_pois(rt, initial)
  others <- generate_exp_growth_pois(0.0 * rt, mean_other)
  target_fac_data <- tibble(
    reference_date = reference_dates,
    report_date = end_reference,
    geo_type = "state",
    geo_value = geo_value,
    asof = end_reference,
    metric = "count_ed_visits",
    run_id = 0,
    facility = facility,
    !!target_disease := yt,
    Total = yt + others,
  ) |>
    pivot_longer(
      cols = c(all_of(target_disease), "Total"),
      names_to = "disease",
      values_to = "value"
    )
  return(target_fac_data)
}

#' Generate Fake Facility Data
#'
#' This function generates fake facility data for a specified
#' number of facilities within a given date range and writes
#' the data to a parquet file.
#'
#' @param facilities_to_simulate a tibble of facility/disease
#' pairs to simulate with columns `facility_id`, `geo_value`, and
#' `target_disease`.
#' @param private_data_dir A string specifying the directory
#' where the generated data will be saved.
#' @param start_reference A Date object specifying the start
#' date for the data generation. Default is "2024-06-01".
#' @param end_reference A Date object specifying the end date
#' for the data generation. Default is "2024-12-25".
#' @param initial A numeric value specifying the initial value
#' for the data generation. Default is 10.
#' @param mean_other A numeric value specifying the mean value for
#' other data points. Default is 200.
#'
#' @return This function does not return a value. It writes the
#' generated data to a parquet file as a side effect.
generate_fake_facility_data <-
  function(facilities_to_simulate,
           private_data_dir = path(getwd()),
           start_reference = as.Date("2024-06-01"),
           end_reference = as.Date("2024-12-21"),
           initial = 10,
           mean_other = 200) {
    nssp_etl_gold_dir <- path(private_data_dir, "nssp_etl_gold")
    dir_create(nssp_etl_gold_dir, recurse = TRUE)

    purrr::pmap(
      facilities_to_simulate,
      \(facility_id, geo_value, target_disease) {
        create_facility_test_data(
          facility = facility_id,
          start_reference = start_reference,
          end_reference = end_reference,
          geo_value = geo_value,
          initial = initial,
          mean_other = mean_other,
          target_disease = target_disease
        )
      }
    ) |>
      bind_rows() |>
      write_parquet(path(nssp_etl_gold_dir, end_reference, ext = "parquet"))
  }

#' Generate State Level Data
#'
#' This function generates state-level test data for a
#' specified disease over a given time period.
#' @param facilities_to_simulate a tibble of facility/disease
#' pairs to simulate with columns `facility_id`, `geo_value`, and
#' `target_disease`.
#' @param private_data_dir A string specifying the directory
#' where the generated data will be stored.
#' @param start_reference A Date object specifying the start date
#' for the data generation period. Default is "2024-06-01".
#' @param end_reference A Date object specifying the end date for the data
#' generation period. Default is "2024-12-25".
#' @param initial A numeric value specifying the initial value for the data
#' generation (per facility). Default is 10.
#' @param mean_other A numeric value specifying the mean value for
#' other data points (per facility). Default is 200.
#'
#' @return This function does not return a value. It writes the generated data
#' to parquet files in the specified directory as a side effect.
generate_fake_state_level_data <-
  function(facilities_to_simulate,
           private_data_dir = path(getwd()),
           start_reference = as.Date("2024-06-01"),
           end_reference = as.Date("2024-12-21"),
           initial = 10,
           mean_other = 200,
           n_forecast_days = 28) {
    gold_dir <- path(private_data_dir, "nssp_state_level_gold")
    dir_create(gold_dir, recurse = TRUE)

    comp_dir <- path(private_data_dir, "nssp-etl")
    dir_create(comp_dir, recurse = TRUE)

    state_data <-
      purrr::pmap(
        facilities_to_simulate,
        \(facility_id, geo_value, target_disease) {
          create_facility_test_data(
            facility = facility_id,
            start_reference = start_reference,
            end_reference = end_reference +
              lubridate::ddays(n_forecast_days),
            geo_value = geo_value,
            initial = initial,
            mean_other = mean_other,
            target_disease = target_disease
          )
        }
      ) |>
      bind_rows() |>
      group_by(across(c(-facility, -value))) |>
      summarise(value = sum(value), .groups = "drop") |>
      ungroup() |>
      select(-run_id, -asof)

    # Write in-sample state-level data to gold directory
    state_data |>
      filter(reference_date <= end_reference) |>
      mutate(
        any_update_this_day = TRUE,
        reference_date = as.Date(.data$reference_date)
      ) |>
      write_parquet(path(gold_dir, end_reference, ext = "parquet"))

    # Write out-of-sample state-level data to comparison directory
    state_data |>
      filter(reference_date > end_reference) |>
      write_parquet(path(comp_dir, "latest_comprehensive", ext = "parquet"))
  }

#' Generate Fake Parameter Data
#'
#' This function generates fake parameter data for a
#' specified disease and saves it as a parquet file.
#'
#' The function creates a directory for storing the
#' parameter estimates if it does not already exist.
#' It then generates a simple discretized
#' exponential distribution for the generation interval (gi_pmf)
#' and a right truncation probability mass function
#' (rt_truncation_pmf).
#'
#' @param private_data_dir A string specifying the directory
#' where the data will be saved.
#' @param states_to_generate A vector of strings representing
#' individual states for which to generate simulated right truncation
#' PMFs. Default is `"CA"` (create a PMF only for California).
#' @param end_reference A Date object specifying the end
#' reference date for the data. Default is "2024-12-25".
#' @param target_diseases A vector of strings specifying the
#' target disease(s) for the data. Default is
#' `c("COVID-19", "Influenza")`.
generate_fake_param_data <-
  function(private_data_dir = path(getwd()),
           states_to_generate = "CA",
           end_reference = as.Date("2024-12-21"),
           target_diseases = c("COVID-19", "Influenza")) {
    prod_param_estimates_dir <- path(
      private_data_dir,
      "prod_param_estimates"
    )
    dir_create(prod_param_estimates_dir, recurse = TRUE)

    purrr::map(target_diseases, \(target_disease) {
      ## Simple discretized lognormal distribution
      gi_pmf <- seq(0.5, 6.5) |> dexp()
      gi_pmf <- gi_pmf / sum(gi_pmf)
      delay_pmf <- log(seq(1, 11)) |> dnorm(log(3), 0.5)
      delay_pmf <- c(0, delay_pmf / sum(delay_pmf))
      rt_truncation_pmf <- c(1, 0, 0, 0)

      gi_data <- tibble(
        id = 0,
        start_date = as.Date("2024-06-01"),
        end_date = NA,
        reference_date = end_reference,
        disease = target_disease,
        format = "PMF",
        parameter = "generation_interval",
        geo_value = NA,
        value = list(gi_pmf)
      )

      delay_data <- tibble(
        id = 0,
        start_date = as.Date("2024-06-01"),
        end_date = NA,
        reference_date = end_reference,
        disease = target_disease,
        format = "PMF",
        parameter = "delay",
        geo_value = NA,
        value = list(delay_pmf)
      )

      rt_trunc_data <- purrr::imap(
        states_to_generate,
        \(x, i) {
          tibble(
            id = i,
            start_date = as.Date("2024-06-01"),
            end_date = NA,
            reference_date = end_reference,
            disease = target_disease,
            format = "PMF",
            parameter = "right_truncation",
            geo_value = x,
            value = list(rt_truncation_pmf)
          )
        }
      ) |>
        bind_rows()

      return(bind_rows(
        gi_data,
        delay_data,
        rt_trunc_data
      ))
    }) |>
      bind_rows() |>
      write_parquet(
        path(prod_param_estimates_dir, "prod", ext = "parquet")
      )
  }

#' Copy test NWSS wastewater data for end-to-end testing
#'
#' This function copies wastewater data from
#' `pipelines/tests/test_data/nwss_vintages/NWSS-ETL-covid-<date>` to
#' `end_to_end_test_output/private_data/nwss_vintages/
#' NWSS-ETL-covid-<end_reference>`.
#'
#' @param private_data_dir Path to private data directory
#' (default: current working directory).
#' @param test_data_dir Path to the source test data directory.
#' @param end_reference A string to specify the
#' end reference date (e.g., "2024-12-21").
copy_test_nwss_data <- function(
    private_data_dir = fs::path_wd(),
    end_reference = "2024-12-21",
    test_data_dir = fs::path(
      "pipelines/tests/test_data/nwss_vintages",
      paste0(
        "NWSS-ETL-covid-",
        end_reference
      )
    )) {
  ww_dir <- fs::path(
    private_data_dir,
    "nwss_vintages",
    paste0("NWSS-ETL-covid-", end_reference)
  )
  fs::dir_create(ww_dir, recurse = TRUE)
  ww_data <- nanoparquet::read_parquet(
    fs::path(test_data_dir, "bronze", ext = "parquet")
  )
  nanoparquet::write_parquet(
    ww_data,
    fs::path(ww_dir, "bronze", ext = "parquet")
  )
}

main <- function(private_data_dir, target_diseases, n_forecast_days) {
  short_target_diseases <- disease_short_names[target_diseases]
  facilities <- tibble::tibble(
    facility_id = 1:5,
    geo_value = c(
      rep("CA", 3),
      rep("MT", 2)
    )
  )
  to_generate <- tidyr::crossing(
    facilities,
    target_disease = target_diseases
  )

  generate_fake_facility_data(
    to_generate,
    private_data_dir
  )

  generate_fake_state_level_data(
    to_generate,
    private_data_dir,
    n_forecast_days = n_forecast_days
  )
  generate_fake_param_data(
    private_data_dir,
    states_to_generate = c("MT", "CA", "US"),
    target_diseases = short_target_diseases
  )
  copy_test_nwss_data(
    private_data_dir
  )
}

p <- arg_parser("Create simulated epiweekly data.") |>
  add_argument(
    "model_run_dir",
    help = "Directory containing the model data and output."
  ) |>
  add_argument(
    "--target-diseases",
    type = "character",
    default = "COVID-19/Omicron Influenza RSV",
    help = paste0(
      "Target disease(s) for which to simulate data, ",
      "as a whitespace-separated string"
    )
  ) |>
  add_argument(
    "--n-forecast-days",
    type = "integer",
    default = 28,
    help = "Number of days to forecast."
  ) |>
  add_argument(
    "--seed",
    type = "integer",
    default = 123,
    help = "Seed for the pseudorandom number generator."
  )


argv <- parse_args(p)

withr::with_seed(argv$seed, {
  main(
    argv$model_run_dir,
    target_diseases = stringr::str_split_1(argv$target_diseases, " "),
    n_forecast_days = argv$n_forecast_days
  )
})

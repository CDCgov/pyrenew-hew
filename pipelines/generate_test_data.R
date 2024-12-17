script_packages <- c(
  "argparser",
  "arrow",
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

# Set seed for reproducibility
set.seed(123)

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
#' @param initial A numeric value representing the initial expected count for
#' the target disease. Default is 10.0.
#' @param mean_other A numeric value representing the mean count for other
#' diseases. Default is 200.0.
#' @param target_disease A character string representing the name of the target
#' disease. Default is "COVID-19/Omicron".
#'
#' @return A tibble containing the generated test data with columns for
#' reference date, report date, geo type, geo value, as of date, run ID,
#' facility, disease, and value.
create_facility_test_data <- function(facility, start_reference, end_reference,
    initial = 10.0, mean_other = 200.0, target_disease = "COVID-19/Omicron") {
        reference_dates <- seq(start_reference, end_reference, by = "day")
        rt <- 0.25 * cos(2 * pi * as.numeric(difftime(reference_dates,
            start_reference, units = "days")) / 180)
        yt <- generate_exp_growth_pois(rt, initial)
        others <- generate_exp_growth_pois(0.0 * rt, mean_other)
        target_fac_data <- tibble(
            reference_date = reference_dates,
            report_date = end_reference,
            geo_type = "state",
            geo_value = "CA",
            asof = end_reference,
            metric = "count_ed_visits",
            run_id = 0,
            facility = facility,
            !!target_disease := yt,
            Total = yt + others,
        ) |> pivot_longer(cols = c(all_of(target_disease), "Total"),
            names_to = "disease", values_to = "value")
        return(target_fac_data)
    }

#' Generate Fake Facility Data
#'
#' This function generates fake facility data for a specified number of
#' facilities within a given date range and writes the data to a parquet file.
#'
#' @param private_data_dir A string specifying the directory where the generated
#' data will be saved.
#' @param n_facilities An integer specifying the number of facilities to
#' generate data for. Default is 3.
#' @param start_reference A Date object specifying the start date for the data
#' generation. Default is "2024-06-01".
#' @param end_reference A Date object specifying the end date for the data
#' generation. Default is "2024-12-25".
#' @param initial A numeric value specifying the initial value for the data
#' generation. Default is 10.0.
#' @param mean_other A numeric value specifying the mean value for other data
#' points. Default is 200.0.
#' @param target_disease A string specifying the target disease for the data
#' generation. Default is "COVID-19/Omicron".
#'
#' @return This function does not return a value. It writes the generated data
#' to a parquet file.
generate_fake_facility_data <- function(private_data_dir, n_facilities = 3,
    start_reference = as.Date("2024-06-01"),
    end_reference = as.Date("2024-12-25"), initial = 10.0, mean_other = 200.0,
    target_disease = "COVID-19/Omicron") {
    dir_to_create <- path(private_data_dir, "nssp_etl_gold")
    if (!dir_exists(dir_to_create)) {
            dir_create(dir_to_create)
        }

    fac_data <- purrr::map(1:n_facilities, \(i) {
        create_facility_test_data(i, start_reference, end_reference,
            initial, mean_other, target_disease)
    }) |>
    bind_rows() |>
    write_parquet(path(dir_to_create, end_reference, ext = "parquet"))
}

#' Generate State Level Data
#'
#' This function generates state-level test data for a specified disease over a
#' given time period.
#'
#' @param private_data_dir A string specifying the directory where the generated
#' data will be stored.
#' @param start_reference A Date object specifying the start date for the data
#' generation period. Default is "2024-06-01".
#' @param end_reference A Date object specifying the end date for the data
#' generation period. Default is "2024-12-25".
#' @param initial A numeric value specifying the initial value for the data
#' generation. Default is 10.0.
#' @param mean_other A numeric value specifying the mean value for other data
#' points. Default is 200.0.
#' @param target_disease A string specifying the target disease for the data
#' generation. Default is "COVID-19/Omicron".
#'
#' @return This function does not return a value. It writes the generated data
#' to a parquet file in the specified directory.
generate_state_level_data <- function(private_data_dir,
    start_reference = as.Date("2024-06-01"),
    end_reference = as.Date("2024-12-25"), initial = 10.0, mean_other = 200.0,
    target_disease = "COVID-19/Omicron") {
    dir_to_create <- path(private_data_dir, "nssp_state_level_gold")
    if (!dir_exists(dir_to_create)) {
            dir_create(dir_to_create)
        }

    state_data <- create_facility_test_data(1, start_reference, end_reference,
            initial, mean_other, target_disease) |>
    mutate(any_update_this_day = TRUE) |>
    select(-facility, -run_id, -asof) |>
    write_parquet(path(dir_to_create, end_reference, ext = "parquet"))
}

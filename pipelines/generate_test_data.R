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
            run_id = 0,
            facility = facility,
            !!target_disease := yt,
            Total = yt + others,
        ) |> pivot_longer(cols = c(all_of(target_disease), "Total"),
            names_to = "disease", values_to = "value")
        return(target_fac_data)
    }

generate_fake_facility_data <- function(private_data_dir, n_facilities = 3,
    start_reference = as.Date("2024-06-01"),
    end_reference = as.Date("2024-12-25"), initial = 10.0, mean_other = 200.0,
    target_disease = "COVID-19/Omicron") {

    fac_data <- purrr::map(1:n_facilities, \(i) {
        create_facility_test_data(i, start_reference, end_reference,
            initial, mean_other, target_disease)
    }) |>
    bind_rows() |>
    write_parquet(path(private_data_dir, end_reference, ext = "parquet"))
}

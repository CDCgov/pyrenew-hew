# Read commnad line args
library(tidyverse)
library(CFAEpiNow2Pipeline)
library(usa)
library(fs)
library(arrow)
library(here)
library(glue)
library(jsonlite)


prep_data <- function(report_date = today(),
                      min_reference_date = "2000-01-01",
                      max_reference_date = "3000-01-01",
                      last_training_date = max_reference_date,
                      state_abb = "US") {
  prepped_data <- read_data(
    data_path = here(
      path("nssp_demo", "private_data", report_date, ext = "parquet")
    ),
    disease = "COVID-19",
    state_abb = state_abb,
    report_date = report_date,
    max_reference_date = max_reference_date,
    min_reference_date = min_reference_date
  ) %>%
    as_tibble() %>%
    select(date = reference_date, COVID_ED_admissions = confirm) %>%
    mutate(data_type = if_else(date <= last_training_date, "train", "test"))

  train_ed_admissions <- prepped_data %>%
    filter(data_type == "train") %>%
    pull(COVID_ED_admissions)

  test_ed_admissions <- prepped_data %>%
    filter(data_type == "test") %>%
    pull(COVID_ED_admissions)


  state_pop <-
    usa::facts %>%
    left_join(select(usa::states, abb, name)) %>%
    filter(abb == state_abb) %>%
    pull(population)

  nnh_estimates <- read_parquet(
    here(path("nssp_demo",
      "private_data",
      "prod",
      ext = "parquet"
    ))
  )

  generation_interval_pmf <-
    nnh_estimates %>%
    filter(
      is.na(geo_value),
      disease == "COVID-19",
      parameter == "generation_interval"
    ) %>%
    pull(value) %>%
    pluck(1)


  delay_pmf <-
    nnh_estimates %>%
    filter(
      is.na(geo_value),
      disease == "COVID-19",
      parameter == "delay"
    ) %>%
    pull(value) %>%
    pluck(1)

  right_truncation_pmf <-
    nnh_estimates %>%
    filter(
      geo_value == state_abb,
      disease == "COVID-19",
      parameter == "right_truncation"
    ) %>%
    pull(value) %>%
    pluck(1)


  list(
    prepped_date = prepped_data,
    data_for_model_fit = list(
      inf_to_hosp_pmf = delay_pmf,
      generation_interval_pmf = generation_interval_pmf,
      right_truncation_pmf = right_truncation_pmf,
      data_observed_hospital_admissions = train_ed_admissions,
      test_ed_admissions = test_ed_admissions,
      state_pop = state_pop
    )
  )
}


prep_and_save_data <- function(report_date,
                               min_reference_date,
                               max_reference_date,
                               last_training_date,
                               state_abb) {
  # prep data
  dat <- prep_data(
    report_date = report_date,
    min_reference_date = min_reference_date,
    max_reference_date = max_reference_date,
    last_training_date = last_training_date,
    state_abb = state_abb
  )

  actual_first_date <- min(dat$prepped_date$date)
  actual_last_date <- max(dat$prepped_date$date)



  # Create folders
  model_folder_name <- glue(paste0(
    "r_{report_date}_",
    "f_{actual_first_date}_",
    "l_{actual_last_date}_",
    "t_{last_training_date}"
  ))
  model_folder <- here("nssp_demo", "private_data", model_folder_name)
  dir_create(model_folder)

  data_folder <- path(model_folder, state_abb)
  dir_create(data_folder)


  # save state_pop and ed_visits in a single json
  write_json(
    x = dat$data_for_model_fit,
    path = path(data_folder, "data_for_model_fit", ext = "json"),
    auto_unbox = TRUE
  )

  # save whole dataset with forecast indicators as a csv
  write_csv(dat$prepped_date, file = path(data_folder, "data", ext = "csv"))
}

walk(
  usa::state.abb,
  \(x) {
    prep_and_save_data(
      report_date = "2024-09-10",
      min_reference_date = "2000-01-01",
      max_reference_date = "3000-01-01",
      last_training_date = "2024-08-14",
      state_abb = x
    )
  }
)

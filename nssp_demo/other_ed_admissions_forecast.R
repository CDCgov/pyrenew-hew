library(tidyverse)
library(fs)
library(fable)
library(jsonlite)
library(glue)
library(argparser)
library(arrow)

p <- arg_parser("Forecast total ED admissions") %>%
  add_argument(p, "--model_dir",
    help = "Directory containing the model data",
    required = TRUE
  ) %>%
  add_argument("--n_forecast_days",
    help = "Number of days to forecast",
    default = 28L
  ) %>%
  add_argument("--n_samples",
    help = "Number of samples to generate",
    default = 2000L
  )

argv <- parse_args(p)
model_dir <- path(argv$model_dir)
n_forecast_days <- argv$n_forecast_days
n_samples <- arv$n_samples

disease_name_nssp_map <- c(
  "covid-19" = "COVID-19/Omicron",
  "influenza" = "Influenza"
)

base_dir <- path_dir(model_dir)

disease_name_raw <- base_dir %>%
  path_file() %>%
  str_extract("^.+(?=_r_)")

disease_name_nssp <- unname(disease_name_nssp_map[disease_name_raw])

fit_and_forecast <- function(other_data,
                             n_forecast_days = 28,
                             n_samples = 2000) {
  forecast_horizon <- glue("{n_forecast_days} days")

  fit <-
    other_data %>%
    filter(data_type == "train") %>%
    model(
      comb_model = combination_ensemble(
        ETS(log(ED_admissions) ~ trend(method = c("N", "M", "A"))),
        ARIMA(log(ED_admissions))
      ),
      arima = ARIMA(log(ED_admissions)),
      ets = ETS(log(ED_admissions) ~ trend(method = c("N", "M", "A")))
    )

  forecast_samples <- fit |>
    generate(h = forecast_horizon, times = n_samples) |>
    as_tibble() %>%
    mutate(ED_admissions = .sim, .draw = as.integer(.rep)) |>
    filter(.model == "comb_model") %>%
    select(date, .draw, total_ED_admissions = ED_admissions)

  forecast_samples
}

main <- function(model_dir, n_forecast_days = 28, n_samples = 2000) {
  # to do: do this with json data that has dates
  data_path <- path(model_dir, "data", ext = "csv")

  other_data <- read_csv(data_path) %>%
    mutate(disease = if_else(disease == disease_name_nssp,
      "Disease", disease
    )) %>%
    pivot_wider(names_from = disease, values_from = ED_admissions) %>%
    mutate(Other = Total - Disease) %>%
    select(date, ED_admissions = Other, data_type) %>%
    as_tsibble(index = date)

  forecast_samples <- fit_and_forecast(other_data, n_forecast_days, n_samples)

  save_path <- path(model_dir, "other_ed_admissions_forecast", ext = "parquet")
  write_parquet(forecast_samples, save_path)
}


main(model_dir, n_forecast_days, n_samples)
# File will end here once command line version is working
# Temp code to run for all states while command line version doesn't work
# Command line version is dependent on https://github.com/rstudio/renv/pull/2018

base_dir <- path(
  "nssp_demo/private_data/influenza_r_2024-10-21_f_2024-07-16_t_2024-10-13"
)

dir_ls(base_dir, type = "dir") %>%
  walk(.f = function(model_dir) {
    print(path_file(model_dir))
    main(model_dir, n_forecast_days = 28, n_samples = 2000)
  })

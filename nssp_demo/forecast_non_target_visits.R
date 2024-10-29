script_packages <- c(
  "dplyr",
  "tidyr",
  "tibble",
  "readr",
  "stringr",
  "fs",
  "fable",
  "jsonlite",
  "argparser",
  "arrow"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


fit_and_forecast <- function(other_data,
                             n_forecast_days = 28,
                             n_samples = 2000) {
  forecast_horizon <- glue::glue("{n_forecast_days} days")

  fit <-
    other_data |>
    filter(data_type == "train") |>
    model(
      comb_model = combination_ensemble(
        ETS(log(ed_visits) ~ trend(method = c("N", "M", "A"))),
        ARIMA(log(ed_visits))
      ),
      arima = ARIMA(log(ed_visits)),
      ets = ETS(log(ed_visits) ~ trend(method = c("N", "M", "A")))
    )

  forecast_samples <- fit |>
    generate(h = forecast_horizon, times = n_samples) |>
    as_tibble() |>
    mutate(ed_visits = .sim, .draw = as.integer(.rep)) |>
    filter(.model == "comb_model") |>
    select(date, .draw, other_ed_visits = ed_visits)

  forecast_samples
}

main <- function(model_run_dir, n_forecast_days = 28, n_samples = 2000) {
  # to do: do this with json data that has dates
  data_path <- path(model_run_dir, "data", ext = "csv")

  other_data <- read_csv(
    data_path,
    col_types = cols(
      disease = col_character(),
      data_type = col_character(),
      ed_visits = col_double(),
      date = col_date()
    )
  ) |>
    mutate(disease = if_else(
      disease == disease_name_nssp,
      "Disease", disease
    )) |>
    pivot_wider(names_from = disease, values_from = ed_visits) |>
    mutate(Other = Total - Disease) |>
    select(date, ed_visits = Other, data_type) |>
    as_tsibble(index = date)

  forecast_samples <- fit_and_forecast(other_data, n_forecast_days, n_samples)

  save_path <- path(model_run_dir, "other_ed_visits_forecast", ext = "parquet")
  write_parquet(forecast_samples, save_path)
}


p <- arg_parser(
  "Forecast other (non-target-disease) ED visits for a given location."
) |>
  add_argument(
    "--model-run-dir",
    help = "Directory containing the model data and output.",
  ) |>
  add_argument(
    "--n-forecast-days",
    help = "Number of days to forecast.",
    default = 28L
  ) |>
  add_argument(
    "--n-samples",
    help = "Number of samples to generate.",
    default = 2000L
  )

argv <- parse_args(p)
model_run_dir <- path(argv$model_run_dir)
n_forecast_days <- argv$n_forecast_days
n_samples <- argv$n_samples

disease_name_nssp_map <- c(
  "covid-19" = "COVID-19/Omicron",
  "influenza" = "Influenza"
)

base_dir <- path_dir(model_run_dir)

disease_name_raw <- base_dir |>
  path_file() |>
  str_extract("^.+(?=_r_)")

disease_name_nssp <- unname(disease_name_nssp_map[disease_name_raw])

main(model_run_dir, n_forecast_days, n_samples)

script_packages <- c(
  "argparser",
  "arrow",
  "dplyr",
  "forecasttools",
  "fs",
  "ggplot2",
  "lubridate",
  "readr",
  "scoringutils",
  "stringr",
  "tidyr"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


tidy_and_save_mcmc <- function(model_run_dir,
                               model_name,
                               file_name_prefix = "",
                               filter_bad_chains,
                               good_chain_tol) {
  model_dir <- path(model_run_dir, model_name)
  inference_data_path <- path(model_dir, "inference_data",
    ext = "csv"
  )

  tidy_inference_data <- inference_data_path |>
    read_csv(show_col_types = FALSE) |>
    inferencedata_to_tidy_draws()

  if (filter_bad_chains) {
    good_chains <-
      tibble::deframe(tidy_inference_data)$log_likelihood |>
      pivot_longer(-starts_with(".")) |>
      group_by(.iteration, .chain) |>
      summarize(value = sum(value), .groups = "drop") |>
      group_by(.chain) |>
      summarize(value = mean(value)) |>
      filter(value >= max(value) - 2) |>
      pull(.chain)
  } else {
    good_chains <-
      tibble::deframe(tidy_inference_data)$log_likelihood$.chain |>
      unique()
  }

  tidy_inference_data <- tidy_inference_data |>
    mutate(
      data = purrr::map(data, \(x) filter(x, .chain %in% good_chains))
    )

  save_dir <- path(model_dir, "mcmc_tidy")
  dir_create(save_dir)

  purrr::pwalk(tidy_inference_data, .f = function(group_name, data) {
    write_parquet(data, path(save_dir,
      str_c(file_name_prefix, group_name),
      ext = "parquet"
    ))
  })
}


p <- arg_parser("Tidy InferenceData to Parquet files") |>
  add_argument(
    "model_run_dir",
    help = "Directory containing the model data and output.",
  ) |>
  add_argument(
    "--model-name",
    help = "Name of model.",
  ) |>
  add_argument(
    "--no-filter-bad-chains",
    help = paste0(
      "By default, tidy_and_save_mcmc.R filters ",
      "any bad chains from the samples. Set this flag ",
      "to retain them"
    ),
    flag = TRUE
  ) |>
  add_argument(
    "--good-chain-tol",
    help = "Tolerance level for determining good chains.",
    default = 2L
  )

argv <- parse_args(p)
model_run_dir <- path(argv$model_run_dir)
model_name <- argv$model_name
filter_bad_chains <- !argv$no_filter_bad_chains
good_chain_tol <- argv$good_chain_tol

tidy_and_save_mcmc(model_run_dir,
  model_name,
  file_name_prefix = "pyrenew_",
  filter_bad_chains,
  good_chain_tol
)

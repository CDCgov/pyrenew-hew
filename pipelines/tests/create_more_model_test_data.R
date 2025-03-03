# This scripts takes real pyrenew_e output from the test pipeline and augments
# it to create fake preyrenew_h and pyrenew_he output. It then processes the
# remaining parts of the pipeline on the fake output.
# In the future, the pipeline script should be updated, and this script should
# be deleted.
library(purrr)
library(readr)
library(dplyr)
library(fs)
library(glue)
try(source("hewr/R/process_state_forecast.R"))
try(source("pipelines/plot_and_save_state_forecast.R"))
try(source("pipelines/score_forecast.R"))

model_batch_dirs <- c(
  path(
    "pipelines/tests/private_data",
    "influenza_r_2024-12-21_f_2024-10-22_t_2024-12-20"
  )
)

walk(model_batch_dirs, \(model_batch_dir) {
  dir_delete(path(model_batch_dir, "diagnostic_report"))

  file_delete(dir_ls(model_batch_dir, glob = "*.tsv"))
  model_run_dirs <- dir_ls(path(model_batch_dir, "model_runs"))
  walk(
    model_run_dirs,
    \(model_run_dir) {
      print("creating h data for")
      print(model_run_dir)
      file_delete(dir_ls(model_run_dir, glob = "*.pdf", recurse = TRUE))
      file_delete(dir_ls(model_run_dir, glob = "*.rds", recurse = TRUE))

      pyrenew_e_path <- fs::path(model_run_dir, "pyrenew_e")
      pyrenew_h_path <- fs::path(model_run_dir, "pyrenew_h")
      pyrenew_he_path <- fs::path(model_run_dir, "pyrenew_he")
      timeseries_e_path <- fs::path(model_run_dir, "timeseries_e")

      file_delete(dir_ls(pyrenew_e_path, glob = "*.parquet"))
      dir_delete(path(pyrenew_e_path, "mcmc_tidy"))

      fs::dir_copy(pyrenew_e_path, pyrenew_h_path)
      fs::dir_copy(pyrenew_e_path, pyrenew_he_path)

      raw_csv <- path(pyrenew_h_path, "inference_data", ext = "csv") |>
        read_csv()

      new_colnames <- map_chr(
        0:(11 - 4),
        \(i) {
          glue("('log_likelihood', 'observed_hospital_admissions[{i}]', {i})")
        }
      )


      if (all(new_colnames %in% colnames(raw_csv))) {
        new_tbl <- raw_csv
      } else {
        new_tbl <-
          raw_csv |>
          bind_cols(
            map(new_colnames, \(x) rnorm(nrow(raw_csv))) |>
              set_names(new_colnames) |>
              bind_cols()
          )
      }


      new_tbl |>
        write_csv(path(pyrenew_he_path, "inference_data", ext = "csv"))
      system2("Rscript", c(
        "pipelines/convert_inferencedata_to_parquet.R",
        model_run_dir, "--model-name", "pyrenew_he"
      ))

      new_tbl |>
        select(-contains("observed_hospital_admissions")) |>
        write_csv(path(pyrenew_e_path, "inference_data", ext = "csv"))
      system2("Rscript", c(
        "pipelines/convert_inferencedata_to_parquet.R",
        model_run_dir, "--model-name", "pyrenew_e"
      ))

      new_tbl |>
        select(-contains("observed_ed_visits")) |>
        write_csv(path(pyrenew_h_path, "inference_data", ext = "csv"))
      system2("Rscript", c(
        "pipelines/convert_inferencedata_to_parquet.R",
        model_run_dir, "--model-name", "pyrenew_h"
      ))


      ## Process state forecasts
      process_state_forecast(model_run_dir,
        "pyrenew_he",
        "timeseries_e",
        ci_widths = c(0.5, 0.8, 0.95),
        save = TRUE
      )

      process_state_forecast(model_run_dir,
        "pyrenew_e",
        "timeseries_e",
        ci_widths = c(0.5, 0.8, 0.95),
        save = TRUE
      )

      process_state_forecast(model_run_dir,
        "pyrenew_h",
        NULL,
        ci_widths = c(0.5, 0.8, 0.95),
        save = TRUE
      )

      ## Save forecast figures
      save_forecast_figures(
        model_run_dir,
        "pyrenew_he",
        "timeseries_e"
      )

      save_forecast_figures(
        model_run_dir,
        "pyrenew_e",
        "timeseries_e"
      )

      save_forecast_figures(
        model_run_dir,
        "pyrenew_h",
        NULL
      )

      ## Not Rendering Diagnostic Report

      ## Scoring
      read_and_score_location(model_run_dir)
    }
  )
  model_batch_params <- parse_model_batch_dir_path(model_batch_dir)
  output_file <- glue(
    "{model_batch_params$report_date}-",
    "{str_to_lower(model_batch_params$disease)}-hubverse-table.tsv"
  )
  output_path <- path(model_batch_dir, output_file)

  hewr::to_epiweekly_quantile_table(model_batch_dir) |>
    readr::write_tsv(output_path)
})

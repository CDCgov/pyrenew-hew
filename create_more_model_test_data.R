library(tidyverse)
library(fs)
library(glue)
base_paths <- dir_ls(fs::path("/Users/damon/Downloads/Archive/covid-19_r_2024-12-21_f_2024-10-22_t_2024-12-20/model_runs"))
base_paths <- dir_ls(fs::path("/Users/damon/Downloads/Archive/influenza_r_2024-12-21_f_2024-10-22_t_2024-12-20/model_runs"))
# base_path <- base_paths[1]
walk(base_paths, \(base_path) {
  pyrenew_e_path <- fs::path(base_path, "pyrenew_e")
  pyrenew_h_path <- fs::path(base_path, "pyrenew_h")
  pyrenew_he_path <- fs::path(base_path, "pyrenew_he")
  fs::dir_copy(pyrenew_e_path, pyrenew_h_path)
  fs::dir_copy(pyrenew_e_path, pyrenew_he_path)

  raw_csv <- path(pyrenew_he_path, "inference_data", ext = "csv") |>
    read_csv()

  new_colnames <- map_chr(0:(11 - 4), \(i) glue("('log_likelihood', 'observed_hospital_admissions[{i}]', {i})"))


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
  system2("Rscript", c("pipelines/convert_inferencedata_to_parquet.R", base_path, "--model-name", "pyrenew_he"))

  new_tbl |>
    select(-contains("observed_hospital_admissions")) |>
    write_csv(path(pyrenew_e_path, "inference_data", ext = "csv"))
  system2("Rscript", c("pipelines/convert_inferencedata_to_parquet.R", base_path, "--model-name", "pyrenew_e"))

  new_tbl |>
    select(-contains("observed_ed_visits")) |>
    write_csv(path(pyrenew_h_path, "inference_data", ext = "csv"))
  system2("Rscript", c("pipelines/convert_inferencedata_to_parquet.R", base_path, "--model-name", "pyrenew_h"))
})

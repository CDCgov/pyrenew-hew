library(tidyverse)
library(fs)
library(glue)
library(argparser)
p <- arg_parser("Process model batch directories")
p <- add_argument(p, "super_dir", help = "Directory containing model batch directories")
argv <- parse_args(p)
super_dir <- path(argv$super_dir)
dir_delete(path(dir_ls(super_dir), "diagnostic_report"))
dir_delete(path(dir_ls(super_dir), "figures"))
file_delete(dir_ls(dir_ls(super_dir), glob = "*.tsv"))
model_run_dirs <- dir_ls(path(dir_ls(super_dir), "model_runs"))
walk(
  model_run_dirs,
  \(model_run_dir) {
    file_delete(dir_ls(model_run_dir, glob = "*.pdf"))
    file_delete(dir_ls(model_run_dir, glob = "*.rds"))

    pyrenew_e_path <- fs::path(model_run_dir, "pyrenew_e")
    pyrenew_h_path <- fs::path(model_run_dir, "pyrenew_h")
    pyrenew_he_path <- fs::path(model_run_dir, "pyrenew_he")
    timeseries_e_path <- fs::path(model_run_dir, "timeseries_e")


    file_delete(dir_ls(pyrenew_e_path, glob = "*.parquet"))
    fs::dir_delete(timeseries_e_path)
    dir_delete(path(pyrenew_e_path, "mcmc_tidy"))

    fs::dir_copy(pyrenew_e_path, pyrenew_h_path)
    fs::dir_delete(pyrenew_e_path)
    # fs::dir_copy(pyrenew_e_path, pyrenew_he_path)


    raw_csv <- path(pyrenew_h_path, "inference_data", ext = "csv") |>
      read_csv()

    # new_colnames <- map_chr(0:(11 - 4), \(i) glue("('log_likelihood', 'observed_hospital_admissions[{i}]', {i})"))
    #
    #
    # if (all(new_colnames %in% colnames(raw_csv))) {
    new_tbl <- raw_csv
    # } else {
    #   new_tbl <-
    #     raw_csv |>
    #     bind_cols(
    #       map(new_colnames, \(x) rnorm(nrow(raw_csv))) |>
    #         set_names(new_colnames) |>
    #         bind_cols()
    #     )
    # }


    # new_tbl |>
    #   write_csv(path(pyrenew_he_path, "inference_data", ext = "csv"))
    # system2("Rscript", c("pipelines/convert_inferencedata_to_parquet.R", model_run_dir, "--model-name", "pyrenew_he"))

    # new_tbl |>
    #   select(-contains("observed_hospital_admissions")) |>
    #   write_csv(path(pyrenew_e_path, "inference_data", ext = "csv"))
    # system2("Rscript", c("pipelines/convert_inferencedata_to_parquet.R", model_run_dir, "--model-name", "pyrenew_e"))

    new_tbl |>
      select(-contains("observed_ed_visits")) |>
      write_csv(path(pyrenew_h_path, "inference_data", ext = "csv"))
    system2("Rscript", c("pipelines/convert_inferencedata_to_parquet.R", model_run_dir, "--model-name", "pyrenew_h"))
  }
)

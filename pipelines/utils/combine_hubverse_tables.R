#!/usr/bin/env Rscript

main <- function(model_batch_dir, output_path) {
  model_batch_dir |>
    fs::dir_ls(recurse = TRUE, glob = "*/hubverse_table.parquet") |>
    purrr::map(forecasttools::read_tabular) |>
    dplyr::bind_rows() |>
    forecasttools::write_tabular(output_path)
}


p <- argparser::arg_parser(
  "Create a hubverse table from location specific forecast draws."
) |>
  argparser::add_argument(
    "model_batch_dir",
    help = paste0(
      "Directory containing subdirectories that represent ",
      "individual forecast locations, with a directory name ",
      "that indicates the target pathogen and reference date"
    )
  ) |>
  argparser::add_argument(
    "output_path",
    help = "Path to which to save the table."
  )

argv <- argparser::parse_args(p)

main(
  argv$model_batch_dir,
  argv$output_path
)

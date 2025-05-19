#!/usr/bin/env Rscript

#' Create a hubverse table from model output, using
#' utilities from `hewr`.
#'
#' @param model_batch_dir Model batch directory from which
#' to create a hubverse table
#' @param output_path path to save the table as a tsv
#' @return Nothing, saving the table as a side effect.
main <- function(model_batch_dir, output_path) {
  hewr::to_hub_quantile_table(model_batch_dir) |>
    arrow::write_parquet(output_path)
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

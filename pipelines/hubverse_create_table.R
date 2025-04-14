#!/usr/bin/env Rscript


#' Create a hubverse table from model output, using
#' utilities from `hewr`.
#'
#' @param model_batch_dir Model batch directory from which
#' to create a hubverse table
#' @param output_path path to save the table as a tsv
#' @return Nothing, saving the table as a side effect.
main <- function(model_batch_dir,
                 output_path,
                 locations_exclude) {
  hewr::to_epiweekly_quantile_table(model_batch_dir, locations_exclude) |>
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
  ) |>
  argparser::add_argument(
    "--locations-exclude",
    help = paste0(
      "Comma-separated list of two-letter location codes to ",
      "exclude."
    ),
    default = "AS,GU,MO,MP,PR,UM,VI"
  )

argv <- argparser::parse_args(p)
locations_exclude_vec <- unlist(strsplit(argv$locations_exclude, split = ","))

main(
  argv$model_batch_dir,
  argv$output_path,
  locations_exclude_vec
)

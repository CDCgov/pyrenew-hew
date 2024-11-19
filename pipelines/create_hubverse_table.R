#!/usr/bin/env Rscript


#' Create a hubverse table from model output, using
#' utilities from `hewr`.
#'
#' @param model_batch_dir Model batch directory from which
#' to create a hubverse table
#' @param output_path path to save the table as a tsv
#' @param exclude Locations to exclude, as a vector of strings.
#' @return Nothing, saving the table as a side effect.
main <- function(model_batch_dir,
                 output_path,
                 exclude = NULL) {
  hewr::to_epiweekly_quantile_table(
    model_batch_dir,
    exclude = exclude
  ) |>
    readr::write_tsv(output_path)
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
    help = "path to which to save the table"
  ) |>
  argparser::add_argument(
    "--exclude",
    help = "locations to exclude, as a whitespace-separated string",
    default = ""
  )
argv <- argparser::parse_args(p)

main(
  argv$model_batch_dir,
  argv$output_path,
  stringr::str_split_1(argv$exclude, " ")
)

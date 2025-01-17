script_packages <- c(
  "data.table",
  "argparser",
  "stringr"
)

# load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


bind_tables <- function(list_of_table_pairs) {
  sample_metrics <- purrr::map(
    list_of_table_pairs,
    \(x) {
      attr(x$sample_scores, "metrics")
    }
  ) |>
    unlist() |>
    unique()
  quantile_metrics <- purrr::map(
    list_of_table_pairs,
    \(x) {
      attr(x$quantile_scores, "metrics")
    }
  ) |>
    unlist() |>
    unique()

  sample_scores <- purrr::map(
    list_of_table_pairs, "sample_scores"
  ) |>
    data.table::rbindlist(fill = TRUE)

  quantile_scores <- purrr::map(
    list_of_table_pairs, "quantile_scores"
  ) |>
    data.table::rbindlist(fill = TRUE)

  attr(sample_scores, "metrics") <- sample_metrics
  attr(quantile_scores, "metrics") <- quantile_metrics


  return(list(
    sample_scores = sample_scores,
    quantile_scores = quantile_scores
  ))
}

score_table_path <- function(dir,
                             disease,
                             score_file_stem,
                             score_file_ext) {
  return(fs::path(dir,
    glue::glue("{disease}_{score_file_stem}"),
    ext = score_file_ext
  ))
}

read_score_table <- function(dir,
                             disease,
                             score_file_stem,
                             score_file_ext) {
  tab_path <- score_table_path(
    dir,
    disease,
    score_file_stem,
    score_file_ext
  )
  if (!(fs::file_exists(tab_path))) {
    return(NULL)
  } else {
    return(readr::read_rds(tab_path))
  }
}


collate_disease <- function(base_dir,
                            disease,
                            score_file_stem = "score_table",
                            score_file_ext = "rds") {
  message(glue::glue("Collating disease {disease} from dir {base_dir}"))
  tabs <- fs::dir_map(base_dir,
    \(x) {
      read_score_table(
        x,
        disease,
        score_file_stem,
        score_file_ext
      )
    },
    type = "directory"
  )
  return(bind_tables(tabs))
}

collate_and_save <- function(base_dir,
                             save_dir = NULL,
                             diseases = c("COVID-19", "Influenza"),
                             score_file_stem = "score_table",
                             score_file_ext = "rds") {
  if (is.null(save_dir) || is.na(save_dir)) {
    save_dir <- base_dir
  }
  collate_and_save_disease <- function(disease) {
    fs::dir_create(save_dir)
    collate_disease(
      base_dir,
      disease,
      score_file_stem,
      score_file_ext
    ) |>
      readr::write_rds(score_table_path(
        save_dir,
        disease,
        score_file_stem,
        score_file_ext
      ))
  }
  purrr::walk(
    diseases,
    collate_and_save_disease
  )
}

p <- arg_parser(
  paste0(
    "Collate tables of from individual forecast dates, ",
    "following the prod directory structure"
  )
) |>
  add_argument(
    "dir_of_forecast_dirs",
    help = paste0(
      "Base directory containing subdirectories that represent ",
      "individual forecast dates, each of which in turn has ",
      "subdirectories that represent individual disease forecasts."
    )
  ) |>
  add_argument(
    "--save-dir",
    help = paste0(
      "Directory in which to save the results. If NULL or NA,",
      "use the dir_of_forecast_dirs"
    ),
    default = NA
  ) |>
  add_argument(
    "--score-file-stem",
    help = paste0(
      "Name of the score file to look for, ",
      "without the file extension or disease name"
    ),
    default = "score_table"
  ) |>
  add_argument(
    "--score-file-ext",
    help = "File extension for score files.",
    default = "rds"
  )

argv <- parse_args(p)

collate_and_save(
  argv$dir_of_forecast_dirs,
  argv$save_dir,
  score_file_stem = argv$score_file_stem,
  score_file_ext = argv$score_file_ext
)

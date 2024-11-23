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


process_loc_date_score_table <- function(model_run_dir,
                                         score_file_name,
                                         score_file_ext) {
  table_path <- fs::path(model_run_dir,
    score_file_name,
    ext = score_file_ext
  )
  parsed <- hewr::parse_model_run_dir_path(model_run_dir)
  location <- parsed$location

  if (!(fs::file_exists(table_path))) {
    warning(glue::glue(
      "No {score_file_name}.{score_file_ext} found for location ",
      "{location} in directory {model_run_dir}"
    ))
    return(NULL)
  }
  score_table <- readr::read_rds(table_path)

  ## add parsed metadata to both quantile and sample
  ## score tables
  for (x in names(parsed)) {
    score_table$quantile_scores[[x]] <- parsed[[x]]
    score_table$sample_scores[[x]] <- parsed[[x]]
  }

  return(score_table)
}



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

collate_scores_for_date <- function(model_run_dir,
                                    score_file_name = "score_table",
                                    score_file_ext = "rds",
                                    save = FALSE) {
  message(glue::glue("Processing scores from {model_run_dir}..."))
  locations_to_process <-
    fs::dir_ls(fs::path(model_run_dir, "model_runs"),
      type = "directory"
    )

  date_score_table <- purrr::map(
    locations_to_process,
    \(x) {
      process_loc_date_score_table(x,
        score_file_name = score_file_name,
        score_file_ext = score_file_ext
      )
    }
  ) |>
    bind_tables()

  if (save) {
    save_path <- fs::path(model_run_dir,
      score_file_name,
      ext = score_file_ext
    )
    message(glue::glue("Saving score table to {save_path}..."))
    readr::write_rds(date_score_table, save_path)
  }
  message(glue::glue("Done processing scores for {model_run_dir}."))
  return(date_score_table)
}


collate_all_score_tables <- function(model_base_dir,
                                     disease,
                                     score_file_name = "score_table",
                                     score_file_ext = "rds",
                                     score_file_save_path = NULL,
                                     save_batch_scores = FALSE) {
  date_dirs_to_process <- hewr::get_all_model_batch_dirs(
    model_base_dir,
    diseases = disease
  )

  # collate scores across locations for each date
  date_score_tables <- purrr::map(
    date_dirs_to_process,
    \(x) {
      collate_scores_for_date(
        x,
        score_file_name = score_file_name,
        score_file_ext = score_file_ext,
        save = save_batch_scores
      )
    }
  )

  # get all dates, annotate, and combine
  message(
    "Combining date-specific score tables ",
    "to create a full score table..."
  )

  full_score_table <- bind_tables(date_score_tables)

  if (!is.null(score_file_save_path)) {
    message(glue::glue(paste0(
      "Saving full score table to ",
      "{score_file_save_path}..."
    )))
    readr::write_rds(full_score_table, score_file_save_path)
  }

  message("Done creating full score table.")

  return(full_score_table)
}


p <- arg_parser(
  "Collate tables of scores into a single table across locations and dates."
) |>
  add_argument(
    "model_base_dir",
    help = paste0(
      "Base directory containing subdirectories that represent ",
      "individual forecast dates, each of which in turn has ",
      "subdirectories that represent individual location forecasts."
    )
  ) |>
  add_argument(
    "disease",
    help = paste0(
      "Name of the disease for which to collate scores."
    )
  ) |>
  add_argument(
    "--score-file-name",
    help = paste0(
      "Basename of the score file to look for, ",
      "without the file extension"
    ),
    default = "score_table"
  ) |>
  add_argument(
    "--score-file-ext",
    help = "File extension for score files.",
    default = "rds"
  ) |>
  add_argument(
    "--save-batch-scores",
    help = paste0(
      "Save collated scores within individual model ",
      "batch directories Default `TRUE`"
    ),
    default = TRUE
  )

argv <- parse_args(p)

collate_all_score_tables(
  argv$model_base_dir,
  argv$disease,
  score_file_name = argv$score_file_name,
  score_file_ext = argv$score_file_ext,
  score_file_save_path =
    fs::path(argv$model_base_dir,
      glue::glue("{argv$disease}_{argv$score_file_name}"),
      ext = argv$score_file_ext
    ),
  save_batch_scores = argv$save_batch_scores
)

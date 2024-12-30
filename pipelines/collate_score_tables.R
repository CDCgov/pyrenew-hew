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

save_scores <- function(score_table,
                        score_file_save_path = NULL) {
  if (!is.null(score_file_save_path)) {
    message(glue::glue(paste0(
      "Saving score table to ",
      "{score_file_save_path}..."
    )))
    readr::write_rds(score_table, score_file_save_path)
  }
}

read_score_file_if_exists <- function(dir,
                                      score_file_name,
                                      score_file_ext) {
  score_fp <- fs::path(dir, score_file_name, ext = score_file_ext)

  return(if (fs::file_exists(score_fp)) {
    readr::read_rds(score_fp)
  } else {
    NULL
  })
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


  save_path <- if (save) {
    fs::path(model_run_dir,
      score_file_name,
      ext = score_file_ext
    )
  } else {
    NULL
  }

  save_scores(date_score_table, save_path)

  message(glue::glue("Done processing scores for {model_run_dir}."))
  return(date_score_table)
}

collate_from_dirs <- function(dirs,
                              score_file_name,
                              score_file_ext,
                              score_file_save_path = NULL) {
  collated <- purrr::map(dirs, \(dir) {
    read_score_file_if_exists(dir, score_file_name, score_file_ext)
  }) |>
    bind_tables()

  save_scores(collated, score_file_save_path)

  return(collated)
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

  save_scores(full_score_table, score_file_save_path)

  message("Done creating full score table.")

  return(full_score_table)
}

main <- function(dir_of_forecast_dirs,
                 diseases = c("COVID-19", "Influenza"),
                 score_file_names = c(
                   "score_table",
                   "epiweekly_score_table"
                 ),
                 score_file_ext = "rds",
                 save_batch_scores = FALSE) {
  forecast_dirs <- fs::dir_ls(dir_of_forecast_dirs,
    type = "directory"
  )
  collate <- function(dir, filename, disease) {
    savename <- glue::glue("{disease}_{filename}")
    savepath <- fs::path(dir, savename, ext = score_file_ext)
    collate_all_score_tables(
      dir,
      disease,
      score_file_name = filename,
      score_file_ext = score_file_ext,
      score_file_save_path = savepath,
      save_batch_scores = save_batch_scores
    )
  }

  purrr::pwalk(
    tidyr::crossing(
      dir = forecast_dirs,
      filename = score_file_names,
      disease = diseases
    ),
    collate
  )
}


p <- arg_parser(
  "Collate tables of scores into a single table across locations and dates."
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
    "--diseases",
    help = paste0(
      "Name(s) of the disease(s) for which to collate scores, ",
      "as a whitespace-separated string."
    ),
    default = "COVID-19 Influenza"
  ) |>
  add_argument(
    "--score-file-names",
    help = paste0(
      "Basename(s) of the score file(s) to look for, ",
      "without the file extension, as a whitespace-separated string."
    ),
    default = "score_table epiweekly_score_table"
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

main(
  argv$dir_of_forecast_dirs,
  diseases = stringr::str_split_1(argv$diseases, " "),
  score_file_names = stringr::str_split_1(argv$score_file_names, " "),
  score_file_ext = argv$score_file_ext,
  save_batch_scores = argv$save_batch_scores
)

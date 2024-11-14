library(forecasttools)

inferencedata_to_tidy_draws(inference_data_path)

read_pyrenew_samples <- function(inference_data_path,
                                 filter_bad_chains = TRUE,
                                 good_chain_tol = 2) {
  arviz_split <- function(x) {
    x |>
      select(-distribution) |>
      split(f = as.factor(x$distribution))
  }

  pyrenew_samples <-
    read_csv(inference_data_path,
      show_col_types = FALSE
    ) |>
    rename_with(\(varname) str_remove_all(varname, "\\(|\\)|\\'|(, \\d+)")) |>
    rename(
      .chain = chain,
      .iteration = draw
    ) |>
    mutate(across(c(.chain, .iteration), \(x) as.integer(x + 1))) |>
    mutate(
      .draw = tidybayes:::draw_from_chain_and_iteration_(.chain, .iteration),
      .after = .iteration
    ) |>
    pivot_longer(-starts_with("."),
      names_sep = ", ",
      names_to = c("distribution", "name")
    ) |>
    arviz_split() |>
    map(\(x) pivot_wider(x, names_from = name) |> tidy_draws())

  if (filter_bad_chains) {
    good_chains <-
      pyrenew_samples$log_likelihood |>
      pivot_longer(-starts_with(".")) |>
      group_by(.iteration, .chain) |>
      summarize(value = sum(value)) |>
      group_by(.chain) |>
      summarize(value = mean(value)) |>
      filter(value >= max(value) - 2) |>
      pull(.chain)
  } else {
    good_chains <- unique(pyrenew_samples$log_likelihood$.chain)
  }

  good_pyrenew_samples <- map(
    pyrenew_samples,
    \(x) filter(x, .chain %in% good_chains)
  )
  good_pyrenew_samples
}

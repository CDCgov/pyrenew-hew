library(tidyverse)
library(tidybayes)
library(fs)
library(cmdstanr)
library(posterior)
library(jsonlite)
library(scales)
ci_width <- c(0.5, 0.8, 0.95)


hosp_data <- tibble(value = path(
  "notebooks", "data", "fit_hosp_only", "stan_data",
  ext = "json"
) |>
  jsonlite::read_json() |>
  pluck("hosp") |>
  unlist()) |>
  mutate(time = row_number())

stan_files <-
  dir_ls(path("notebooks", "data", "fit_hosp_only"), glob = "*wwinference*") |>
  enframe(name = NULL, value = "file_path") |>
  mutate(file_details = path_ext_remove(path_file(file_path))) |>
  separate_wider_delim(file_details,
    delim = "-",
    names = c("model", "date", "chain", "hash")
  ) |>
  mutate(date = ymd_hm(date)) |>
  filter(date == max(date)) |>
  pull(file_path)

stan_output <-
  dir_ls(path("notebooks", "data", "fit_hosp_only"), glob = "*wwinference*") |>
  read_cmdstan_csv()

pyrenew_output <- read_csv(path(
  "notebooks", "data", "fit_hosp_only", "pyrenew_inference_data",
  ext = "csv"
)) |>
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
  mutate(
    time = name |> str_extract("(?<=\\[)\\d+(?=\\])") |> as.integer() + 1,
    name = name |> str_remove("\\[\\d+\\]")
  )

pyrenew_output_tidy_ci <-
  pyrenew_output |>
  select(-starts_with(".")) |>
  group_by(distribution, name, time) |>
  median_qi(.width = ci_width)



stan_output_tidy_ci <-
  stan_output$post_warmup_draws |>
  tidy_draws() |>
  select(starts_with("pred_")) |>
  pivot_longer(everything()) |>
  mutate(
    time = name |> str_extract("(?<=\\[)\\d+(?=\\])") |> as.integer(),
    name = name |> str_remove("\\[\\d+\\]")
  ) |>
  group_by(name, time) |>
  median_qi(.width = ci_width)




stan_output_tidy_ci |>
  filter(name == "pred_hosp") |>
  ggplot(aes(time, value, ymin = .lower, ymax = .upper)) +
  geom_lineribbon()


combined_hosp_ci <- bind_rows(
  pyrenew_output_tidy_ci |>
    filter(
      distribution == "posterior_predictive",
      name == "observed_hospital_admissions"
    ) |>
    select(-distribution) |>
    mutate(model = "pyrenew"),
  filter(stan_output_tidy_ci, name == "pred_hosp") |>
    mutate(model = "stan")
) |>
  select(-name)

combined_hosp_ci |>
  ggplot(aes(time, value, ymin = .lower, ymax = .upper)) +
  facet_wrap(~model) +
  geom_lineribbon()

combined_hosp_ci |>
  ggplot(aes(time, value)) +
  facet_wrap(~model) +
  geom_lineribbon(aes(ymin = .lower, ymax = .upper), color = "#08519c") +
  scale_fill_brewer(
    name = "Credible Interval Width",
    labels = ~ percent(as.numeric(.))
  ) +
  geom_point(data = hosp_data) +
  cowplot::theme_cowplot() +
  ggtitle("Vignette Data Model Comparison") +
  scale_y_continuous("Hospital Admissions") +
  scale_x_continuous("Time") +
  theme(legend.position = "bottom")

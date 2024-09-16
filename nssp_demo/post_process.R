library(tidyverse)
library(tidybayes)
library(fs)
library(cowplot)
library(glue)
theme_set(theme_minimal_grid())
model_dir <- here(path(
  "nssp_demo",
  "private_data",
  "r_2024-09-10_f_2024-03-13_l_2024-09-09_t_2024-08-14",
  "CA"
))

state_abb <- model_dir %>%
  path_split() %>%
  pluck(1) %>%
  tail(1)


data_path <- path(model_dir, "data", ext = "csv")
posterior_samples_path <- path(model_dir, "pyrenew_inference_data", ext = "csv")


dat <- read_csv(data_path) %>%
  arrange(date) %>%
  mutate(time = row_number() - 1) %>%
  rename(.value = COVID_ED_admissions)

last_training_date <- dat %>%
  filter(data_type == "train") %>%
  pull(date) %>%
  max()
last_data_date <- dat %>%
  pull(date) %>%
  max()

arviz_split <- function(x) {
  x %>%
    select(-distribution) %>%
    split(f = as.factor(x$distribution))
}

pyrenew_samples <-
  read_csv(posterior_samples_path) %>%
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


hosp_ci <-
  pyrenew_samples$posterior_predictive %>%
  gather_draws(observed_hospital_admissions[time]) %>%
  median_qi(.width = c(0.5, 0.8, 0.95)) %>%
  mutate(date = dat$date[time + 1])



library(scales)
forecast_plot <-
  ggplot(mapping = aes(date, .value)) +
  geom_lineribbon(
    data = hosp_ci,
    mapping = aes(ymin = .lower, ymax = .upper),
    color = "#08519c", key_glyph = draw_key_rect
  ) +
  geom_point(mapping = aes(shape = data_type), data = dat) +
  scale_y_continuous("Emergency Department Admissions") +
  scale_x_date("Date") +
  scale_fill_brewer(
    name = "Credible Interval Width",
    labels = ~ percent(as.numeric(.))
  ) +
  scale_shape_discrete("Data Type", labels = str_to_title) +
  geom_vline(xintercept = last_training_date, linetype = "dashed") +
  annotate(
    geom = "text",
    x = last_training_date,
    y = -Inf,
    label = "Fit Period ←\n",
    hjust = "right",
    vjust = "bottom"
  ) +
  annotate(
    geom = "text",
    x = last_training_date,
    y = -Inf, label = "→ Forecast Period\n",
    hjust = "left",
    vjust = "bottom",
  ) +
  ggtitle(glue("NSSP-based forecast for {state_abb}"),
    subtitle = glue("as of {last_data_date}")
  ) +
  theme(legend.position = "bottom")

save_plot(
  filename = path(model_dir, "forecast_plot", ext = "pdf"),
  plot = forecast_plot,
  device = cairo_pdf, base_height = 6
)

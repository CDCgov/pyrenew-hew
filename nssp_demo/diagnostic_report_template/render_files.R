library(tidyverse)
library(fs)


setwd("nssp_demo/diagnostic_report_template")
quarto_template_path <- path("demo.qmd")
base_dir <- path(
  "~/pyrenew-hew", "nssp_demo",
  "private_data",
  "pyrenew-test-output",
  "influenza_r_2024-11-06_f_2024-08-18_t_2024-10-31"
)
# parse this from CLI
site_output_dir <- path_file(base_dir)
dir_create(site_output_dir)

quarto_render_tbl <- tibble(model_dir = dir_ls(base_dir, type = "directory")) |>
  mutate(state = path_file(model_dir)) |>
  mutate(output_file = path(state, ext = "html")) |>
  select(model_dir, output_file) |>
  head(2)

quarto_render_tbl |> pwalk(function(model_dir, output_file) {
  quarto_render(
    input = quarto_template_path,
    output_file = output_file,
    execute_params = list(model_dir_raw = model_dir)
  )
})


fs::file_move(
  path = quarto_render_tbl$output_file,
  new_path = path(
    site_output_dir,
    quarto_render_tbl$output_file
  )
)
fs::dir_move("demo_files", path(site_output_dir, "demo_files"))
# The problem is demo_files gets replaced on every render
# I think this could be better done using a quarto project or using profiles
# or quarto project profiles?
# https://quarto.org/docs/projects/quarto-projects.html
# https://github.com/quarto-dev/quarto-cli/discussions/7805
# https://github.com/quarto-dev/quarto-cli/discussions/5654

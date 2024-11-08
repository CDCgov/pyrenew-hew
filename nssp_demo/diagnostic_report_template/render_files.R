library(tidyverse)
library(fs)

diagnostic_report_dir <- path("nssp_demo", "diagnostic_report_template")

base_dir <- path(
  "~/pyrenew-hew", "nssp_demo",
  "private_data",
  "pyrenew-test-output",
  "influenza_r_2024-11-06_f_2024-08-18_t_2024-10-31"
)
# parse this from CLI


site_output_dir <- path(diagnostic_report_dir, path_file(base_dir))
dir_create(site_output_dir)

quarto_render_tbl <-
  tibble(state_dir = dir_ls(base_dir, type = "directory")) |>
  mutate(qmd_path = path(site_output_dir, path_file(state_dirs), ext = "qmd") |>
    path_rel(diagnostic_report_dir))



original_wd <- getwd()
setwd(diagnostic_report_dir)
quarto_template_path <- path("demo", ext = "qmd")
walk(quarto_render_tbl$qmd_path, \(x) file_copy(quarto_template_path, x))
pwalk(
  quarto_render_tbl,
  function(state_dir, qmd_path) {
    quarto_render(
      input = qmd_path,
      execute_params = list(model_dir_raw = state_dir)
    )
  }
)

file_delete(quarto_render_tbl$qmd_path)
setwd(original_wd)

library(tidyverse)
library(fs)
library(quarto)

base_dir <- path(
  "~/pyrenew-hew",
  "private_data",
  "pyrenew-test-output",
  "influenza_r_2024-11-19_f_2024-08-16_t_2024-11-13"
)
# parse this from CLI

# The site should be contained in a single directory for easy linking between
# pages and sharing html files
site_output_dir <- path(base_dir, "diagnostic_report")
template_dir <- dir <- path("pipelines", "diagnostic_report")
css_file_name <- path("custom", ext = "scss")

template_css_path <- path(template_dir, css_file_name) |> path_real()
site_output_css_path <- path(site_output_dir, css_file_name)
template_qmd_path <- path(template_dir, "template", ext = "qmd")


site_output_css_path_real <- tryCatch(
  path_real(site_output_css_path),
  error = function(e) {
    message("An error occurred: ", e$message)
    FALSE
  }
)

# Temporarily create template css in working directory
# otherwise quarto_render won't be able to find it
if (template_css_path != site_output_css_path_real) {
  file_copy(template_css_path, site_output_css_path, overwrite = TRUE)
}


quarto_render_tbl <-
  tibble(state_dir = dir_ls(path(base_dir, "model_runs"),
    type = "directory"
  )) |>
  mutate(qmd_path = path(site_output_dir, path_file(state_dir), ext = "qmd"))

dir_create(site_output_dir)

# Copy template with new file names to output directory
walk(quarto_render_tbl$qmd_path, function(x) {
  file_copy(template_qmd_path, x, overwrite = TRUE)
})

# Render all qmd's
pwalk(
  quarto_render_tbl,
  function(state_dir, qmd_path) {
    quarto_render(
      input = qmd_path,
      execute_params = list(model_dir_raw = state_dir)
    )
  }
)

# Delete qmd's
file_delete(quarto_render_tbl$qmd_path)

# Clean up css in working directory
if (template_css_path != site_output_css_path_real) {
  file_delete(site_output_css_path)
}

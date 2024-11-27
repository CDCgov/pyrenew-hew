script_packages <- c(
  "argparser",
  "fs",
  "hewr",
  "quarto"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})

render_webpage <- function(model_run_dir, template_qmd_path) {
  model_run_dir <- path_real(model_run_dir)
  template_qmd_path <- path_real(template_qmd_path)
  location <- parse_model_run_dir_path(model_run_dir)$location

  model_batch_dir <- model_run_dir |>
    path_dir() |>
    path_dir()

  site_dir <- path(model_batch_dir, "diagnostic_report")

  template_dir <- path_dir(template_qmd_path)
  css_files <- dir_ls(template_dir, regexp = "\\.s?css$")
  page_css <- path(site_dir, path_file(css_files))
  page_qmd <- path(site_dir, location, ext = "qmd")
  page_html <- page_qmd |> path_ext_set("html")

  dir_create(site_dir)

  file_copy(path = css_files, new_path = page_css, overwrite = TRUE)
  file_copy(path = template_qmd_path, page_qmd, overwrite = TRUE)

  quarto_render(
    input = page_qmd,
    execute_params = list(model_dir_raw = model_run_dir)
  )
  message("rendered ", page_html)

  file_delete(page_qmd)
}

p <- arg_parser("Render diagnostic dashboard for a single location forecast") |>
  add_argument(
    "model_run_dir",
    help = "Directory containing the model data and output."
  ) |>
  add_argument("--template_qmd_path",
    help = "Path to template qmd",
    default = "pipelines/diagnostic_report/template.qmd"
  )


argv <- parse_args(p)
message(argv$model_run_dir)
message(argv$template_qmd_path)
render_webpage(
  model_run_dir = path_real(argv$model_run_dir),
  template_qmd_path = path_real(argv$template_qmd_path)
)

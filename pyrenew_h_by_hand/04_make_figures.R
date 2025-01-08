library(fs)
library(tidyverse)
library(forecasttools)
library(hewr)
library(glue)
p <- arg_parser("Process model batch directories")
p <- add_argument(p, "super_dir", help = "Directory containing model batch directories")
argv <- parse_args(p)
super_dir <- path(argv$super_dir)

hubverse_tables <- dir_ls(dir_ls(super_dir), glob = "*hubverse-table.csv")


observed_data_url_key <- c(
  "COVID-19" = "https://raw.githubusercontent.com/CDCgov/covid19-forecast-hub/refs/heads/main/target-data/covid-hospital-admissions.csv",
  "Influenza" = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/refs/heads/main/target-data/target-hospital-admissions.csv"
)

walk(hubverse_tables, \(hubverse_table) {
  model_batch_dir <- path_dir(hubverse_table)
  model_info <- parse_model_batch_dir_path(model_batch_dir)


  all_plots <- plot_hubverse_file_quantiles(hubverse_table,
    observed_data_path = observed_data_url_key[model_info[["disease"]]],
    start_date = model_info[["first_training_date"]]
  )

  plot_save_tbl <- tibble(
    plot_obj = all_plots,
    file_path = path(model_batch_dir,
      names(all_plots),
      ext = "pdf"
    )
  )
  walk2(
    plot_save_tbl$plot_obj,
    plot_save_tbl$file_path,
    \(plot_obj, name) cowplot::save_plot(name,
      plot = plot_obj,
      base_height = 4
    )
  )


  combined_pdf_name <- glue("{model_info[['report_date']]}-{str_to_lower(model_info[['disease']])}-hubverse-plots")


  system2("pdfunite",
    args = c(
      plot_save_tbl$file_path,
      path(model_batch_dir, combined_pdf_name, ext = "pdf")
    )
  )

  file_delete(plot_save_tbl$file_path)
})

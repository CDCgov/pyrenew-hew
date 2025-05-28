library(tidyverse)
library(arrow)
library(fs)
private_data_dir <- path("pipelines/tests/end_to_end_test_output/private_data")

dir_ls(private_data_dir, recurse = TRUE, type = "file") |> path_split() |> map_chr(5)
dat <- tibble(file_path = dir_ls(private_data_dir, recurse = TRUE, type = "file")) |> 
  mutate(name = file_path |> path_split() |> map_chr(5),
         value = map(file_path, read_parquet)) |> 
  select(name, value) |> 
  deframe()



# nssp-etl ----------------------------------------------------------------
# # A tibble: 6 × 7
# reference_date report_date geo_type geo_value metric          disease          value
# <date>         <date>      <chr>    <chr>     <chr>           <chr>            <int>
# 1 2024-06-01     2025-01-18  state    CA        count_ed_visits COVID-19/Omicron    37
# 2 2024-06-01     2025-01-18  state    CA        count_ed_visits Influenza           35
# 3 2024-06-01     2025-01-18  state    CA        count_ed_visits RSV                 41
# 4 2024-06-01     2025-01-18  state    CA        count_ed_visits Total             1927
# 5 2024-06-01     2025-01-18  state    MT        count_ed_visits COVID-19/Omicron    19
# 6 2024-06-01     2025-01-18  state    MT        count_ed_visits Influenza           28



# nssp_etl_gold -----------------------------------------------------------
# # A tibble: 6 × 10
# reference_date report_date geo_type geo_value asof       metric          run_id facility disease          value
# <date>         <date>      <chr>    <chr>     <date>     <chr>            <dbl>    <int> <chr>            <int>
# 1 2024-06-01     2024-12-21  state    CA        2024-12-21 count_ed_visits      0        1 COVID-19/Omicron    10
# 2 2024-06-01     2024-12-21  state    CA        2024-12-21 count_ed_visits      0        1 Total              220
# 3 2024-06-02     2024-12-21  state    CA        2024-12-21 count_ed_visits      0        1 COVID-19/Omicron    21
# 4 2024-06-02     2024-12-21  state    CA        2024-12-21 count_ed_visits      0        1 Total              213
# 5 2024-06-03     2024-12-21  state    CA        2024-12-21 count_ed_visits      0        1 COVID-19/Omicron    13
# 6 2024-06-03     2024-12-21  state    CA        2024-12-21 count_ed_visits      0        1 Total              223

# nssp_state_level_gold ---------------------------------------------------
# # A tibble: 6 × 8
# reference_date report_date geo_type geo_value metric          disease          value any_update_this_day
# <date>         <date>      <chr>    <chr>     <chr>           <chr>            <int> <lgl>              
# 1 2024-06-01     2025-01-18  state    CA        count_ed_visits COVID-19/Omicron    37 TRUE               
# 2 2024-06-01     2025-01-18  state    CA        count_ed_visits Influenza           35 TRUE               
# 3 2024-06-01     2025-01-18  state    CA        count_ed_visits RSV                 41 TRUE               
# 4 2024-06-01     2025-01-18  state    CA        count_ed_visits Total             1927 TRUE               
# 5 2024-06-01     2025-01-18  state    MT        count_ed_visits COVID-19/Omicron    19 TRUE               
# 6 2024-06-01     2025-01-18  state    MT        count_ed_visits Influenza           28 TRUE               


# nwss_vintages -----------------------------------------------------------
# # A tibble: 6 × 12
# sample_collect_date lab_id wwtp_id pcr_target_avg_conc sample_location sample_matrix  pcr_target_units   pcr_target wwtp_jurisdiction population_served quality_flag lod_sewage
# <date>               <dbl>   <dbl>               <dbl> <chr>           <chr>          <chr>              <chr>      <chr>                         <int> <chr>             <dbl>
# 1 2024-11-10              11      11             79.6    wwtp            raw wastewater copies/l wastewat… sars-cov-2 CA                          2000000 NA                 4.51
# 2 2024-12-02              12      12              0.0159 wwtp            raw wastewater copies/l wastewat… sars-cov-2 CA                          1000000 n                  6.54
# 3 2024-11-29              10      10             19.6    wwtp            raw wastewater copies/l wastewat… sars-cov-2 CA                          4000000 no                 3.62
# 4 2024-10-26              10      10           1497.     wwtp            raw wastewater copies/l wastewat… sars-cov-2 CA                          4000000 no                 3.62
# 5 2024-11-29              13      13              1.12   wwtp            raw wastewater copies/l wastewat… sars-cov-2 CA                           500000 n                  5.95
# 6 2024-11-30              12      12              2.78   wwtp            raw wastewater copies/l wastewat… sars-cov-2 CA                          1000000 n                  6.54


# prod_param_estimates ----------------------------------------------------
# # A tibble: 6 × 9
# id start_date end_date reference_date disease   format parameter           geo_value          value
# <dbl> <date>     <lgl>    <date>         <chr>     <chr>  <chr>               <chr>     <list<double>>
# 1     0 2024-06-01 NA       2024-12-21     COVID-19  PMF    generation_interval NA                   [7]
# 2     0 2024-06-01 NA       2024-12-21     COVID-19  PMF    delay               NA                  [12]
# 3     1 2024-06-01 NA       2024-12-21     COVID-19  PMF    right_truncation    MT                   [4]
# 4     2 2024-06-01 NA       2024-12-21     COVID-19  PMF    right_truncation    CA                   [4]
# 5     3 2024-06-01 NA       2024-12-21     COVID-19  PMF    right_truncation    US                   [4]
# 6     0 2024-06-01 NA       2024-12-21     Influenza PMF    generation_interval NA                   [7]
# Required Data


``` r
library(tidyverse)
```

    ── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
    ✔ dplyr     1.1.4     ✔ readr     2.1.5
    ✔ forcats   1.0.0     ✔ stringr   1.5.1
    ✔ ggplot2   3.5.2     ✔ tibble    3.2.1
    ✔ lubridate 1.9.4     ✔ tidyr     1.3.1
    ✔ purrr     1.0.4
    ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ✖ dplyr::filter() masks stats::filter()
    ✖ dplyr::lag()    masks stats::lag()
    ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors

``` r
library(arrow)
```


    Attaching package: 'arrow'

    The following object is masked from 'package:lubridate':

        duration

    The following object is masked from 'package:utils':

        timestamp

``` r
library(fs)
library(here)
```

    here() starts at /Users/damon/Documents/GitHub/pyrenew-hew

``` r
private_data_dir <- here("pipelines/tests/end_to_end_test_output/private_data")

private_data_dir |>
  dir_ls(recurse = TRUE, type = "file") |>
  path_rel(private_data_dir)
```

    nssp-etl/latest_comprehensive.parquet
    nssp_etl_gold/2024-12-21.parquet
    nssp_state_level_gold/2024-12-21.parquet
    nwss_vintages/NWSS-ETL-covid-2024-12-21/bronze.parquet
    prod_param_estimates/prod.parquet

``` r
dat <- tibble(
  file_path = dir_ls(private_data_dir, recurse = TRUE, type = "file")
) |>
  mutate(
    name = file_path |> path_split() |> map_chr(11),
    value = map(file_path, read_parquet)
  ) |>
  select(name, value) |>
  deframe()

dat$latest_comprehensive <- dat$`nssp-etl`
dat$`nssp-etl` <- NULL
```

``` r
head(dat$nssp_etl_gold)
```

    # A tibble: 6 × 10
      reference_date report_date geo_type geo_value asof       metric         run_id
      <date>         <date>      <chr>    <chr>     <date>     <chr>           <dbl>
    1 2024-06-01     2024-12-21  state    CA        2024-12-21 count_ed_visi…      0
    2 2024-06-01     2024-12-21  state    CA        2024-12-21 count_ed_visi…      0
    3 2024-06-02     2024-12-21  state    CA        2024-12-21 count_ed_visi…      0
    4 2024-06-02     2024-12-21  state    CA        2024-12-21 count_ed_visi…      0
    5 2024-06-03     2024-12-21  state    CA        2024-12-21 count_ed_visi…      0
    6 2024-06-03     2024-12-21  state    CA        2024-12-21 count_ed_visi…      0
    # ℹ 3 more variables: facility <int>, disease <chr>, value <int>

``` r
dat$nssp_etl_gold |>
  distinct(geo_type, geo_value, asof, metric, run_id, facility, disease)
```

    # A tibble: 20 × 7
       geo_type geo_value asof       metric          run_id facility disease
       <chr>    <chr>     <date>     <chr>            <dbl>    <int> <chr>
     1 state    CA        2024-12-21 count_ed_visits      0        1 COVID-19/Omicr…
     2 state    CA        2024-12-21 count_ed_visits      0        1 Total
     3 state    CA        2024-12-21 count_ed_visits      0        1 Influenza
     4 state    CA        2024-12-21 count_ed_visits      0        1 RSV
     5 state    CA        2024-12-21 count_ed_visits      0        2 COVID-19/Omicr…
     6 state    CA        2024-12-21 count_ed_visits      0        2 Total
     7 state    CA        2024-12-21 count_ed_visits      0        2 Influenza
     8 state    CA        2024-12-21 count_ed_visits      0        2 RSV
     9 state    CA        2024-12-21 count_ed_visits      0        3 COVID-19/Omicr…
    10 state    CA        2024-12-21 count_ed_visits      0        3 Total
    11 state    CA        2024-12-21 count_ed_visits      0        3 Influenza
    12 state    CA        2024-12-21 count_ed_visits      0        3 RSV
    13 state    MT        2024-12-21 count_ed_visits      0        4 COVID-19/Omicr…
    14 state    MT        2024-12-21 count_ed_visits      0        4 Total
    15 state    MT        2024-12-21 count_ed_visits      0        4 Influenza
    16 state    MT        2024-12-21 count_ed_visits      0        4 RSV
    17 state    MT        2024-12-21 count_ed_visits      0        5 COVID-19/Omicr…
    18 state    MT        2024-12-21 count_ed_visits      0        5 Total
    19 state    MT        2024-12-21 count_ed_visits      0        5 Influenza
    20 state    MT        2024-12-21 count_ed_visits      0        5 RSV

``` r
head(dat$nssp_state_level_gold)
```

    # A tibble: 6 × 8
      reference_date report_date geo_type geo_value metric          disease    value
      <date>         <date>      <chr>    <chr>     <chr>           <chr>      <int>
    1 2024-06-01     2025-01-18  state    CA        count_ed_visits COVID-19/…    37
    2 2024-06-01     2025-01-18  state    CA        count_ed_visits Influenza     35
    3 2024-06-01     2025-01-18  state    CA        count_ed_visits RSV           41
    4 2024-06-01     2025-01-18  state    CA        count_ed_visits Total       1927
    5 2024-06-01     2025-01-18  state    MT        count_ed_visits COVID-19/…    19
    6 2024-06-01     2025-01-18  state    MT        count_ed_visits Influenza     28
    # ℹ 1 more variable: any_update_this_day <lgl>

``` r
dat$nssp_state_level_gold |>
  distinct(geo_type, geo_value, metric, disease, any_update_this_day)
```

    # A tibble: 8 × 5
      geo_type geo_value metric          disease          any_update_this_day
      <chr>    <chr>     <chr>           <chr>            <lgl>
    1 state    CA        count_ed_visits COVID-19/Omicron TRUE
    2 state    CA        count_ed_visits Influenza        TRUE
    3 state    CA        count_ed_visits RSV              TRUE
    4 state    CA        count_ed_visits Total            TRUE
    5 state    MT        count_ed_visits COVID-19/Omicron TRUE
    6 state    MT        count_ed_visits Influenza        TRUE
    7 state    MT        count_ed_visits RSV              TRUE
    8 state    MT        count_ed_visits Total            TRUE

``` r
head(dat$nwss_vintages)
```

    # A tibble: 6 × 12
      sample_collect_date lab_id wwtp_id pcr_target_avg_conc sample_location
      <date>               <dbl>   <dbl>               <dbl> <chr>
    1 2024-11-10              11      11             79.6    wwtp
    2 2024-12-02              12      12              0.0159 wwtp
    3 2024-11-29              10      10             19.6    wwtp
    4 2024-10-26              10      10           1497.     wwtp
    5 2024-11-29              13      13              1.12   wwtp
    6 2024-11-30              12      12              2.78   wwtp
    # ℹ 7 more variables: sample_matrix <chr>, pcr_target_units <chr>,
    #   pcr_target <chr>, wwtp_jurisdiction <chr>, population_served <int>,
    #   quality_flag <chr>, lod_sewage <dbl>

``` r
dat$nwss_vintages |>
  distinct(
    lab_id,
    wwtp_id,
    sample_location,
    sample_matrix,
    pcr_target_units,
    pcr_target,
    wwtp_jurisdiction,
    population_served,
    quality_flag,
  )
```

    # A tibble: 8 × 9
      lab_id wwtp_id sample_location sample_matrix  pcr_target_units    pcr_target
       <dbl>   <dbl> <chr>           <chr>          <chr>               <chr>
    1     11      11 wwtp            raw wastewater copies/l wastewater sars-cov-2
    2     12      12 wwtp            raw wastewater copies/l wastewater sars-cov-2
    3     10      10 wwtp            raw wastewater copies/l wastewater sars-cov-2
    4     13      13 wwtp            raw wastewater copies/l wastewater sars-cov-2
    5      3       3 wwtp            raw wastewater copies/l wastewater sars-cov-2
    6      0       0 wwtp            raw wastewater copies/l wastewater sars-cov-2
    7      1       1 wwtp            raw wastewater copies/l wastewater sars-cov-2
    8      2       2 wwtp            raw wastewater copies/l wastewater sars-cov-2
    # ℹ 3 more variables: wwtp_jurisdiction <chr>, population_served <int>,
    #   quality_flag <chr>

``` r
head(dat$prod_param_estimates)
```

    # A tibble: 6 × 9
         id start_date end_date reference_date disease   format parameter  geo_value
      <dbl> <date>     <lgl>    <date>         <chr>     <chr>  <chr>      <chr>
    1     0 2024-06-01 NA       2024-12-21     COVID-19  PMF    generatio… <NA>
    2     0 2024-06-01 NA       2024-12-21     COVID-19  PMF    delay      <NA>
    3     1 2024-06-01 NA       2024-12-21     COVID-19  PMF    right_tru… MT
    4     2 2024-06-01 NA       2024-12-21     COVID-19  PMF    right_tru… CA
    5     3 2024-06-01 NA       2024-12-21     COVID-19  PMF    right_tru… US
    6     0 2024-06-01 NA       2024-12-21     Influenza PMF    generatio… <NA>
    # ℹ 1 more variable: value <list<double>>

``` r
dat$prod_param_estimates |>
  distinct(reference_date, disease, format, parameter, geo_value)
```

    # A tibble: 15 × 5
       reference_date disease   format parameter           geo_value
       <date>         <chr>     <chr>  <chr>               <chr>
     1 2024-12-21     COVID-19  PMF    generation_interval <NA>
     2 2024-12-21     COVID-19  PMF    delay               <NA>
     3 2024-12-21     COVID-19  PMF    right_truncation    MT
     4 2024-12-21     COVID-19  PMF    right_truncation    CA
     5 2024-12-21     COVID-19  PMF    right_truncation    US
     6 2024-12-21     Influenza PMF    generation_interval <NA>
     7 2024-12-21     Influenza PMF    delay               <NA>
     8 2024-12-21     Influenza PMF    right_truncation    MT
     9 2024-12-21     Influenza PMF    right_truncation    CA
    10 2024-12-21     Influenza PMF    right_truncation    US
    11 2024-12-21     RSV       PMF    generation_interval <NA>
    12 2024-12-21     RSV       PMF    delay               <NA>
    13 2024-12-21     RSV       PMF    right_truncation    MT
    14 2024-12-21     RSV       PMF    right_truncation    CA
    15 2024-12-21     RSV       PMF    right_truncation    US

``` r
head(dat$latest_comprehensive)
```

    # A tibble: 6 × 7
      reference_date report_date geo_type geo_value metric          disease    value
      <date>         <date>      <chr>    <chr>     <chr>           <chr>      <int>
    1 2024-06-01     2025-01-18  state    CA        count_ed_visits COVID-19/…    37
    2 2024-06-01     2025-01-18  state    CA        count_ed_visits Influenza     35
    3 2024-06-01     2025-01-18  state    CA        count_ed_visits RSV           41
    4 2024-06-01     2025-01-18  state    CA        count_ed_visits Total       1927
    5 2024-06-01     2025-01-18  state    MT        count_ed_visits COVID-19/…    19
    6 2024-06-01     2025-01-18  state    MT        count_ed_visits Influenza     28

``` r
dat$latest_comprehensive |> distinct(geo_type, metric, geo_value, disease)
```

    # A tibble: 8 × 4
      geo_type metric          geo_value disease
      <chr>    <chr>           <chr>     <chr>
    1 state    count_ed_visits CA        COVID-19/Omicron
    2 state    count_ed_visits CA        Influenza
    3 state    count_ed_visits CA        RSV
    4 state    count_ed_visits CA        Total
    5 state    count_ed_visits MT        COVID-19/Omicron
    6 state    count_ed_visits MT        Influenza
    7 state    count_ed_visits MT        RSV
    8 state    count_ed_visits MT        Total

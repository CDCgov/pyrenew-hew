# NSSP Demo Workflow

## 1. Prepare data


### Now

`prep_data.R` reads in a `private_data/report_date.parquet`
and `private_data/prod.parquet` from disk (what's in these files?).
It provides a function `prep_data`
that takes the arguments: `disease`, `report_date`, `min_reference_date`,
`max_reference_date`, `last_training_date`, `state_abb`.

To create a dataframe (for plotting) and a `data_for_model_fit` list (data that is
read in the model fitting step).

The function `prep_and_save_data` has the same arguments as `prep_data` and
saves the results in `private_data/{str_to_lower(disease)}_r_{report_date}_f_{actual_first_date}_l_{actual_last_date}_t_{last_training_date}/{state_abb}`.

`disease`, `report_date`, `min_reference_date`, `max_reference_date`, and `last_training_date` are all specified in the script.

The script uses `purrr::walk` to save data for each `state_abb`.

### In the future

`disease`, `report_date`, `min_reference_date`, `max_reference_date`, and
`last_training_date`, and `state_abb` should be specified as command line
arguments.

The path to `report_date.parquet` and `prod.parquet` should be specified as
command line arguments.

Eventually, `report_date.parquet` and `prod.parquet` should be read from azure blob.

## 2. Fitting the model

### Now

Models are fit by calling `python fit_model.py --model_dir MODEL_DIR` from the
command line, where MODEL_DIR is of the form `private_data/{str_to_lower(disease)}_r_{report_date}_f_{actual_first_date}_l_{actual_last_date}_t_{last_training_date}/{state_abb}`

The model is fit, non-converging chains are pruned, and forecasting is
performed. The results are saved as a csv in `private_data/{str_to_lower(disease)}_r_{report_date}_f_{actual_first_date}_l_{actual_last_date}_t_{last_training_date}/{state_abb}/pyrenew_inference_data.csv`


### In the future

Model fitting and forecasting should be separated into different scripts. All
chains should be saved, regardless of convergence.

## 3. Post-processing

### Now

`post_process.R` contains a function `make_forecast_fig` that takes `model_dir`
as an argument. It creates a forecast plot.

The script uses `purrr::map` and `purrr::pwalk` to create and save forecast plots for
every sub-directory in
`private_data/{str_to_lower(disease)}_r_{report_date}_f_{actual_first_date}_l_{actual_last_date}_t_{last_training_date}`.

It then uses `pdfunnite` to save a combined pdf.

### In the future

More plots and diagnoistics should be added.
There should be some intermediate script that prepares model output for
plotting, which can be run on each model in parallel.

Then there can be one final script to create the figures and other diagnostics,
which may involve combining data from multiple model fits.

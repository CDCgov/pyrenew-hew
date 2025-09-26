import argparse
import datetime as dt
import pickle
from pathlib import Path

import arviz as az
import forecasttools as ft
import polarbayes as pb
import polars as pl

from pipelines.utils import build_pyrenew_hew_model_from_dir
from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData
from pyrenew_hew.utils import (
    flags_from_pyrenew_model_name,
)


def generate_and_save_predictions(
    model_run_dir: str | Path,
    model_name: str,
    n_forecast_points: int,
    predict_ed_visits: bool = False,
    predict_hospital_admissions: bool = False,
    predict_wastewater: bool = False,
) -> None:
    model_run_dir = Path(model_run_dir)
    model_dir = Path(model_run_dir, model_name)
    if not model_dir.exists():
        raise FileNotFoundError(f"The directory {model_dir} does not exist.")
    mcmc_output_dir = model_dir / "mcmc_output"
    mcmc_output_dir.mkdir(parents=True, exist_ok=True)

    my_data = PyrenewHEWData.from_json(
        json_file_path=Path(model_dir) / "data" / "data_for_model_fit.json",
        **flags_from_pyrenew_model_name(model_name),
    )

    my_model = build_pyrenew_hew_model_from_dir(
        model_dir,
        **flags_from_pyrenew_model_name(model_name),
    )

    my_model._init_model(1, 1)
    fresh_sampler = my_model.mcmc.sampler

    with open(
        model_dir / "posterior_samples.pickle",
        "rb",
    ) as file:
        my_model.mcmc = pickle.load(file)

    my_model.mcmc.sampler = fresh_sampler
    forecast_data = my_data.to_forecast_data(n_forecast_points)

    posterior_predictive = my_model.posterior_predictive(
        data=forecast_data,
        sample_ed_visits=predict_ed_visits,
        sample_hospital_admissions=predict_hospital_admissions,
        sample_wastewater=predict_wastewater,
    )

    idata = az.from_numpyro(my_model.mcmc, posterior_predictive=posterior_predictive)

    ft.arviz.replace_all_dim_suffix(idata, ["time", "site_id"], inplace=True)

    date_details_df = pl.DataFrame(
        {
            "dim_name": [
                "observed_ed_visits_time",
                "observed_hospital_admissions_time",
                "site_level_log_ww_conc_time",
            ],
            "start_date": [
                forecast_data.first_data_dates["ed_visits"].astype(dt.datetime),
                forecast_data.first_data_dates["hospital_admissions"].astype(
                    dt.datetime
                ),
                forecast_data.first_data_dates["wastewater"].astype(dt.datetime),
            ],
            "interval": [
                dt.timedelta(days=my_data.nssp_step_size),
                dt.timedelta(days=my_data.nhsn_step_size),
                dt.timedelta(days=my_data.nwss_step_size),
            ],
        }
    ).filter(pl.col("dim_name").is_in(ft.arviz.get_all_dims(idata)))

    for row in date_details_df.iter_rows(named=True):
        ft.arviz.assign_coords_from_start_step(idata, **row, inplace=True)

    # Save one netcdf for reloading
    idata.to_netcdf(str(mcmc_output_dir / "original_inference_data.nc"))
    ft.arviz.prune_chains_by_rel_diff(idata, rel_diff_thresh=0.9, inplace=True)

    idata.to_netcdf(str(mcmc_output_dir / "inference_data.nc"))

    tidy_posterior_predictive = (
        pb.gather_draws(
            idata,
            group="posterior_predictive",
            var_names=date_details_df.get_column("dim_name")
            .str.strip_suffix("_time")
            .to_list(),
        )
        .pipe(ft.coalesce_common_columns, "_time", "date")
        .rename({"site_level_log_ww_conc_site_id": "lab_site_index"}, strict=False)
    )

    tidy_posterior_predictive.write_parquet(
        mcmc_output_dir / "tidy_posterior_predictive.parquet"
    )

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Do posterior prediction from a pyrenew-hew fit.")
    )
    parser.add_argument(
        "model_run_dir",
        type=Path,
        help=(
            "Path to a directory containing the model fitting data "
            "and the posterior chains. "
            "The completed predictive samples will be saved here."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use for generating predictions.",
    )
    parser.add_argument(
        "--n-forecast-points",
        type=int,
        default=0,
        help="Number of time points to forecast (Default: 0).",
    )
    parser.add_argument(
        "--predict-ed-visits",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="If provided, generate posterior predictions for ED visits.",
    )
    parser.add_argument(
        "--predict-hospital-admissions",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help=("If provided, generate posterior predictions for hospital admissions."),
    )
    parser.add_argument(
        "--predict-wastewater",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="If provided, generate posterior predictions for wastewater.",
    )

    args = parser.parse_args()

    generate_and_save_predictions(**vars(args))

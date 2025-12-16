import datetime as dt
import json
from pathlib import Path
from typing import Self

import jax.numpy as jnp
import numpy as np
import polars as pl
from jax.typing import ArrayLike
from pyrenew.time import (
    create_date_time_spine,
    get_end_date,
    get_n_data_days,
    validate_mmwr_dates,
)


class PyrenewHEWData:
    """
    Class for holding input data
    to a PyrenewHEW model.
    """

    def __init__(
        self,
        nssp_training_data: pl.DataFrame | None = None,
        nhsn_training_data: pl.DataFrame | None = None,
        nwss_training_data: pl.DataFrame | None = None,
        nssp_step_size: int | None = None,
        nhsn_step_size: int | None = None,
        nwss_step_size: int | None = None,
        n_ed_visits_data_days: int | None = None,
        n_hospital_admissions_data_days: int | None = None,
        n_wastewater_data_days: int | None = None,
        first_ed_visits_date: np.datetime64 | None = None,
        first_hospital_admissions_date: np.datetime64 | None = None,
        first_wastewater_date: np.datetime64 | None = None,
        right_truncation_offset: int | None = None,
        n_ww_lab_sites: int | None = None,
        lab_site_to_subpop_map: ArrayLike | None = None,
        population_size: int | None = None,
    ) -> None:
        self.n_ed_visits_data_days_ = n_ed_visits_data_days
        self.n_hospital_admissions_data_days_ = n_hospital_admissions_data_days
        self.n_wastewater_data_days_ = n_wastewater_data_days
        self.nssp_step_size = nssp_step_size
        self.nhsn_step_size = nhsn_step_size
        self.nwss_step_size = nwss_step_size
        self.nssp_training_data = nssp_training_data
        self.nhsn_training_data = nhsn_training_data
        self.nwss_training_data = nwss_training_data
        self.n_ww_lab_sites_ = n_ww_lab_sites
        self.lab_site_to_subpop_map_ = lab_site_to_subpop_map
        self.right_truncation_offset = right_truncation_offset
        self.population_size = population_size

        validate_mmwr_dates([first_hospital_admissions_date])

        self.first_ed_visits_date_ = first_ed_visits_date
        self.first_hospital_admissions_date_ = first_hospital_admissions_date
        self.first_wastewater_date_ = first_wastewater_date

    @classmethod
    def from_json(
        cls,
        json_file_path: str | Path,
        fit_ed_visits: bool = False,
        fit_hospital_admissions: bool = False,
        fit_wastewater: bool = False,
    ) -> Self:
        """
        Create a PyrenewHEWData instance from a JSON file.

        Parameters
        ----------
        json_file_path : str | Path
            Path to the data file in json format.
        fit_ed_visits : bool, optional
            Whether to fit ED visits data. Defaults to False.
        fit_hospital_admissions : bool, optional
            Whether to fit hospital admissions data. Defaults to False.
        fit_wastewater : bool, optional
            Whether to fit wastewater data. Defaults to False.

        Returns
        -------
        PyrenewHEWData
        """
        with open(
            json_file_path,
        ) as file:
            model_data = json.load(file)
        nssp_training_data = (
            pl.DataFrame(
                model_data["nssp_training_data"],
                schema={
                    "date": pl.Date,
                    "geo_value": pl.String,
                    "observed_ed_visits": pl.Float64,
                    "other_ed_visits": pl.Float64,
                    "data_type": pl.String,
                },
            )
            if fit_ed_visits
            else None
        )
        nhsn_training_data = (
            pl.DataFrame(
                model_data["nhsn_training_data"],
                schema={
                    "weekendingdate": pl.Date,
                    "jurisdiction": pl.String,
                    "hospital_admissions": pl.Float64,
                    "data_type": pl.String,
                },
            )
            if fit_hospital_admissions
            else None
        )
        nwss_training_data = (
            pl.DataFrame(
                model_data["nwss_training_data"],
                schema_overrides={
                    "date": pl.Date,
                    "lab_index": pl.Int64,
                    "site_index": pl.Int64,
                },
            )
            if fit_wastewater
            else None
        )

        return cls(
            nssp_training_data=nssp_training_data,
            nhsn_training_data=nhsn_training_data,
            nwss_training_data=nwss_training_data,
            population_size=jnp.array(model_data["loc_pop"]).item(),
            right_truncation_offset=model_data["right_truncation_offset"],
            nhsn_step_size=model_data["nhsn_step_size"]
            if fit_hospital_admissions
            else None,
            nssp_step_size=model_data["nssp_step_size"] if fit_ed_visits else None,
            nwss_step_size=model_data["nwss_step_size"] if fit_wastewater else None,
        )

    @property
    def n_ed_visits_data_days(self):
        return get_n_data_days(
            n_points=self.n_ed_visits_data_days_,
            date_array=self.dates_observed_ed_visits,
        )

    @property
    def n_hospital_admissions_data_days(self):
        return get_n_data_days(
            n_points=self.n_hospital_admissions_data_days_,
            date_array=self.dates_observed_hospital_admissions,
            timestep_days=7,
        )

    @property
    def n_wastewater_data_days(self):
        return get_n_data_days(
            n_points=self.n_wastewater_data_days_,
            date_array=self.dates_observed_disease_wastewater,
        )

    @property
    def dates_observed_ed_visits(self):
        if self.nssp_training_data is not None:
            return self.nssp_training_data.get_column("date").unique().to_numpy()

    @property
    def dates_observed_hospital_admissions(self):
        if self.nhsn_training_data is not None:
            return (
                self.nhsn_training_data.get_column("weekendingdate").unique().to_numpy()
            )

    @property
    def dates_observed_disease_wastewater(self):
        if self.nwss_training_data is not None:
            return self.nwss_training_data.get_column("date").unique().to_numpy()

    @property
    def first_wastewater_date(self):
        if self.dates_observed_disease_wastewater is not None:
            return self.dates_observed_disease_wastewater.min()
        return self.first_wastewater_date_

    @property
    def first_ed_visits_date(self):
        if self.dates_observed_ed_visits is not None:
            return self.dates_observed_ed_visits.min()
        return self.first_ed_visits_date_

    @property
    def first_hospital_admissions_date(self):
        if self.data_observed_disease_hospital_admissions is not None:
            return self.dates_observed_hospital_admissions.min()
        return self.first_hospital_admissions_date_

    @property
    def last_wastewater_date(self):
        return get_end_date(
            self.first_wastewater_date,
            self.n_wastewater_data_days,
            timestep_days=1,
        )

    @property
    def last_ed_visits_date(self):
        return get_end_date(
            self.first_ed_visits_date,
            self.n_ed_visits_data_days,
            timestep_days=1,
        )

    @property
    def last_hospital_admissions_date(self):
        return get_end_date(
            self.first_hospital_admissions_date,
            self.n_hospital_admissions_data_days,
            timestep_days=7,
        )

    @property
    def first_data_dates(self):
        return dict(
            ed_visits=self.first_ed_visits_date,
            hospital_admissions=self.first_hospital_admissions_date,
            wastewater=self.first_wastewater_date,
        )

    @property
    def last_data_dates(self):
        return dict(
            ed_visits=self.last_ed_visits_date,
            hospital_admissions=self.last_hospital_admissions_date,
            wastewater=self.last_wastewater_date,
        )

    @property
    def first_data_date_overall(self):
        return min(filter(None, self.first_data_dates.values()))

    @property
    def last_data_date_overall(self):
        return max(filter(None, self.last_data_dates.values()))

    @property
    def n_days_post_init(self):
        return (
            (self.last_data_date_overall - self.first_data_date_overall)
            // np.timedelta64(1, "D")
            + 1
        ).item()

    @property
    def data_observed_disease_ed_visits(self):
        if self.nssp_training_data is not None:
            return self.nssp_training_data.get_column("observed_ed_visits").to_numpy()

    @property
    def data_observed_total_ed_visits(self):
        if self.nssp_training_data is not None:
            return self.nssp_training_data.get_column("other_ed_visits").to_numpy()

    @property
    def data_observed_disease_hospital_admissions(self):
        if self.nhsn_training_data is not None:
            return self.nhsn_training_data.get_column("hospital_admissions").to_numpy()

    @property
    def site_subpop_spine(self):
        if self.nwss_training_data is not None:
            site_indices = (
                self.nwss_training_data.select(["site_index", "site", "site_pop"])
                .unique()
                .sort("site_index", descending=False)
            )

            total_pop_ww = (
                self.nwss_training_data.unique(["site_pop", "site"])
                .get_column("site_pop")
                .sum()
            )

            total_pop_no_ww = self.population_size - total_pop_ww
            add_auxiliary_subpop = total_pop_no_ww > 0

            if add_auxiliary_subpop:
                aux_subpop = pl.DataFrame(
                    {
                        "site_index": [None],
                        "site": [None],
                        "site_pop": [total_pop_no_ww],
                    }
                )
            else:
                aux_subpop = pl.DataFrame(schema=site_indices.schema)
            site_subpop_spine = (
                pl.concat([aux_subpop, site_indices], how="vertical_relaxed")
                .with_columns(
                    subpop_index=pl.col("site_index").cum_count().alias("subpop_index"),
                    subpop_name=pl.format("Site: {}", pl.col("site")).fill_null(
                        "remainder of population"
                    ),
                )
                .rename({"site_pop": "subpop_pop"})
            )
            return site_subpop_spine

    @property
    def date_time_spine(self):
        return create_date_time_spine(
            self.first_data_date_overall, self.last_data_date_overall
        )

    @property
    def wastewater_data_extended(self):
        if self.nwss_training_data is not None:
            return (
                self.nwss_training_data.join(
                    self.date_time_spine, on="date", how="left", coalesce=True
                )
                .join(
                    self.site_subpop_spine,
                    on=["site_index", "site"],
                    how="left",
                    coalesce=True,
                )
                .with_row_index("wastewater_observation_index")
            )

    @property
    def data_observed_disease_wastewater_conc(self):
        if self.nwss_training_data is not None:
            return self.wastewater_data_extended.get_column(
                "log_genome_copies_per_ml"
            ).to_numpy()

    @property
    def ww_censored(self):
        if self.nwss_training_data is not None:
            return (
                self.wastewater_data_extended.filter(pl.col("below_lod") == 1)
                .get_column("wastewater_observation_index")
                .to_numpy()
            )

    @property
    def ww_uncensored(self):
        if self.nwss_training_data is not None:
            return (
                self.wastewater_data_extended.filter(pl.col("below_lod") == 0)
                .get_column("wastewater_observation_index")
                .to_numpy()
            )

    @property
    def model_t_obs_wastewater(self):
        if self.nwss_training_data is not None:
            return self.wastewater_data_extended.get_column("t").to_numpy()

    @property
    def model_t_obs_ed_visits(self):
        if self.nssp_training_data is not None:
            return (
                self.nssp_training_data.join(
                    self.date_time_spine, on="date", how="left"
                )
                .get_column("t")
                .unique()
                .to_numpy()
            )
        return None

    @property
    def model_t_obs_hospital_admissions(self):
        if self.nhsn_training_data is not None:
            return (
                self.nhsn_training_data.join(
                    self.date_time_spine,
                    left_on="weekendingdate",
                    right_on="date",
                    how="left",
                )
                .get_column("t")
                .to_numpy()
            )
        return None

    @property
    def ww_observed_subpops(self):
        if self.nwss_training_data is not None:
            return self.wastewater_data_extended.get_column("subpop_index").to_numpy()

    @property
    def ww_observed_lab_sites(self):
        if self.nwss_training_data is not None:
            return self.wastewater_data_extended.get_column("lab_site_index").to_numpy()

    @property
    def ww_log_lod(self):
        if self.nwss_training_data is not None:
            return self.wastewater_data_extended.get_column("log_lod").to_numpy()

    @property
    def n_ww_lab_sites(self):
        if self.nwss_training_data is not None:
            return self.wastewater_data_extended["lab_site_index"].n_unique()
        return self.n_ww_lab_sites_

    @property
    def lab_site_to_subpop_map(self):
        if self.nwss_training_data is not None:
            return (
                (
                    self.wastewater_data_extended["lab_site_index", "subpop_index"]
                    .unique()
                    .sort(by="lab_site_index", descending=False)
                )
                .get_column("subpop_index")
                .to_numpy()
            )
        return self.lab_site_to_subpop_map_

    def to_forecast_data(self, n_forecast_points: int) -> Self:
        """
        Create a new PyrenewHEWData instance for forecasting.

        This method extends the current data object to include forecast points,
        converting from observed data to a structure suitable for forecasting.

        Parameters
        ----------
        n_forecast_points : int
            Number of additional days to forecast beyond the current data.

        Returns
        -------
        PyrenewHEWData
            A new instance configured for forecasting with extended time range.

        Notes
        -----
        The method handles different temporal resolutions for data streams:

        - ED visits and wastewater data are daily, so they extend by
          n_forecast_points days.
        - Hospital admissions are weekly (MMWR epiweeks), so the number of
          weeks is calculated as total days divided by 7 (integer division).
        """
        # Calculate total forecast period
        n_days = self.n_days_post_init + n_forecast_points
        n_weeks = n_days // 7

        # Find the first Saturday on or after first_data_date_overall
        first_dow = self.first_data_date_overall.astype(dt.datetime).weekday()
        to_first_sat = (5 - first_dow) % 7  # Saturday is weekday 5
        first_mmwr_ending_date = self.first_data_date_overall + np.timedelta64(
            to_first_sat, "D"
        )

        return PyrenewHEWData(
            n_ed_visits_data_days=n_days,
            n_hospital_admissions_data_days=n_weeks,
            n_wastewater_data_days=n_days,
            first_ed_visits_date=self.first_data_date_overall,
            first_hospital_admissions_date=first_mmwr_ending_date,
            first_wastewater_date=self.first_data_date_overall,
            right_truncation_offset=None,  # by default, want forecasts of complete reports
            n_ww_lab_sites=self.n_ww_lab_sites,
            lab_site_to_subpop_map=self.lab_site_to_subpop_map,
        )

import datetime
from typing import Self

import jax.numpy as jnp
import polars as pl
from jax.typing import ArrayLike


class PyrenewHEWData:
    """
    Class for holding input data
    to a PyrenewHEW model.
    """

    def __init__(
        self,
        nssp_training_data: pl.DataFrame = None,
        nhsn_training_data: pl.DataFrame = None,
        nwss_training_data: pl.DataFrame = None,
        n_ed_visits_data_days: int = None,
        n_hospital_admissions_data_days: int = None,
        n_wastewater_data_days: int = None,
        first_ed_visits_date: datetime.date = None,
        first_hospital_admissions_date: datetime.date = None,
        first_wastewater_date: datetime.date = None,
        right_truncation_offset: int = None,
        site_subpop_spine: pl.DataFrame = None,
        pop_fraction: ArrayLike = None,
        n_ww_lab_sites: int = None,
        ww_censored: ArrayLike = None,
        ww_uncensored: ArrayLike = None,
        ww_observed_subpops: ArrayLike = None,
        ww_observed_times: ArrayLike = None,
        ww_observed_lab_sites: ArrayLike = None,
        lab_site_to_subpop_map: ArrayLike = None,
        ww_log_lod: ArrayLike = None,
    ) -> None:
        self.n_ed_visits_data_days_ = n_ed_visits_data_days
        self.n_hospital_admissions_data_days_ = n_hospital_admissions_data_days
        self.n_wastewater_data_days_ = n_wastewater_data_days
        self.nssp_training_data = nssp_training_data
        self.nhsn_training_data = nhsn_training_data
        self.nwss_training_data = nwss_training_data
        self.first_ed_visits_date_ = first_ed_visits_date
        self.first_hospital_admissions_date_ = first_hospital_admissions_date
        self.first_wastewater_date_ = first_wastewater_date
        self.n_ww_lab_sites_ = n_ww_lab_sites
        self.ww_uncensored_ = ww_uncensored
        self.ww_censored_ = ww_censored
        self.ww_observed_lab_sites_ = ww_observed_lab_sites
        self.ww_observed_subpops_ = ww_observed_subpops
        self.ww_observed_times_ = ww_observed_times
        self.lab_site_to_subpop_map_ = lab_site_to_subpop_map
        self.ww_log_lod_ = ww_log_lod
        self.right_truncation_offset = right_truncation_offset
        self.pop_fraction = pop_fraction
        self.site_subpop_spine = site_subpop_spine

    @property
    def n_ed_visits_data_days(self):
        return self.get_n_data_days(
            n_datapoints=self.n_ed_visits_data_days_,
            date_array=self.dates_observed_ed_visits,
        )

    @property
    def n_hospital_admissions_data_days(self):
        return self.get_n_data_days(
            n_datapoints=self.n_hospital_admissions_data_days_,
            date_array=self.dates_observed_hospital_admissions,
            timestep_days=7,
        )

    @property
    def n_wastewater_data_days(self):
        return self.get_n_data_days(
            n_datapoints=self.n_wastewater_data_days_,
            date_array=self.dates_observed_disease_wastewater,
        )

    @property
    def dates_observed_ed_visits(self):
        if self.nssp_training_data is not None:
            return self.nssp_training_data.get_column("date").unique()

    @property
    def dates_observed_hospital_admissions(self):
        if self.nhsn_training_data is not None:
            return self.nhsn_training_data.get_column(
                "weekendingdate"
            ).unique()

    @property
    def dates_observed_disease_wastewater(self):
        if self.nwss_training_data is not None:
            return self.nwss_training_data.get_column("date").unique()

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
        return self.get_end_date(
            self.first_wastewater_date,
            self.n_wastewater_data_days,
            timestep_days=1,
        )

    @property
    def last_ed_visits_date(self):
        return self.get_end_date(
            self.first_ed_visits_date,
            self.n_ed_visits_data_days,
            timestep_days=1,
        )

    @property
    def last_hospital_admissions_date(self):
        return self.get_end_date(
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
            self.last_data_date_overall - self.first_data_date_overall
        ).days

    @property
    def data_observed_disease_ed_visits(self):
        if self.nssp_training_data is not None:
            return (
                self.nssp_training_data.filter(pl.col("disease") != "Total")
                .get_column("ed_visits")
                .to_numpy()
            )

    @property
    def data_observed_total_ed_visits(self):
        if self.nssp_training_data is not None:
            return (
                self.nssp_training_data.filter(pl.col("disease") == "Total")
                .get_column("ed_visits")
                .to_numpy()
            )

    @property
    def data_observed_disease_hospital_admissions(self):
        if self.nhsn_training_data is not None:
            return self.nhsn_training_data.get_column(
                "hospital_admissions"
            ).to_numpy()

    @property
    def date_time_spine(self):
        if self.nwss_training_data is not None:
            date_time_spine = pl.DataFrame(
                {
                    "date": pl.date_range(
                        start=self.first_data_date_overall,
                        end=self.last_data_date_overall,
                        interval="1d",
                        eager=True,
                    )
                }
            ).with_row_index("t")
            return date_time_spine

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
                .with_row_index("ind_rel_to_observed_times")
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
                .get_column("ind_rel_to_observed_times")
                .to_numpy()
            )
        return self.ww_censored_

    @property
    def ww_uncensored(self):
        if self.nwss_training_data is not None:
            return (
                self.wastewater_data_extended.filter(pl.col("below_lod") == 0)
                .get_column("ind_rel_to_observed_times")
                .to_numpy()
            )
        return self.ww_uncensored_

    @property
    def ww_observed_times(self):
        if self.nwss_training_data is not None:
            return self.wastewater_data_extended.get_column("t").to_numpy()
        return self.ww_observed_times_

    @property
    def ww_observed_subpops(self):
        if self.nwss_training_data is not None:
            return self.wastewater_data_extended.get_column(
                "subpop_index"
            ).to_numpy()
        return self.ww_observed_subpops_

    @property
    def ww_observed_lab_sites(self):
        if self.nwss_training_data is not None:
            return self.wastewater_data_extended.get_column(
                "lab_site_index"
            ).to_numpy()
        return self.ww_observed_lab_sites_

    @property
    def ww_log_lod(self):
        if self.nwss_training_data is not None:
            return self.wastewater_data_extended.get_column(
                "log_lod"
            ).to_numpy()
        return self.ww_log_lod_

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
                    self.wastewater_data_extended[
                        "lab_site_index", "subpop_index"
                    ]
                    .unique()
                    .sort(by="lab_site_index", descending=False)
                )
                .get_column("subpop_index")
                .to_numpy()
            )
        return self.lab_site_to_subpop_map_

    def get_end_date(
        self,
        first_date: datetime.date,
        n_datapoints: int,
        timestep_days: int = 1,
    ) -> datetime.date:
        """
        Get end date from a first date and a number of datapoints,
        with handling of None values and non-daily timeseries
        """
        if first_date is None:
            if n_datapoints != 0:
                raise ValueError(
                    "Must provide an initial date if "
                    "n_datapoints is non-zero. "
                    f"Got n_datapoints = {n_datapoints} "
                    "but first_date was `None`"
                )
            result = None
        else:
            result = first_date + datetime.timedelta(
                days=n_datapoints * timestep_days
            )
        return result

    def get_n_data_days(
        self,
        n_datapoints: int = None,
        date_array: ArrayLike = None,
        timestep_days: int = 1,
    ) -> int:
        if n_datapoints is None and date_array is None:
            return 0
        elif date_array is not None and n_datapoints is not None:
            raise ValueError(
                "Must provide at most one out of a "
                "number of datapoints to simulate and "
                "an array of dates data is observed."
            )
        elif date_array is not None:
            return (
                max(date_array) - min(date_array)
            ).days // timestep_days + 1
        else:
            return n_datapoints

    def date_to_model_time(self, first_data_date_overall, dates_observed):
        offset = (first_data_date_overall - dates_observed).days

    def to_forecast_data(self, n_forecast_points: int) -> Self:
        n_days = self.n_days_post_init + n_forecast_points
        n_weeks = n_days // 7
        return PyrenewHEWData(
            n_ed_visits_data_days=n_days,
            n_hospital_admissions_data_days=n_weeks,
            n_wastewater_data_days=n_days,
            first_ed_visits_date=self.first_data_date_overall,
            first_hospital_admissions_date=(self.first_data_date_overall),
            first_wastewater_date=self.first_data_date_overall,
            right_truncation_offset=None,  # by default, want forecasts of complete reports
            n_ww_lab_sites=self.n_ww_lab_sites,
            ww_uncensored=self.ww_uncensored,
            ww_censored=self.ww_censored,
            ww_observed_lab_sites=self.ww_observed_lab_sites,
            ww_observed_subpops=self.ww_observed_subpops,
            ww_observed_times=self.ww_observed_times,
            lab_site_to_subpop_map=self.lab_site_to_subpop_map,
            ww_log_lod=self.ww_log_lod,
            pop_fraction=self.pop_fraction,
        )

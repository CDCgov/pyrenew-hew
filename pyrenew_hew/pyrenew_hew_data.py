import datetime
from typing import Self

import jax.numpy as jnp
from jax.typing import ArrayLike

from pyrenew_hew.pyrenew_wastewater_data import PyrenewWastewaterData


class PyrenewHEWData:
    """
    Class for holding input data
    to a PyrenewHEW model.
    """

    def __init__(
        self,
        n_ed_visits_data_days: int = None,
        n_hospital_admissions_data_days: int = None,
        n_wastewater_data_days: int = None,
        data_observed_disease_ed_visits: ArrayLike = None,
        data_observed_disease_hospital_admissions: ArrayLike = None,
        right_truncation_offset: int = None,
        first_ed_visits_date: datetime.date = None,
        first_hospital_admissions_date: datetime.date = None,
        first_wastewater_date: datetime.date = None,
        n_ww_lab_sites: int = None,
        ww_censored: ArrayLike = None,
        ww_uncensored: ArrayLike = None,
        ww_observed_subpops: ArrayLike = None,
        ww_observed_times: ArrayLike = None,
        ww_observed_lab_sites: ArrayLike = None,
        lab_site_to_subpop_map: ArrayLike = None,
        ww_log_lod: ArrayLike = None,
        date_observed_disease_wastewater: ArrayLike = None,
        data_observed_disease_wastewater_conc: ArrayLike = None,
        pop_fraction: ArrayLike = jnp.array([1]),
    ) -> None:
        self.n_ed_visits_data_days_ = n_ed_visits_data_days
        self.n_hospital_admissions_data_days_ = n_hospital_admissions_data_days
        self.n_wastewater_data_days_ = n_wastewater_data_days
        self.data_observed_disease_ed_visits = data_observed_disease_ed_visits
        self.data_observed_disease_hospital_admissions = (
            data_observed_disease_hospital_admissions
        )
        self.right_truncation_offset = right_truncation_offset
        self.first_ed_visits_date = first_ed_visits_date
        self.first_hospital_admissions_date = first_hospital_admissions_date
        self.first_wastewater_date_ = first_wastewater_date
        self.date_observed_disease_wastewater = (
            date_observed_disease_wastewater
        )
        self.pop_fraction = pop_fraction
        self.data_observed_disease_wastewater_conc = (
            data_observed_disease_wastewater_conc
        )
        self.ww_censored = ww_censored
        self.ww_uncensored = ww_uncensored
        self.ww_observed_times = ww_observed_times
        self.ww_observed_subpops = ww_observed_subpops
        self.ww_observed_lab_sites = ww_observed_lab_sites
        self.ww_log_lod = ww_log_lod
        self.n_ww_lab_sites = n_ww_lab_sites
        self.lab_site_to_subpop_map = lab_site_to_subpop_map

    @property
    def n_ed_visits_data_days(self):
        return self.get_n_data_days(
            n_datapoints=self.n_ed_visits_data_days_,
            data_array=self.data_observed_disease_ed_visits,
        )

    @property
    def n_hospital_admissions_data_days(self):
        return self.get_n_data_days(
            n_datapoints=self.n_hospital_admissions_data_days_,
            data_array=self.data_observed_disease_hospital_admissions,
        )

    @property
    def n_wastewater_data_days(self):
        return self.get_n_wastewater_data_days(
            n_datapoints=self.n_wastewater_data_days_,
            date_array=self.date_observed_disease_wastewater,
        )

    @property
    def first_wastewater_date(self):
        if self.date_observed_disease_wastewater is not None:
            return self.date_observed_disease_wastewater.min()
        return self.first_wastewater_date_

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
        self, n_datapoints: int = None, data_array: ArrayLike = None
    ) -> int:
        if n_datapoints is None and data_array is None:
            return 0
        elif data_array is not None and n_datapoints is not None:
            raise ValueError(
                "Must provide at most one out of a "
                "number of datapoints to simulate and "
                "an array of observed data."
            )
        elif data_array is not None:
            return data_array.shape[0]
        else:
            return n_datapoints

    def get_n_wastewater_data_days(
        self, n_datapoints: int = None, date_array: ArrayLike = None
    ) -> int:
        if n_datapoints is None and date_array is None:
            return 0
        elif date_array is not None and n_datapoints is not None:
            raise ValueError(
                "Must provide at most one out of a "
                "number of datapoints to simulate and "
                "an array of dates wastewater data is "
                "observed."
            )
        elif date_array is not None:
            return (max(date_array) - min(date_array)).days
        else:
            return n_datapoints

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
            data_observed_disease_wastewater_conc=None,
        )

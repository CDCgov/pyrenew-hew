import datetime
from typing import Self

from jax.typing import ArrayLike


class PyrenewHEWData:
    """
    Class for holding input data
    to a PyrenewHEW model.
    """

    def __init__(
        self,
        n_ed_visits_datapoints: int = None,
        n_hospital_admissions_datapoints: int = None,
        n_wastewater_datapoints: int = None,
        data_observed_disease_ed_visits: ArrayLike = None,
        data_observed_disease_hospital_admissions: ArrayLike = None,
        data_observed_disease_wastewater: ArrayLike = None,
        right_truncation_offset: int = None,
        first_ed_visits_date: datetime.datetime.date = None,
        first_hospital_admissions_date: datetime.datetime.date = None,
        first_wastewater_date: datetime.datetime.date = None,
    ) -> None:
        self.n_ed_visits_datapoints_ = n_ed_visits_datapoints
        self.n_hospital_admissions_datapoints_ = (
            n_hospital_admissions_datapoints
        )
        self.n_wastewater_datapoints_ = n_wastewater_datapoints

        self.data_observed_disease_ed_visits = data_observed_disease_ed_visits
        self.data_observed_disease_hospital_admissions = (
            data_observed_disease_hospital_admissions
        )
        self.data_observed_disease_wastewater = (
            data_observed_disease_wastewater
        )

        self.right_truncation_offset = right_truncation_offset

        self.first_ed_visits_date = first_ed_visits_date
        self.first_hospital_admissions_date = first_hospital_admissions_date
        self.first_wastewater_date = first_wastewater_date

    @property
    def n_ed_visits_datapoints(self):
        return self.get_n_datapoints(
            n_datapoints=self.n_ed_visits_datapoints_,
            data_array=self.data_observed_disease_ed_visits,
        )

    @property
    def n_hospital_admissions_datapoints(self):
        return self.get_n_datapoints(
            n_datapoints=self.n_hospital_admissions_datapoints_,
            data_array=self.data_observed_disease_hospital_admissions,
        )

    @property
    def n_wastewater_datapoints(self):
        return self.get_n_datapoints(
            n_datapoints=self.n_wastewater_datapoints_,
            data_array=self.data_observed_disease_wastewater,
        )

    @property
    def last_ed_visits_date(self):
        return self.get_end_date(
            self.first_ed_visits_date,
            self.n_ed_visits_datapoints,
            timestep_days=1,
        )

    @property
    def last_hospital_admissions_date(self):
        return self.get_end_date(
            self.first_hospital_admissions_date,
            self.n_hospital_admissions_datapoints,
            timestep_days=7,
        )

    @property
    def last_wastewater_date(self):
        return self.get_end_date(
            self.first_wastewater_date,
            self.n_wastewater_datapoints,
            timestep_days=1,
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
        return min(filter(None, self.last_data_dates.values()))

    @property
    def n_days_post_init(self):
        return (
            self.last_data_date_overall - self.first_data_date_overall
        ).days

    def get_end_date(
        self,
        first_date: datetime.datetime.date,
        n_datapoints: int,
        timestep_days: int = 1,
    ) -> datetime.datetime.date:
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

    def get_n_datapoints(
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

    def to_forecast_data(self, n_forecast_points: int) -> Self:
        n_days = self.n_days_post_init + n_forecast_points
        n_weeks = n_days // 7
        return PyrenewHEWData(
            n_ed_visits_datapoints=n_days,
            n_hospital_admissions_datapoints=n_weeks,
            n_wastewater_datapoints=n_days,
            first_ed_visits_date=self.first_data_date_overall,
            first_hospital_admissions_date=(self.first_data_date_overall),
            first_wastewater_date=self.first_data_date_overall,
            right_truncation_offset=0,
        )

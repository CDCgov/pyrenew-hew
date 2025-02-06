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
        n_ed_visits_datapoints: int = None,
        n_hospital_admissions_datapoints: int = None,
        n_wastewater_datapoints: int = None,
        data_observed_disease_ed_visits: ArrayLike = None,
        data_observed_disease_hospital_admissions: ArrayLike = None,
        right_truncation_offset: int = None,
        first_ed_visits_date: datetime.date = None,
        first_hospital_admissions_date: datetime.date = None,
        first_wastewater_date: datetime.date = None,
        wastewater_data: pl.DataFrame = None,
        population_size: int = None,
        shedding_offset: float = 1e-8,
        pop_fraction: ArrayLike = jnp.array([1]),
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
        self.right_truncation_offset = right_truncation_offset
        self.first_ed_visits_date = first_ed_visits_date
        self.first_hospital_admissions_date = first_hospital_admissions_date
        self.first_wastewater_date_ = first_wastewater_date
        self.wastewater_data = wastewater_data
        self.population_size = population_size
        self.shedding_offset = shedding_offset
        self.pop_fraction_ = pop_fraction

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
        return self.get_n_wastewater_datapoints(
            n_datapoints=self.n_wastewater_datapoints_,
            date_array=(
                None
                if self.wastewater_data is None
                else self.wastewater_data["t"].to_list()
            ),
        )

    @property
    def first_wastewater_date(self):
        if self.wastewater_data is not None:
            return self.wastewater_data["date"].min()
        return self.first_wastewater_date_

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
        return max(filter(None, self.last_data_dates.values()))

    @property
    def n_days_post_init(self):
        return (
            self.last_data_date_overall - self.first_data_date_overall
        ).days

    @property
    def pop_fraction(self):
        if self.wastewater_data is not None:
            pop_covered_by_ww_surv = (
                self.wastewater_data["subpop_index", "subpop_pop"]
                .unique()
                .sort(by="subpop_index", descending=False)["subpop_pop"]
                .to_numpy()
            )
            subpop_sizes = jnp.concat(
                [
                    jnp.array(
                        [
                            self.population_size
                            - jnp.sum(pop_covered_by_ww_surv)
                        ]
                    ),
                    pop_covered_by_ww_surv,
                ]
            )
            return subpop_sizes / self.population_size
        return self.pop_fraction_

    @property
    def data_observed_disease_wastewater(self):
        if self.wastewater_data is not None:
            return self.wastewater_data["log_genome_copies_per_ml"].to_numpy()

    @property
    def ww_censored(self):
        if self.wastewater_data is not None:
            return self.wastewater_data.filter(pl.col("below_lod") == 1)[
                "ind_rel_to_observed_times"
            ].to_numpy()
        return None

    @property
    def ww_uncensored(self):
        if self.wastewater_data is not None:
            return self.wastewater_data.filter(pl.col("below_lod") == 0)[
                "ind_rel_to_observed_times"
            ].to_numpy()

    @property
    def ww_observed_times(self):
        if self.wastewater_data is not None:
            return self.wastewater_data["t"].to_numpy()

    @property
    def ww_observed_subpops(self):
        if self.wastewater_data is not None:
            return self.wastewater_data["subpop_index"].to_numpy()

    @property
    def ww_observed_lab_sites(self):
        if self.wastewater_data is not None:
            return self.wastewater_data["lab_site_index"].to_numpy()

    @property
    def ww_log_lod(self):
        if self.wastewater_data is not None:
            return self.wastewater_data["log_lod"].to_numpy()

    @property
    def n_ww_lab_sites(self):
        if self.wastewater_data is not None:
            return self.wastewater_data["lab_site_index"].n_unique()

    @property
    def lab_site_to_subpop_map(self):
        if self.wastewater_data is not None:
            return (
                self.wastewater_data["lab_site_index", "subpop_index"]
                .unique()
                .sort(by="lab_site_index")
            )["subpop_index"].to_numpy()

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

    # need a different function here since wastewater data
    # array differs from ed visits nad hosp admissions data arrays,
    # multiple entries possible for a single date
    def get_n_wastewater_datapoints(
        self, n_datapoints: int = None, date_array: ArrayLike = None
    ) -> int:
        if n_datapoints is None and date_array is None:
            return 0
        elif date_array is not None and n_datapoints is not None:
            raise ValueError(
                "Must provide at most one out of a "
                "number of datapoints to simulate and "
                "an array of observed wastewater "
                "concentration data."
            )
        elif date_array is not None:
            return max(date_array)
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
            pop_fraction=self.pop_fraction,
            wastewater_data=self.wastewater_data,
            right_truncation_offset=None,  # by default, want forecasts of complete reports
        )

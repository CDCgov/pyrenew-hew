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
        n_ed_visits_data_days: int = None,
        n_hospital_admissions_data_days: int = None,
        n_wastewater_data_days: int = None,
        data_observed_disease_ed_visits: ArrayLike = None,
        data_observed_disease_hospital_admissions: ArrayLike = None,
        data_observed_disease_wastewater: pl.DataFrame = None,
        right_truncation_offset: int = None,
        first_ed_visits_date: datetime.date = None,
        first_hospital_admissions_date: datetime.date = None,
        first_wastewater_date: datetime.date = None,
        population_size: int = None,
        shedding_offset: float = 1e-8,
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
        self.data_observed_disease_wastewater = (
            data_observed_disease_wastewater
        )
        self.population_size = population_size
        self.shedding_offset = shedding_offset
        self.pop_fraction_ = pop_fraction

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
            date_array=(
                None
                if self.data_observed_disease_wastewater is None
                else self.data_observed_disease_wastewater["date"]
            ),
        )

    @property
    def first_wastewater_date(self):
        if self.data_observed_disease_wastewater is not None:
            return self.data_observed_disease_wastewater["date"].min()
        return self.first_wastewater_date_

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
    def last_wastewater_date(self):
        return self.get_end_date(
            self.first_wastewater_date,
            self.n_wastewater_data_days,
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
    def site_subpop_spine(self):
        ww_data_present = self.data_observed_disease_wastewater is not None
        if ww_data_present:
            # Check if auxiliary subpopulation needs to be added
            add_auxiliary_subpop = (
                self.population_size
                > self.data_observed_disease_wastewater.select(
                    pl.col("site_pop", "site", "lab", "lab_site_index")
                )
                .unique()
                .sum()
                .get_column("site_pop")
                .item()
            )
            site_indices = (
                self.data_observed_disease_wastewater.select(
                    ["site_index", "site", "site_pop"]
                )
                .unique()
                .sort("site_index")
            )
            if add_auxiliary_subpop:
                aux_subpop = pl.DataFrame(
                    {
                        "site_index": [None],
                        "site": [None],
                        "site_pop": [
                            self.population_size
                            - site_indices.select(pl.col("site_pop"))
                            .sum()
                            .get_column("site_pop")
                            .item()
                        ],
                    }
                )
            else:
                aux_subpop = pl.DataFrame()
            site_subpop_spine = (
                pl.concat([aux_subpop, site_indices], how="vertical_relaxed")
                .with_columns(
                    subpop_index=pl.col("site_index")
                    .cum_count()
                    .alias("subpop_index"),
                    subpop_name=pl.format(
                        "Site: {}", pl.col("site")
                    ).fill_null("remainder of population"),
                )
                .rename({"site_pop": "subpop_pop"})
            )
        else:
            site_subpop_spine = pl.DataFrame(
                {
                    "site_index": [None],
                    "site": [None],
                    "subpop_pop": [self.population_size],
                    "subpop_index": [1],
                    "subpop_name": ["total population"],
                }
            )
        return site_subpop_spine

    @property
    def date_time_spine(self):
        if self.data_observed_disease_wastewater is not None:
            date_time_spine = pl.DataFrame(
                {
                    "date": pl.date_range(
                        start=self.first_wastewater_date,
                        end=self.last_wastewater_date,
                        interval="1d",
                        eager=True,
                    )
                }
            )
            return date_time_spine.with_row_index("t")

    @property
    def wastewater_data_extended(self):
        if self.data_observed_disease_wastewater is not None:
            return (
                self.data_observed_disease_wastewater.join(
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
    def pop_fraction(self):
        if self.data_observed_disease_wastewater is not None:
            subpop_sizes = self.site_subpop_spine["subpop_pop"].to_numpy()
            return subpop_sizes / self.population_size
        return self.pop_fraction_

    @property
    def data_observed_disease_wastewater_conc(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended[
                "log_genome_copies_per_ml"
            ].to_numpy()

    @property
    def ww_censored(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended.filter(
                pl.col("below_lod") == 1
            )["ind_rel_to_observed_times"].to_numpy()
        return None

    @property
    def ww_uncensored(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended.filter(
                pl.col("below_lod") == 0
            )["ind_rel_to_observed_times"].to_numpy()

    @property
    def ww_observed_times(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended["t"].to_numpy()

    @property
    def ww_observed_subpops(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended["subpop_index"].to_numpy()

    @property
    def ww_observed_lab_sites(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended["lab_site_index"].to_numpy()

    @property
    def ww_log_lod(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended["log_lod"].to_numpy()

    @property
    def n_ww_lab_sites(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended["lab_site_index"].n_unique()

    @property
    def lab_site_to_subpop_map(self):
        if self.data_observed_disease_wastewater is not None:
            return (
                self.wastewater_data_extended["lab_site_index", "subpop_index"]
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
                "an array of observed wastewater "
                "concentration data."
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
            pop_fraction=self.pop_fraction,
            right_truncation_offset=None,  # by default, want forecasts of complete reports
        )

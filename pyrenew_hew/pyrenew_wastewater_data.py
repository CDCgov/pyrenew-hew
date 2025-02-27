import datetime
from typing import Self

import jax
import jax.numpy as jnp
import polars as pl
from jax.typing import ArrayLike


class PyrenewWastewaterData:
    """
    Class for holding wastewater input data
    to a PyrenewHEW model.
    """

    def __init__(
        self,
        data_observed_disease_wastewater: pl.DataFrame = None,
        population_size: int = None,
    ) -> None:
        self.data_observed_disease_wastewater = (
            data_observed_disease_wastewater
        )
        self.population_size = population_size

    @property
    def site_subpop_spine(self):
        ww_data_present = self.data_observed_disease_wastewater is not None
        if ww_data_present:
            site_indices = (
                self.data_observed_disease_wastewater.select(
                    ["site_index", "site", "site_pop"]
                )
                .unique()
                .sort("site_index")
            )

            total_pop_ww = (
                self.data_observed_disease_wastewater.unique(
                    ["site_pop", "site"]
                )
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
                        start=self.date_observed_disease_wastewater.min(),
                        end=self.date_observed_disease_wastewater.max(),
                        interval="1d",
                        eager=True,
                    )
                }
            ).with_row_index("t")
            return date_time_spine

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
    def date_observed_disease_wastewater(self):
        if self.data_observed_disease_wastewater is not None:
            return self.data_observed_disease_wastewater.get_column(
                "date"
            ).unique()

    @property
    def data_observed_disease_wastewater_conc(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended.get_column(
                "log_genome_copies_per_ml"
            ).to_numpy()

    @property
    def ww_censored(self):
        if self.data_observed_disease_wastewater is not None:
            return (
                self.wastewater_data_extended.filter(pl.col("below_lod") == 1)
                .get_column("ind_rel_to_observed_times")
                .to_numpy()
            )
        return None

    @property
    def ww_uncensored(self):
        if self.data_observed_disease_wastewater is not None:
            return (
                self.wastewater_data_extended.filter(pl.col("below_lod") == 0)
                .get_column("ind_rel_to_observed_times")
                .to_numpy()
            )

    @property
    def ww_observed_times(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended.get_column("t").to_numpy()

    @property
    def ww_observed_subpops(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended.get_column(
                "subpop_index"
            ).to_numpy()

    @property
    def ww_observed_lab_sites(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended.get_column(
                "lab_site_index"
            ).to_numpy()

    @property
    def ww_log_lod(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended.get_column(
                "log_lod"
            ).to_numpy()

    @property
    def n_ww_lab_sites(self):
        if self.data_observed_disease_wastewater is not None:
            return self.wastewater_data_extended["lab_site_index"].n_unique()

    @property
    def lab_site_to_subpop_map(self):
        if self.data_observed_disease_wastewater is not None:
            return (
                (
                    self.wastewater_data_extended[
                        "lab_site_index", "subpop_index"
                    ]
                    .unique()
                    .sort(by="lab_site_index")
                )
                .get_column("subpop_index")
                .to_numpy()
            )

    def to_pyrenew_hew_data_args(self):
        return {
            attr: value
            for attr, value in (
                (attr, getattr(self, attr))
                for attr, prop in self.__class__.__dict__.items()
                if isinstance(prop, property)
            )
            if isinstance(value, ArrayLike)
        }

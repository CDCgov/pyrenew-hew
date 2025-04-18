import polars as pl


class PyrenewWastewaterData:
    """
    Class for holding wastewater input data
    to a PyrenewHEW model.
    """

    def __init__(
        self,
        nwss_training_data: pl.DataFrame = None,
        population_size: int = None,
    ) -> None:
        self.nwss_training_data = nwss_training_data
        self.population_size = population_size

    @property
    def site_subpop_spine(self):
        ww_data_present = self.nwss_training_data is not None
        if ww_data_present:
            site_indices = (
                self.nwss_training_data.select(
                    ["site_index", "site", "site_pop"]
                )
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

    def to_pyrenew_hew_data_args(self):
        return {attr: getattr(self, attr) for attr in ["site_subpop_spine"]}

import arviz as az
import matplotlib.pyplot as plt


def compute_eti(dataset, eti_prob):
    eti_bdry = dataset.quantile(
        ((1 - eti_prob) / 2, 1 / 2 + eti_prob / 2), dim=("chain", "draw")
    )
    return eti_bdry.values.T


def plot_posterior(idata, name):
    x_data = idata.posterior[f"{name}_dim_0"]
    y_data = idata.posterior[name]
    fig, axes = plt.subplots(figsize=(6, 5))
    az.plot_hdi(
        x_data,
        hdi_data=compute_eti(y_data, 0.9),
        color="C0",
        smooth=False,
        fill_kwargs={"alpha": 0.3},
        ax=axes,
    )

    az.plot_hdi(
        x_data,
        hdi_data=compute_eti(y_data, 0.5),
        color="C0",
        smooth=False,
        fill_kwargs={"alpha": 0.6},
        ax=axes,
    )

    # Add median of the posterior to the figure
    median_ts = y_data.median(dim=["chain", "draw"])

    plt.plot(
        x_data,
        median_ts,
        color="C0",
        label="Median",
    )

    axes.legend()
    axes.set_title(name, fontsize=10)
    axes.set_xlabel("Time", fontsize=10)
    axes.set_ylabel(name, fontsize=10)
    return fig


def plot_predictive(idata, prior=False):
    prior_or_post_text = "Prior" if prior else "Posterior"
    predictive_obj = (
        idata.prior_predictive if prior else idata.posterior_predictive
    )

    x_data = predictive_obj["observed_ed_visits_dim_0"]
    y_data = predictive_obj["observed_ed_visits"]

    fig, axes = plt.subplots(figsize=(6, 5))
    az.plot_hdi(
        x_data,
        hdi_data=compute_eti(y_data, 0.9),
        color="C0",
        smooth=False,
        fill_kwargs={"alpha": 0.3},
        ax=axes,
    )

    az.plot_hdi(
        x_data,
        hdi_data=compute_eti(y_data, 0.5),
        color="C0",
        smooth=False,
        fill_kwargs={"alpha": 0.6},
        ax=axes,
    )

    # Add median of the posterior to the figure
    median_ts = y_data.median(dim=["chain", "draw"])

    plt.plot(
        x_data,
        median_ts,
        color="C0",
        label="Median",
    )
    plt.scatter(
        idata.observed_data["observed_ed_visits_dim_0"],
        idata.observed_data["observed_ed_visits"],
        color="black",
    )
    axes.legend()
    axes.set_title(f"{prior_or_post_text} Predictive Admissions", fontsize=10)
    axes.set_xlabel("Time", fontsize=10)
    axes.set_ylabel("Hospital Admissions", fontsize=10)

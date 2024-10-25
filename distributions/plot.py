import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


def plot_binomial_distributions(probability_distibutions: dict) -> None:
    # fig = plt.figure(figsize=(20, 10))
    fig, ax = plt.subplots(figsize=(20, 10))

    for mu, probability_distibution in probability_distibutions.items():
        ax.bar(
            probability_distibution["x"],
            probability_distibution["y"],
            label=f"µ = {mu}",
            color=probability_distibution["color"],
            alpha=0.3,
        )
        ax.scatter(
            probability_distibution["x"],
            probability_distibution["y_computed"],
            marker="x",
            label=f"µ = {mu} computed",
            s=100,
        )

    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    plt.xlim([-1, 16])
    plt.ylim([0, 0.4])
    plt.xlabel("Number m of observations x = 1 in N = 15 experiments", fontsize=13)
    plt.ylabel("p(m)", fontsize=13)
    plt.title("Binomial Distributions")
    plt.legend(loc=1)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig("plots/binomial.png")


def plot_poisson(poisson_distibutions: dict) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))

    for _lambda, poisson_distibution in poisson_distibutions.items():
        ax.bar(
            poisson_distibution["x"],
            poisson_distibution["y"],
            label=f"λ = {_lambda}",
            color=poisson_distibution["color"],
            alpha=0.3,
        )

    ax.grid(axis="x", linestyle="--", alpha=0.7, zorder=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    plt.xlim([-1, 21])
    plt.ylim([0, 0.5])
    plt.xlabel("x", fontsize=20)
    plt.ylabel("pmf", fontsize=20)
    plt.title("Poisson distribution")
    plt.legend(loc=1)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig("plots/poisson.png")


def plot_lines(
    prob_densities: dict,
    title: str,
    xlable: str,
    ylable: str,
    x_lim: tuple,
    y_lim: tuple,
    filename: str,
    smooth_lines=True,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))

    for prob_density in prob_densities.values():
        if smooth_lines:
            x = np.linspace(prob_density["x"].min(), prob_density["x"].max(), 300)
            spl = make_interp_spline(
                prob_density["x"], prob_density["y"], k=3
            )  # type: BSpline
            y = spl(x)
        else:
            x = prob_density["x"]
            y = prob_density["y"]
        ax.plot(
            x, y, label=prob_density["label"], color=prob_density["color"], linewidth=3
        )

    ax.grid(axis="x", linestyle="--", alpha=0.7, zorder=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel(xlable, fontsize=20)
    plt.ylabel(ylable, fontsize=20)
    plt.title(title, fontsize=25)
    plt.legend(loc=1)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(f"plots/{filename}")

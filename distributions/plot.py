import matplotlib.pyplot as plt
import numpy as np


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


def plot_exponential(
    X: np.ndarray, prob_density: np.ndarray, title: str, ylable: str, filename: str
) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.plot(X, prob_density, label="Training Error", color="green")

    ax.grid(axis="x", linestyle="--", alpha=0.7, zorder=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    plt.xlim([-1, 21])
    plt.ylim([0, 1.2])
    plt.xlabel("x", fontsize=20)
    plt.ylabel(ylable, fontsize=20)
    plt.title(title)
    plt.legend(loc=1)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(f"plots/{filename}")

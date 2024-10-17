import matplotlib.pyplot as plt
import numpy as np


def univariate_normal(mean: float, variance: float, x: np.ndarray) -> list:
    distribution = [
        1
        / np.sqrt(2 * np.pi * variance)
        * np.exp(-((mean - val) ** 2) / (2 * variance))
        for val in x
    ]
    return distribution


def get_conditional_mean_and_variance(
    mean_vector: np.ndarray, covariance_matrix: np.ndarray, condition: float
) -> tuple:
    mean = mean_vector[0] + covariance_matrix[0, 1] * (1 / covariance_matrix[1, 1]) * (
        condition - mean_vector[1]
    )
    variance = (
        covariance_matrix[0, 0]
        - covariance_matrix[0, 1]
        * (1 / covariance_matrix[1, 1])
        * covariance_matrix[1, 0]
    )

    return mean, variance


def plot_distributions(ax, x: np.ndarray, distributions: list, y_limit: tuple) -> None:
    for distribution in distributions:
        ax.plot(x, distribution["distribution"], label=distribution["label"])
    ax.set_xlabel("$x$", fontsize=13)
    ax.set_ylabel("density: $p(x)$", fontsize=13)
    ax.set_title("Univariate marginal distributions distributions")
    ax.set_ylim(y_limit)
    ax.set_xlim([-3, 13])
    ax.legend(loc=1)


if __name__ == "__main__":
    mean_vector = np.array([0.0, 2.0])
    covariance_matrix = np.array([[0.3, -1], [-1, 5]])

    conditional_mean, conditional_variance = get_conditional_mean_and_variance(
        mean_vector, covariance_matrix, -1
    )

    x = np.linspace(-5, 9, num=300)

    marginal_distributions = [
        {
            "distribution": univariate_normal(
                mean_vector[0], covariance_matrix[0, 0], x
            ),
            "label": f"$N({mean_vector[0]}, {covariance_matrix[0, 0]})-Marginal$",
        },
        {
            "distribution": univariate_normal(
                mean_vector[1], covariance_matrix[1, 1], x
            ),
            "label": f"$N({mean_vector[1]}, {covariance_matrix[1, 1]})-Marginal$",
        },
    ]

    conditional_distributions = [
        {
            "distribution": univariate_normal(
                conditional_mean, conditional_variance, x
            ),
            "label": f"$N({conditional_mean}, {conditional_variance})-Conditional$",
        },
    ]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    plot_distributions(ax1, x, marginal_distributions, (0, 1))
    plot_distributions(ax2, x, conditional_distributions, (0, 1.5))

    fig.subplots_adjust(bottom=0.15)
    plt.savefig("univariate_normal_marginal_distribution.png")
    plt.show()

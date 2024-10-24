import numpy as np

import plot


np.random.seed(0)

NO_OF_EXPERIMENTS = 10000


def get_probability_distribution(mu: float, n: int) -> tuple[list]:
    true_counts = np.random.binomial(n, mu, NO_OF_EXPERIMENTS)

    # computing the probability distribution
    x = []
    y = []
    unique, counts = np.unique(true_counts, return_counts=True)
    for true_count, count in zip(unique, counts):
        x.append(true_count)
        y.append(count / NO_OF_EXPERIMENTS)
    return x, y


if __name__ == "__main__":
    probability_distibutions = {}
    distribution_means = {0.1: "blue", 0.4: "red", 0.75: "green"}
    for mu, color in distribution_means.items():
        true_count, count = get_probability_distribution(mu, 15)
        probability_distibutions[mu] = {"x": true_count, "y": count, "color": color}

    plot.plot_binomial_distributions(probability_distibutions)

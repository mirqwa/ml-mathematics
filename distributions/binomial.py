import math

import numpy as np

import plot


np.random.seed(0)

NO_OF_EXPERIMENTS = 100000


def compute_probability_density(mu: float, true_counts: list):
    # n!/(m! * (n-m)!) * mu^m * (1-mu)^(n-m)
    probabilities = []
    for true_count in true_counts:
        prob = (
            math.factorial(15)
            / (math.factorial(true_count) * math.factorial(15 - true_count))
            * mu**true_count
            * (1 - mu) ** (15 - true_count)
        )
        probabilities.append(prob)
    return probabilities


def get_probability_distribution(mu: float, n: int) -> tuple[list]:
    true_counts = np.random.binomial(n, mu, NO_OF_EXPERIMENTS)

    # computing the probability distribution
    x = []
    y = []
    unique, counts = np.unique(true_counts, return_counts=True)
    for true_count, count in zip(unique, counts):
        x.append(true_count)
        y.append(count / NO_OF_EXPERIMENTS)
    computed_probs = compute_probability_density(mu, x)
    return x, y, computed_probs


if __name__ == "__main__":
    probability_distibutions = {}
    distribution_means = {0.1: "blue", 0.4: "red", 0.75: "green"}
    for mu, color in distribution_means.items():
        true_count, probs, computed_probs = get_probability_distribution(mu, 15)
        probability_distibutions[mu] = {
            "x": true_count,
            "y": probs,
            "y_computed": computed_probs,
            "color": color,
        }

    plot.plot_binomial_distributions(probability_distibutions)

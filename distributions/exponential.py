import numpy as np

import plot


def get_density_and_cumulative(mu: float) -> list[np.ndarray]:
    X = np.linspace(0, 20, 21)
    _lambda = 1 / mu
    prob_density = np.array([_lambda * np.exp(-_lambda * x) for x in X])
    cumulative_prob = np.array([1 - np.exp(-_lambda * x) for x in X])
    return X, prob_density, cumulative_prob


if __name__ == "__main__":
    prob_densities = {}
    cum_probs = {}
    distribution_means = {0.5: "blue", 1: "red", 5: "green"}
    for mu, color in distribution_means.items():
        X, prob_density, cum_prob = get_density_and_cumulative(mu)
        prob_densities[mu] = {"x": X, "y": prob_density, "color": color}
    plot.plot_exponential(
        prob_densities, "Exponential distribution", "pdf", "exponential_pdf.png"
    )

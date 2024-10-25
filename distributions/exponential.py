import numpy as np

import plot


def get_density_and_cumulative(mu: float) -> list[np.ndarray]:
    X = np.linspace(0, 20, 21)
    _lambda = 1 / mu
    prob_density = np.array([_lambda * np.exp(-_lambda * x) for x in X])
    cumulative_prob = np.array([1 - np.exp(-_lambda * x) for x in X])
    return X, prob_density, cumulative_prob


if __name__ == "__main__":
    X, prob_density, cum_prob = get_density_and_cumulative(1)
    plot.plot_exponential(
        X, prob_density, "Exponential distribution", "pdf", "exponential_pdf.png"
    )

import numpy as np

import plot


def get_density_and_cumulative(mu: float) -> list[np.ndarray]:
    X = np.linspace(0, 20, 50)
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
        prob_densities[mu] = {
            "x": X,
            "y": prob_density,
            "label": f"λ = {1 / mu}",
            "color": color,
        }
        cum_probs[mu] = {
            "x": X,
            "y": cum_prob,
            "label": f"λ = {1 / mu}",
            "color": color,
        }
    plot.plot_lines(
        prob_densities,
        "PDF for exponential distribution",
        "x",
        "pdf",
        (-1, 21),
        (0, 2.2),
        "exponential_pdf.png",
    )
    plot.plot_lines(
        cum_probs,
        "CDF for exponential distribution",
        "x",
        "cdf",
        (-1, 21),
        (0, 1.2),
        "exponential_cdf.png",
    )

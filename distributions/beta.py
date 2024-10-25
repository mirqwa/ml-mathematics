import numpy as np
from scipy.special import gamma

import plot


def get_distribution(alpha: float, beta: float) -> tuple:
    mus = np.linspace(0.0001, 0.99999, 1000)
    coeff = gamma(alpha + beta) / (gamma(alpha) * gamma(beta))
    distribution = np.array(
        [coeff * mu ** (alpha - 1) * (1 - mu) ** (beta - 1) for mu in mus]
    )
    return mus, distribution


if __name__ == "__main__":
    prob_distributions = {}
    dist_properties = {
        "α = 0.5 = β": {"alpha": 0.5, "beta": 0.5, "color": "blue"},
        "α = 1 = β": {"alpha": 1.0, "beta": 1.0, "color": "orange"},
        "α = 2, β = 0.3": {"alpha": 2.0, "beta": 0.3, "color": "green"},
        "α = 4, β = 10": {"alpha": 4.0, "beta": 10.0, "color": "pink"},
        "α = 5, β = 1": {"alpha": 5.0, "beta": 1.0, "color": "brown"},
    }
    for label, properties in dist_properties.items():
        mus, distribution = get_distribution(properties["alpha"], properties["beta"])
        prob_distributions[label] = {
            "x": mus,
            "y": distribution,
            "label": label,
            "color": properties["color"],
        }
    plot.plot_lines(
        prob_distributions,
        "Beta distribution",
        "µ",
        "p(µ|α, β)",
        (0, 1),
        (0, 10),
        "beta.png",
        smooth_lines=False,
    )

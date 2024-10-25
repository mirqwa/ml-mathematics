import math

import numpy as np

import plot


def get_probabilities(_lambda: float):
    X = np.linspace(0, 20, 21)
    probs = np.array(
        [(np.exp(-_lambda) * _lambda**x) / math.factorial(int(x)) for x in X]
    )
    return X, probs


if __name__ == "__main__":
    prob_distributions = {}
    distribution_means = {2: "blue", 10: "red", 15: "green"}
    for _lambda, color in distribution_means.items():
        x, probs = get_probabilities(_lambda)
        prob_distributions[_lambda] = {"x": x, "y": probs, "color": color}

    plot.plot_poisson(prob_distributions)

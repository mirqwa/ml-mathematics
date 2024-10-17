import matplotlib.pyplot as plt
import numpy as np


def univariate_normal(mean: float, variance: float, x: np.ndarray) -> list:
    distribution = [
        1 / np.sqrt(2 * np.pi * variance) * np.exp(-((mean - val) ** 2) / (2 * variance))
        for val in x
    ]
    return distribution


if __name__ == "__main__":
    x = np.linspace(-3, 5, num=150)
    fig = plt.figure(figsize=(5, 3))
    plt.plot(x, univariate_normal(0, 1, x), label="$N(0, 1)$")
    plt.plot(x, univariate_normal(2, 3, x), label="$n(2, 3)$")
    plt.plot(x, univariate_normal(0, 0.2, x), label="$n(0, 0.2)$")
    plt.xlabel("$x$", fontsize=13)
    plt.ylabel("density: $p(x)$", fontsize=13)
    plt.title("Univariate normal distributions")
    plt.ylim([0, 1])
    plt.xlim([-3, 5])
    plt.legend(loc=1)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig("univariate_normal_distribution.png")
    plt.show()

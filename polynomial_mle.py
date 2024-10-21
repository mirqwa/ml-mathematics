# getting the parameters of linear regression using MLE
# the expected param values θ = [2.1, -0.5, 1.9, 0.1, -1.5]
# yₙ = − sin(xₙ / 5) + cos(xₙ) + ϵ, ϵ∼(N|0,σ²) for σ=0.2
import math

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)


def generate_data() -> tuple[np.ndarray]:
    X = np.linspace(-5, 5, num=100)
    gaussian_noise = np.random.normal(0, 0.2, size=(100,))
    Y = np.array([-math.sin(x / 5) + math.cos(x) for x in X]) + gaussian_noise
    return X, Y


def plot_data(X: np.ndarray, Y: np.ndarray) -> None:
    fig = plt.figure(figsize=(5, 3))
    plt.scatter(X, Y, marker="x")
    fig.subplots_adjust(bottom=0.15)
    plt.savefig("polynomial_mle.png")
    plt.show()


if __name__ == "__main__":
    X, Y = generate_data()
    plot_data(X, Y)

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


def compute_parameters_and_predict(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # computes the params using MLE
    # θ = (ΦᵀΦ)-¹Φᵀy
    phi_x = np.array([[1, x, x**2, x**3, x**4] for x in X])
    phi_x_transpose = np.transpose(phi_x)
    theta = np.dot(np.linalg.inv(np.dot(phi_x_transpose, phi_x)), np.dot(phi_x_transpose, Y))
    prediction = np.dot(phi_x, theta)
    return prediction


def get_noise_variance_estimation(Y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    # computes the variance of the noise
    # given by the empirical mean of the squared distance between the noisy
    # observation and the noise-free function
    squared_distance = (Y - prediction) ** 2
    noise_variance = squared_distance.sum() / len(Y)


def plot_data(X: np.ndarray, Y: np.ndarray, prediction: np.ndarray) -> None:
    fig = plt.figure(figsize=(5, 3))
    plt.scatter(X, Y, marker="+", label="Training data")
    plt.plot(X, prediction, label="MLE", color="red")
    plt.xlabel("$x$", fontsize=13)
    plt.ylabel("$y$", fontsize=13)
    plt.title("MLE for polynomial")
    plt.legend(loc=1)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig("polynomial_mle.png")
    plt.show()


if __name__ == "__main__":
    X, Y = generate_data()
    prediction = compute_parameters_and_predict(X, Y)
    get_noise_variance_estimation(Y, prediction)
    plot_data(X, Y, prediction)

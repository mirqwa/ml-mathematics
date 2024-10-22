# getting the parameters of linear regression using MLE
# the expected param values θ = [2.1, -0.5, 1.9, 0.1, -1.5]
# yₙ = − sin(xₙ / 5) + cos(xₙ) + ϵ, ϵ∼(N|0,σ²) for σ=0.2
import math

import numpy as np

import plot_data

np.random.seed(0)


def generate_data() -> tuple[np.ndarray]:
    X = np.linspace(-5, 5, num=300)
    gaussian_noise = np.random.normal(0, 0.2, size=(300,))
    Y = np.array([-math.sin(x / 5) + math.cos(x) for x in X]) + gaussian_noise
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = (
        indices[: int(X.shape[0] * (2 / 3))],
        indices[int(X.shape[0] * (2 / 3)) :],
    )
    training_idx.sort()
    test_idx.sort()
    train_data = (X[training_idx], Y[training_idx])
    test_data = (X[test_idx], Y[test_idx])
    return train_data, test_data


def get_polynomial_input(X: np.ndarray, degree: int) -> np.ndarray:
    return np.array([[x**d for d in range(degree + 1)] for x in X])


def compute_parameters_and_predict(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # computes the params using MLE
    # θ = (ΦᵀΦ)-¹Φᵀy
    phi_x = get_polynomial_input(X, 4)
    phi_x_transpose = np.transpose(phi_x)
    theta = np.dot(
        np.linalg.inv(np.dot(phi_x_transpose, phi_x)), np.dot(phi_x_transpose, Y)
    )
    prediction = np.dot(phi_x, theta)
    return theta, prediction


def calculate_rmse(Y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    # computes the rmse between the actual and the predicted values
    squared_distance = (Y - prediction) ** 2
    return math.sqrt(squared_distance.sum() / len(Y))


if __name__ == "__main__":
    train_data, test_data = generate_data()
    X, Y = train_data
    X_test, Y_test = test_data
    theta, train_prediction = compute_parameters_and_predict(X, Y)
    rmse = calculate_rmse(Y, np.dot(get_polynomial_input(X, 4), theta))
    plot_data.plot_actual_and_predicted(
        X, Y, train_prediction, "plots/polynomial_mle.png", "MLE for polynomial"
    )

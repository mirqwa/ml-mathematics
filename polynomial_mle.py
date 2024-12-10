# getting the parameters of linear regression using MLE
# the expected param values θ = [2.1, -0.5, 1.9, 0.1, -1.5]
# yₙ = − sin(xₙ / 5) + cos(xₙ) + ϵ, ϵ∼(N|0,σ²) for σ=0.2
import math

import numpy as np

import plot_data

np.random.seed(0)


def generate_data() -> tuple[np.ndarray]:
    X = np.array([-4.5, -2.5, -0.5, 0, 1, 1.5, 3.6, 4, 4.25, 4.75])
    # X = np.linspace(-5, 5, num=10)
    gaussian_noise = np.random.normal(0, 0.2, size=(X.shape[0],))
    Y = np.array([-math.sin(x / 5) + math.cos(x) for x in X]) + gaussian_noise

    X_test = np.linspace(-5, 5, num=200)
    X_test = X_test[~np.isin(X_test, X)]
    test_gaussian_noise = np.random.normal(0, 0.2, size=(X_test.shape[0],))
    Y_test = (
        np.array([-math.sin(x / 5) + math.cos(x) for x in X_test]) + test_gaussian_noise
    )
    train_data = (X, Y)
    test_data = (X_test, Y_test)
    return train_data, test_data


def get_polynomial_input(X: np.ndarray, degree: int) -> np.ndarray:
    return np.array([[x**d for d in range(degree + 1)] for x in X])


def get_noise_variance_estimation(Y: np.ndarray, prediction: np.ndarray) -> float:
    squared_distance = (Y - prediction) ** 2
    noise_variance = squared_distance.sum() / len(Y)
    return noise_variance


def compute_parameters_and_predict(
    X: np.ndarray, Y: np.ndarray, degree: int
) -> np.ndarray:
    # computes the params using MLE
    # θ = (ΦᵀΦ)-¹Φᵀy
    phi_x = get_polynomial_input(X, degree)
    phi_x_transpose = np.transpose(phi_x)
    theta = np.dot(
        np.linalg.inv(np.dot(phi_x_transpose, phi_x)), np.dot(phi_x_transpose, Y)
    )
    prediction = np.dot(phi_x, theta)
    noise_variance = get_noise_variance_estimation(Y, prediction)
    return theta, prediction, noise_variance


def calculate_rmse(Y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    # computes the rmse between the actual and the predicted values
    squared_distance = (Y - prediction) ** 2
    return math.sqrt(squared_distance.sum() / len(Y))


def main() -> None:
    train_data, test_data = generate_data()
    X, Y = train_data
    X_test, Y_test = test_data
    train_rmse = []
    test_rmse = []
    for degree in range(10):
        theta, train_prediction, noise_variance = compute_parameters_and_predict(
            X, Y, degree
        )
        print(f"The estimated noise variance for degree={degree}:", noise_variance)
        train_rmse.append(
            calculate_rmse(Y, np.dot(get_polynomial_input(X, degree), theta))
        )
        test_rmse.append(
            calculate_rmse(Y_test, np.dot(get_polynomial_input(X_test, degree), theta))
        )
        plot_data.plot_actual_and_predicted(
            X,
            Y,
            train_prediction,
            f"plots/polynomial_mle_{degree}.png",
            f"MLE for polynomial with D={degree}",
        )

    plot_data.plot_rmse(train_rmse, test_rmse)


if __name__ == "__main__":
    main()
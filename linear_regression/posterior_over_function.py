import math

import numpy as np

import plot


np.random.seed(0)


def get_polynomial_input(X: np.ndarray, degree: int) -> np.ndarray:
    return np.array([[x**d for d in range(degree + 1)] for x in X])


def get_predictions():
    X = np.array([-4.5, -2.5, -0.5, 0, 1, 1.5, 3.6, 4, 4.25, 4.75])
    phi = get_polynomial_input(X, 5)
    variance = 0.2
    gaussian_noise = np.random.normal(0, math.sqrt(variance), size=(X.shape[0],))
    y = np.array([-math.sin(x / 5) + math.cos(x) for x in X]) + gaussian_noise
    y = y.reshape((10, 1))
    S_0 = 0.25 * np.identity(6)
    M_0 = np.zeros((6, 1))
    S_N = np.linalg.inv(np.linalg.inv(S_0) + variance * np.dot(np.transpose(phi), phi))
    M_N = np.dot(
        S_N,
        np.dot(np.linalg.inv(S_0), M_0) + variance * np.dot(np.transpose(phi), y),
    )
    Y = []
    for i in range(10):
        predictions = np.random.normal(
            np.dot(phi, M_N),
            np.abs(np.dot(np.dot(phi, S_N), np.transpose(phi)) + variance),
        )
        for pred in predictions:
            Y.append(
                {"y": pred, "color": "red", "width": 1, "chart_type": "line"}
            )
    return X, Y


if __name__ == "__main__":
    X, Y = get_predictions()

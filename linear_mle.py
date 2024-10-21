# getting the parameters of linear regression using MLE
# the expected param values θ = [2.1, -0.5, 1.9, 0.1, -1.5]
# y = xᵀθ + ϵ, ϵ∼(N|0,σ²) for σ=0.2
import numpy as np


np.random.seed(0)

EXPECTED_THETA = np.array([2.1, -0.5, 1.9, 0.1, -1.5])


def generate_data() -> None:
    # generate random feature values
    X = np.random.rand(100, 5) * 10
    X = X.round(2)

    gaussian_noise = np.random.normal(0, 0.2, size=(100,))

    Y = np.dot(X, EXPECTED_THETA) + gaussian_noise
    
    return X, Y


if __name__ == "__main__":
    X, Y = generate_data()

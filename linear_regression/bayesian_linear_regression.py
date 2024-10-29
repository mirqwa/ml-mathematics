import numpy as np

import plot

np.random.seed(0)


def get_prior_over_functions():
    X = np.linspace(-5, 5, 200)
    phi = np.array([[x**d for d in range(6)] for x in X])

    Y = []
    for i in range(10000):
        if i == 9999:
            theta = np.zeros((6, 1))
            color = "black"
            width = 1
            chart_type = "line"
        else:
            theta = np.random.normal(np.zeros((6, 1)), 0.25 * np.ones((6, 1)))
            if i < 9985:
                color = "grey"
                width = 5
                chart_type = "scatter"
            else:
                color = "red"
                width = 1
                chart_type = "line"
        prediction = np.sum(np.dot(phi, theta), axis=1)
        Y.append(
            {"y": prediction, "color": color, "width": width, "chart_type": chart_type}
        )

    return X, Y


if __name__ == "__main__":
    X, Y = get_prior_over_functions()
    plot.plot_lines(
        X,
        Y,
        "Bayesian Linear Regression",
        "x",
        "y",
        (-5, 5),
        (-5, 5),
        "bayesian_regression.png",
    )

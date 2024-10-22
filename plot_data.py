import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


def plot_actual_and_predicted(
    X: np.ndarray, Y: np.ndarray, prediction: np.ndarray, file_name: str, title: str
) -> None:
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(X, Y, marker="+", label="Training data")
    X_new = np.linspace(X.min(), X.max(), 300)
    spl = make_interp_spline(X, prediction, k=3)  # type: BSpline
    prediction_smooth = spl(X_new)
    plt.plot(X_new, prediction_smooth, label="MLE", color="red")
    plt.xlim([-5, 5])
    plt.xlabel("$x$", fontsize=13)
    plt.ylabel("y=−sin(xₙ/5) + cos(xₙ) + ϵ", fontsize=13)
    plt.title(title)
    plt.legend(loc=1)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(file_name)


def plot_rmse(train_rmse: list, test_rmse: list) -> None:
    fig = plt.figure(figsize=(12, 5))
    X = range(0, len(train_rmse))

    # training RMSE plot
    plt.plot(X, train_rmse, label="Training Error", color="green")

    # test RMSE plot
    plt.plot(X, test_rmse, label="Test Error", color="red")

    plt.xlim([0, 10])
    plt.ylim([0, 20])
    plt.xlabel("Degree of Polynomial", fontsize=13)
    plt.ylabel("RMSE", fontsize=13)
    plt.title("RMSE with polynomial degree")
    plt.legend(loc=1)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig("plots/rmse.png")

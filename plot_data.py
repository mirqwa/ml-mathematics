import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


def plot_actual_and_predicted(
    X: np.ndarray, Y: np.ndarray, prediction: np.ndarray, file_name: str, title: str
) -> None:
    fig = plt.figure(figsize=(5, 3))
    plt.scatter(X, Y, marker="+", label="Training data")
    X_new = np.linspace(X.min(), X.max(), 300)
    spl = make_interp_spline(X, prediction, k=3)  # type: BSpline
    prediction_smooth = spl(X_new)
    plt.plot(X_new, prediction_smooth, label="MLE", color="red")
    plt.xlim([-5, 5])
    plt.xlabel("$x$", fontsize=13)
    plt.ylabel("$y$", fontsize=13)
    plt.title(title)
    plt.legend(loc=1)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(file_name)
    #plt.show()

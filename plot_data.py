import matplotlib.pyplot as plt
import numpy as np


def plot_actual_and_predicted(
    X: np.ndarray, Y: np.ndarray, prediction: np.ndarray, file_name: str, title: str
) -> None:
    fig = plt.figure(figsize=(5, 3))
    plt.scatter(X, Y, marker="+", label="Training data")
    plt.plot(X, prediction, label="MLE", color="red")
    plt.xlabel("$x$", fontsize=13)
    plt.ylabel("$y$", fontsize=13)
    plt.title(title)
    plt.legend(loc=1)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(file_name)
    plt.show()

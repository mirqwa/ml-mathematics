import matplotlib.pyplot as plt
import numpy as np


def plot_lines(
    x: np.ndarray,
    Y: list,
    title: str,
    xlable: str,
    ylable: str,
    x_lim: tuple,
    y_lim: tuple,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(20, 10))

    for y in Y:
        if y["chart_type"] == "line":
            ax.plot(x, y["y"], color=y["color"], linewidth=y["width"])
        else:
            ax.scatter(x, y["y"], color=y["color"], s=200)

    ax.grid(axis="x", linestyle="--", alpha=0.7, zorder=0)
    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel(xlable, fontsize=20)
    plt.ylabel(ylable, fontsize=20)
    plt.title(title, fontsize=25)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(f"plots/{filename}")

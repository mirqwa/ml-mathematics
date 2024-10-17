import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator


def bivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    determinant = np.linalg.det(covariance)
    inverse = np.linalg.inv(covariance)
    x_m_transpose = np.transpose(x_m)
    return (
        1.0
        / (np.sqrt((2 * np.pi) ** d * determinant))
        * np.exp(-(x_m_transpose * inverse * x_m) / 2)
    )


# Plot bivariate distribution
def generate_surface(mean, covariance, d):
    """Helper function to generate density surface."""
    nb_of_x = 100  # grid size
    x1s = np.linspace(-5, 5, num=nb_of_x)
    x2s = np.linspace(-5, 5, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s)  # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i, j] = bivariate_normal(
                np.matrix([[x1[i, j]], [x2[i, j]]]), d, mean, covariance
            )
    return x1, x2, pdf  # x1, x2, pdf(x1,x2)


def plot_surface(ax, X, Y, Z, title) -> None:
    # Plot the surface.
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap="rainbow", linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(Z.min() - 0.02, Z.max() + 0.02)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter("{x:.02f}")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("$x_1$", fontsize=13)
    ax.set_ylabel("$x_2$", fontsize=13)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


def plot_distribution(ax, title: str, mean: list, covariance: list, d: int) -> None:
    bivariate_mean = np.matrix(mean)  # Mean
    bivariate_covariance = np.matrix(covariance)  # Covariance
    x1, x2, p = generate_surface(bivariate_mean, bivariate_covariance, d)
    plot_surface(ax, x1, x2, p, title)


if __name__ == "__main__":
    # subplot
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(8, 4), subplot_kw={"projection": "3d"}
    )
    d = 2  # number of dimensions

    # Plot of independent Normals
    plot_distribution(
        axes[0, 0], "Independent variables", [[0.0], [0.0]], [[1.0, 0.0], [0.0, 1.0]], d
    )

    # Plot of correlated Normals
    plot_distribution(
        axes[0, 1], "Correlated variables: 0.8", [[0.0], [1.0]], [[1.0, 0.8], [0.8, 1.0]], d
    )

    # Plot of correlated Normals
    plot_distribution(
        axes[1, 0], "Correlated variables: -8", [[0.0], [1.0]], [[1.0, -0.8], [-0.8, 1.0]], d
    )

    # Plot of correlated Normals
    plot_distribution(
        axes[1, 1], "Correlated variables: 0.5", [[0.0], [1.0]], [[1.0, 0.5], [0.5, 1.0]], d
    )

    plt.suptitle("Bivariate normal distributions", fontsize=13, y=0.95)
    plt.savefig("Bivariate_normal_distributon")
    plt.show()

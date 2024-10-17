import numpy as np
import matplotlib.pyplot as plt


def multivariate_normal(x, d, mean, covariance):
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
    x3s = np.linspace(-5, 5, num=nb_of_x)
    x1, x2, x3 = np.meshgrid(x1s, x2s, x3s)  # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            for k in range(nb_of_x):
                pdf[i, j, k] = multivariate_normal(
                    np.matrix([[x1[i, j, k]], [x2[i, j, k]], [x3[i, j, k]]]), d, mean, covariance
                )
    return x1, x2, x3, pdf  # x1, x2, pdf(x1,x2)


if __name__ == "__main__":
    # subplot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    d = 3  # number of dimensions

    # Plot of independent Normals
    multivariate_mean = np.matrix([[0.0], [0.0], [0.0]])  # Mean
    miltivariate_covariance = np.matrix(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )  # Covariance
    x1, x2, x3, p = generate_surface(multivariate_mean, miltivariate_covariance, d)
    breakpoint()
    # Plot bivariate distribution
    con = ax1.contourf(x1, x2, p, 100, cmap="rainbow")
    ax1.set_xlabel("$x_1$", fontsize=13)
    ax1.set_ylabel("$x_2$", fontsize=13)
    ax1.axis([-2.5, 2.5, -2.5, 2.5])
    ax1.set_aspect("equal")
    ax1.set_title("Independent variables", fontsize=12)

    # # Plot of correlated Normals
    # bivariate_mean = np.matrix([[0.0], [1.0]])  # Mean
    # bivariate_covariance = np.matrix([[1.0, 0.8], [0.8, 1.0]])  # Covariance
    # x1, x2, p = generate_surface(bivariate_mean, bivariate_covariance, d)
    # # Plot bivariate distribution
    # con = ax2.contourf(x1, x2, p, 100, cmap="rainbow")
    # ax2.set_xlabel("$x_1$", fontsize=13)
    # ax2.set_ylabel("$x_2$", fontsize=13)
    # ax2.axis([-2.5, 2.5, -1.5, 3.5])
    # ax2.set_aspect("equal")
    # ax2.set_title("Correlated variables", fontsize=12)

    # Add colorbar and title
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(con, cax=cbar_ax)
    cbar.ax.set_ylabel("$p(x_1, x_2)$", fontsize=13)
    plt.suptitle("Multivariate normal distributions", fontsize=13, y=0.95)
    plt.savefig("Multivariate_normal_distributon")
    plt.show()

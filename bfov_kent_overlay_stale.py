# Imports necessary libraries and functions.
from numpy import *
from numpy.random import randn, uniform
from scipy.special import jv as I_, gamma as G_
from matplotlib import pyplot as plt

# --- Projection Functions ---

# Converts equirectangular coordinates (longitude, latitude) to 3D Cartesian coordinates on the unit sphere.
def projectEquirectangular2Sphere(u, w, h):
    phi = u[:, 1] * (pi / float(h))     # Latitude angle (phi)
    theta = u[:, 0] * (2. * pi / float(w))  # Longitude angle (theta)
    sinphi = sin(phi)
    return vstack([sinphi * cos(theta), sinphi * sin(theta), cos(phi)]).T  # Cartesian coordinates (x, y, z)

# Converts 3D Cartesian coordinates on the unit sphere to equirectangular coordinates.
def projectSphere2Equirectangular(x, w, h):
    phi = squeeze(asarray(arccos(clip(x[:, 2], -1, 1))))     # Latitude angle (phi)
    theta = squeeze(asarray(arctan2(x[:, 1], x[:, 0])))      # Longitude angle (theta)
    theta[theta < 0] += 2 * pi         # Ensure theta is in [0, 2*pi)
    return vstack([theta * float(w) / (2. * pi), phi * float(h) / pi])

# --- Angle to Rotation Matrix ---

# Constructs a rotation matrix (Q) from Euler angles (alpha, eta, psi).
def angle2Gamma(alpha, eta, psi):
    gamma_1 = asarray([cos(alpha), sin(alpha) * cos(eta), sin(alpha) * sin(eta)])  # Unit mean axis
    gamma_2 = asarray([-cos(psi) * sin(alpha), cos(psi) * cos(alpha) * cos(eta) - sin(psi) * sin(eta),
                       cos(psi) * cos(alpha) * sin(eta) + sin(psi) * cos(eta)])  # Unit major axis
    gamma_3 = asarray([sin(psi) * sin(alpha), -sin(psi) * cos(alpha) * cos(eta) - cos(psi) * sin(eta),
                       -sin(psi) * cos(alpha) * sin(eta) + cos(psi) * cos(eta)])  # Unit minor axis
    return asarray([gamma_1, gamma_2, gamma_3])

# --- Kent (FB5) Distribution ---

# Calculates the probability density of the Kent (FB5) distribution at a given point x on the sphere.
def FB5(Theta, x):
    def __c(kappa, beta, terms=10):  # Normalization constant calculation
        su = 0
        for j in range(terms):
            su += G_(j + .5) / G_(j + 1) * beta ** (2 * j) * (2 / kappa) ** (2 * j + .5) * I_(2 * j + .5, kappa)
        return 2 * pi * su

    kappa, beta, Q = Theta           # Unpack parameters
    gamma_1, gamma_2, gamma_3 = Q    # Unpack rotation matrix

    # Ensure parameters are valid
    assert beta >= 0 and beta < kappa / 2
    assert isclose(dot(gamma_1, gamma_2), 0) and isclose(dot(gamma_2, gamma_3), 0)

    # Equation (2) from the paper: Kent distribution formula
    return __c(kappa, beta) ** (-1) * exp(
        kappa * dot(gamma_1, x) + beta * (dot(gamma_2, x) ** 2 - dot(gamma_3, x) ** 2))

# --- Main Execution ---

if __name__ == '__main__':
    h, w = 320, 640             # Height and width of equirectangular map
    v, u = mgrid[0:h:1, 0:w:1] # Create grid indices
    X = projectEquirectangular2Sphere(vstack((u.reshape(-1), v.reshape(-1))).T, w, h)  # Grid on sphere

    # --- Using Kent (FB5) with fixed parameters ---
    kappa = 1.0   # Concentration (larger values make the distribution more concentrated)
    beta = 0.     # Ovalness (0 for vMF, larger values increase ovalness)
    alpha = 2.5    # Euler angle 1
    eta = 1      # Euler angle 2
    psi = 1     # Euler angle 3
    Q = angle2Gamma(alpha, eta, psi) # Construct rotation matrix
    Theta = (kappa, beta, Q)  # Parameters for the Kent distribution

    P = asarray([FB5(Theta, x) for x in X])  # Calculate probabilities for all points

    # --- Visualization ---
    plt.imshow(P.reshape((h, w)))  # Show heatmap of probabilities
    plt.show()

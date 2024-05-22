import cv2
import numpy as np
import math
from numpy.linalg import norm
from skimage.io import imread
from scipy.special import jv as I_, gamma as G_
from matplotlib import pyplot as plt
from numpy import *
from typing import Tuple
import json


class Rotation:
    @staticmethod
    def Rx(alpha):
        return np.asarray([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])

    @staticmethod
    def Ry(beta):
        return np.asarray([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])

    @staticmethod
    def Rz(gamma):
        return np.asarray([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

class Plotting:
    @staticmethod
    def plotEquirectangular(image, kernel, color):
        resized_image = image
        kernel = kernel.astype(np.int32)
        hull = cv2.convexHull(kernel)
        cv2.polylines(resized_image,
                      [hull],
                      isClosed=True,
                      color=color,
                      thickness=2)
        return resized_image

# --- Helper Functions ---
def plot_circles(img, arr, color, alpha=0.4):
    """
    Plots circles with transparency on an image.

    Args:
        img: The image to draw circles on.
        arr: An array containing the center coordinates of the circles.
        color: The color of the circles (B, G, R).
        alpha: The transparency level (0.0 - 1.0), where 0.0 is fully transparent and 1.0 is fully opaque.

    Returns:
        The image with transparent circles drawn on it.
    """

    overlay = img.copy()

    for center in arr:
        cv2.circle(overlay, center, 10, (*color, int(255 * alpha)), -1)

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

# Helper function to calculate the projected coordinates of a point in 3D space
def project_point(point, R, w, h):
    point_rotated = np.dot(R, point / np.linalg.norm(point))
    phi = np.arctan2(point_rotated[0], point_rotated[2])
    theta = np.arcsin(point_rotated[1])
    u = (phi / (2 * np.pi) + 0.5) * w
    v = h - (-theta / np.pi + 0.5) * h
    return u, v

def plot_bfov(image: np.ndarray, v00: float, u00: float, 
              fov_lat: float, fov_long: float,
              color: Tuple[int, int, int], h: int, w: int) -> np.ndarray:
    """Plots a binocular field of view (BFOV) overlay on an equirectangular image.

    This function takes an equirectangular image and parameters defining the 
    position and size of a BFOV. It calculates the projection of a hemispherical 
    grid onto the image plane and marks the resulting points with circles.

    Args:
        image: The equirectangular image as a NumPy array (H x W x 3).
        v00: The vertical coordinate of the BFOV center in the image (pixels).
        u00: The horizontal coordinate of the BFOV center in the image (pixels).
        fov_lat: The latitude angle of the BFOV (radians).
        fov_long: The longitude angle of the BFOV (radians).
        color: The RGB color of the BFOV circles (e.g., (255, 0, 0) for red).
        h: The height of the image (pixels).
        w: The width of the image (pixels).

    Returns:
        The modified image with the BFOV overlay as a NumPy array (H x W x 3).
    """

    # Shift the image to center the BFOV
    t = int(w // 2 - u00)
    u00 += t
    image = np.roll(image, t, axis=1)

    # Calculate angles and projection parameters
    phi00 = (u00 - w / 2) * (2 * np.pi / w)
    theta00 = -(v00 - h / 2) * (np.pi / h)
    r = 10
    d_lat = r / (2 * np.tan(fov_lat / 2))
    d_long = r / (2 * np.tan(fov_long / 2))

    # Create rotation matrix
    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))

    # Create grid of points
    p = np.array([[i * d_lat / d_long, j, d_lat] 
                  for i in range(-(r - 1) // 2, (r + 1) // 2 + 1)
                  for j in range(-(r - 1) // 2, (r + 1) // 2 + 1)])

    # Project points and create kernel
    kernel = np.array([project_point(point, R, w, h) for point in p]).astype(np.int32)

    # Plot circles and equirectangular lines
    image = plot_circles(image, kernel, color)
    image = plotEquirectangular(image, kernel, color)

    # Shift the image back
    image = np.roll(image, w - t, axis=1)

    return image


def deg_to_rad(degrees):
    return [math.radians(degree) for degree in degrees]

# --- Projection Functions (Kent/FB5) ---

def projectEquirectangular2Sphere(u, w, h):
    phi = u[:, 1] * (pi / float(h))     # Latitude angle (phi)
    theta = u[:, 0] * (2. * pi / float(w))  # Longitude angle (theta)
    sinphi = sin(phi)
    return vstack([sinphi * cos(theta), sinphi * sin(theta), cos(phi)]).T  # Cartesian coordinates (x, y, z)

def projectSphere2Equirectangular(x, w, h):
    phi = squeeze(asarray(arccos(clip(x[:, 2], -1, 1))))     # Latitude angle (phi)
    theta = squeeze(asarray(arctan2(x[:, 1], x[:, 0])))      # Longitude angle (theta)
    theta[theta < 0] += 2 * pi         # Ensure theta is in [0, 2*pi)
    return vstack([theta * float(w) / (2. * pi), phi * float(h) / pi])

# --- Angle to Rotation Matrix ---

def angle2Gamma(alpha, eta, psi):
    gamma_1 = asarray([cos(alpha), sin(alpha) * cos(eta), sin(alpha) * sin(eta)])  # Unit mean axis
    gamma_2 = asarray([-cos(psi) * sin(alpha), cos(psi) * cos(alpha) * cos(eta) - sin(psi) * sin(eta),
                       cos(psi) * cos(alpha) * sin(eta) + sin(psi) * cos(eta)])  # Unit major axis
    gamma_3 = asarray([sin(psi) * sin(alpha), -sin(psi) * cos(alpha) * cos(eta) - cos(psi) * sin(eta),
                       -sin(psi) * cos(alpha) * sin(eta) + cos(psi) * cos(eta)])  # Unit minor axis
    return asarray([gamma_1, gamma_2, gamma_3])

# --- Kent (FB5) Distribution ---
def FB5(Theta, x):
    def __c(kappa, beta, terms=10):
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



if __name__ == "__main__":
    image = imread('/home/mstveras/360-indoor/images/7fB0x.jpg')
    h, w = image.shape[:2]
    with open('/home/mstveras/360-indoor/annotations/7fB0x.json', 'r') as f:
        data = json.load(f)
    boxes = data['boxes']
    classes = data['class']
    color_map = {4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 12: (255, 255, 0), 17: (0, 255, 255), 25: (255, 0, 255), 26: (128, 128, 0), 27: (0, 128, 128), 30: (128, 0, 128), 34: (128, 128, 128), 35: (64, 0, 0), 36: (0, 64, 0)}
    for i in range(len(boxes)):
        box = boxes[i]
        u00, v00, _, _, a_lat1, a_long1, class_name = box
        a_lat = np.radians(a_long1)
        a_long = np.radians(a_lat1)
        color = color_map.get(classes[i], (255, 255, 255))
        image = plot_bfov(image, v00, u00, a_lat, a_long, color, h, w)
    
    cv2.imwrite('final_image.png', image)

    # --- 2. Kent (FB5) Visualization ---
    v, u = mgrid[0:h:1, 0:w:1]
    X = projectEquirectangular2Sphere(vstack((u.reshape(-1), v.reshape(-1))).T, w, h)

    # --- Fixed Kent Parameters ---
    kappa = 10.0
    beta = 2.0
    alpha = 0.5
    eta = 1.2
    psi = -0.3
    Q = angle2Gamma(alpha, eta, psi)
    Theta = (kappa, beta, Q)

    P = asarray([FB5(Theta, x) for x in X])

    # --- Convert Probability to Image ---
    P_grayscale = (P - P.min()) / (P.max() - P.min()) * 255
    P_image = P_grayscale.reshape((h, w)).astype(np.uint8)
    heatmap = cv2.applyColorMap(P_image, cv2.COLORMAP_HOT)
    image_with_heatmap = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)

    # --- 3. Display or Save the Image ---
    cv2.imwrite('bfov_kent_overlay.png', image_with_heatmap) 

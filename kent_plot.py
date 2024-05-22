import cv2
import numpy as np
import math
from numpy.linalg import norm
from skimage.io import imread
from scipy.special import jv as I_, gamma as G_
from matplotlib import pyplot as plt
from numpy import *
import json
from typing import Tuple, Dict

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

def plot_bfov(
        image: np.ndarray,  # Equirectangular image (H x W x 3)
        v00: int,  # Vertical center coordinate of the BFOV
        u00: int,  # Horizontal center coordinate of the BFOV
        a_lat: float,  # Latitude angle of the BFOV (radians)
        a_long: float,  # Longitude angle of the BFOV (radians)
        color: Tuple[int, int, int],  # Color of the BFOV (BGR format)
        h,
        w
    ) -> np.ndarray:

    # Shift the image horizontally to ensure the BFOV center aligns with the image center.
    t = int(w//2 - u00)
    u00 += t
    image = np.roll(image, t, axis=1)

    # Calculate spherical coordinates (phi, theta) of the BFOV center in radians.
    phi00 = (u00 - w / 2.) * ((2. * np.pi) / w)
    theta00 = -(v00 - h / 2.) * (np.pi / h)

    # Define a grid of points within the BFOV using the given radius (r) and angles (a_lat, a_long).
    r = 10
    d_lat = r / (2 * np.tan(a_lat / 2))
    d_long = r / (2 * np.tan(a_long / 2))
    p = []
    for i in range(-(r - 1) // 2, (r + 1) // 2):
        for j in range(-(r - 1) // 2, (r + 1) // 2):
            p += [np.asarray([i * d_lat / d_long, j, d_lat])]

    # Create a rotation matrix to transform the grid points based on the BFOV center.
    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))  
    p = np.asarray([np.dot(R, (p[ij] / norm(p[ij]))) for ij in range(r * r)])  

    # Convert the 3D grid points into spherical coordinates (phi, theta).
    phi = np.asarray([np.arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
    theta = np.asarray([np.arcsin(p[ij][1]) for ij in range(r * r)])

    # Convert the spherical coordinates to 2D pixel coordinates (u, v).
    u = (phi / (2 * np.pi) + 1. / 2.) * w
    v = h - (-theta / np.pi + 1. / 2.) * h

    # Create a kernel (array of pixel coordinates) for plotting.
    kernel = np.stack((u, v), axis=-1).astype(np.int32)

    # Plot circles at the calculated pixel coordinates to visualize the BFOV.
    image = plot_circles(image, kernel, color)
    image = Plotting.plotEquirectangular(image, kernel, color)  # This line seems redundant; consider removing.

    # Shift the image back to its original position.
    image = np.roll(image, w - t, axis=1)

    return image


def deg_to_rad(degrees):
    return [math.radians(degree) for degree in degrees]

def projectEquirectangular2Sphere(u: np.ndarray, w: int, h: int) -> np.ndarray:
    """Projects 2D pixel coordinates (u, v) from an equirectangular image onto a unit sphere.

    Args:
        u (np.ndarray): Pixel coordinates (shape: (N, 2)).
        w (int): Image width.
        h (int): Image height.

    Returns:
        np.ndarray: Cartesian coordinates (x, y, z) on the unit sphere (shape: (N, 3)).
    """

    phi = u[:, 1] * (pi / float(h))
    theta = u[:, 0] * (2. * pi / float(w))
    sinphi = sin(phi)
    return vstack([sinphi * cos(theta), sinphi * sin(theta), cos(phi)]).T

def projectSphere2Equirectangular(x: np.ndarray, w: int, h: int) -> np.ndarray:
    """Projects 3D Cartesian coordinates (x, y, z) from a unit sphere onto an equirectangular image.

    Args:
        x (np.ndarray): Cartesian coordinates (shape: (N, 3)).
        w (int): Image width.
        h (int): Image height.

    Returns:
        np.ndarray: Pixel coordinates (u, v) in the equirectangular image (shape: (N, 2)).
    """
    phi = squeeze(asarray(arccos(clip(x[:, 2], -1, 1))))      # Latitude angle (phi)
    theta = squeeze(asarray(arctan2(x[:, 1], x[:, 0])))     # Longitude angle (theta)
    theta[theta < 0] += 2 * pi                                # Ensure theta is in [0, 2*pi)
    return vstack([theta * float(w) / (2. * pi), phi * float(h) / pi])

def angle2Gamma(alpha, eta, psi):
    gamma_1 = asarray([cos(alpha), sin(alpha) * cos(eta), sin(alpha) * sin(eta)])  # Unit mean axis
    gamma_2 = asarray([-cos(psi) * sin(alpha), cos(psi) * cos(alpha) * cos(eta) - sin(psi) * sin(eta),
                       cos(psi) * cos(alpha) * sin(eta) + sin(psi) * cos(eta)])  # Unit major axis
    gamma_3 = asarray([sin(psi) * sin(alpha), -sin(psi) * cos(alpha) * cos(eta) - cos(psi) * sin(eta),
                       -sin(psi) * cos(alpha) * sin(eta) + cos(psi) * cos(eta)])  # Unit minor axis
    return asarray([gamma_1, gamma_2, gamma_3])

def FB5(Theta: Tuple[float, float, np.ndarray], x: np.ndarray) -> float:
    """
    Calculates the probability density function (PDF) of the FB5 (Kent) distribution.

    Args:
        Theta: Tuple (kappa, beta, Q) defining the distribution.
            - kappa: Concentration parameter (spread around mean direction).
            - beta: Ovalness parameter (eccentricity).
            - Q: 3x3 rotation matrix (orientation).
        x: 3D unit vector on the unit sphere.

    Returns:
        The PDF of the FB5 distribution at point x.

    Raises:
        AssertionError: If parameters are invalid (beta or Q).
    """
    
    def __c(kappa, beta, terms=10):
        """
        Helper function to calculate the normalization constant of the FB5 distribution.

        Args:
            kappa: Concentration parameter.
            beta: Ovalness parameter.
            terms: Number of terms to include in the summation (default: 10).

        Returns:
            The calculated normalization constant.
        """
        su = 0
        for j in range(terms):
            su += G_(j + .5) / G_(j + 1) * beta ** (2 * j) * (2 / kappa) ** (2 * j + .5) * I_(2 * j + .5, kappa)
        return 2 * pi * su

    # Main function body
    kappa, beta, Q = Theta             # Unpack parameters
    gamma_1, gamma_2, gamma_3 = Q      # Unpack rotation matrix

    # Ensure parameters are valid
    assert beta >= 0 and beta < kappa / 2
    assert isclose(dot(gamma_1, gamma_2), 0) and isclose(dot(gamma_2, gamma_3), 0)

    # Calculate and return the PDF using Equation (2)
    return __c(kappa, beta) ** (-1) * exp(
        kappa * dot(gamma_1, x) + beta * (dot(gamma_2, x) ** 2 - dot(gamma_3, x) ** 2))

def load_image_and_annotations(image_path: str, annotation_path: str) -> Tuple[np.ndarray, int, int, list, list]:
    """Loads an equirectangular image and extracts bounding box and class information from a JSON annotation file.

    Args:
        image_path (str): The path to the equirectangular image file.
        annotation_path (str): The path to the JSON annotation file.

    Returns:
        Tuple[np.ndarray, int, int, list, list]: A tuple containing:
            - The resized image as a NumPy array.
            - The height of the image in pixels.
            - The width of the image in pixels.
            - A list of bounding boxes extracted from the annotation file.
            - A list of class labels corresponding to each bounding box.
    """
    image = cv2.imread(image_path)
    #image = cv2.resize(image, (920, 460))
    h, w = image.shape[:2]

    with open(annotation_path, 'r') as f:
        data = json.load(f)
    boxes = data['boxes']
    classes = data['class']
    return image, h, w, boxes, classes

def generate_color_map() -> Dict[int, Tuple[int, int, int]]:
    """Creates a dictionary mapping class IDs to BGR colors.

    Returns:
        Dict[int, Tuple[int, int, int]]: A dictionary where keys are class IDs and values are BGR color tuples.
    """
    return {
        4: (0, 0, 255), 5: (0, 255, 0), 6: (255, 0, 0), 12: (255, 255, 0), 17: (0, 255, 255),
        25: (255, 0, 255), 26: (128, 128, 0), 27: (0, 128, 128), 30: (128, 0, 128), 34: (128, 128, 128),
        35: (64, 0, 0), 36: (0, 64, 0)
    }

def visualize_kent_distribution(
    image: np.ndarray, 
    h: int, 
    w: int, 
    kappa: float = 10.0, 
    beta: float = 2.0, 
    alpha: float = 0.5, 
    eta: float = 1.2, 
    psi: float = -0.3
) -> np.ndarray:
    """Visualizes the Kent distribution on an equirectangular image.

    Args:
        image (np.ndarray): The equirectangular image as a NumPy array.
        h (int): The height of the image in pixels.
        w (int): The width of the image in pixels.
        kappa (float, optional): The concentration parameter of the Kent distribution. Defaults to 10.0.
        beta (float, optional): The ovalness parameter of the Kent distribution. Defaults to 2.0.
        alpha (float, optional): The first Euler angle for the rotation matrix. Defaults to 0.5.
        eta (float, optional): The second Euler angle for the rotation matrix. Defaults to 1.2.
        psi (float, optional): The third Euler angle for the rotation matrix. Defaults to -0.3.

    Returns:
        np.ndarray: The image overlaid with the Kent distribution heatmap.
    """
    v, u = mgrid[0:h:1, 0:w:1]
    X = projectEquirectangular2Sphere(vstack((u.reshape(-1), v.reshape(-1))).T, w, h)

    Q = angle2Gamma(alpha, eta, psi)
    Theta = (kappa, beta, Q)

    P = asarray([FB5(Theta, x) for x in X])
    P_grayscale = (P - P.min()) / (P.max() - P.min()) * 255
    P_image = P_grayscale.reshape((h, w)).astype(np.uint8)
    heatmap = cv2.applyColorMap(P_image, cv2.COLORMAP_HOT)
    image_with_heatmap = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return image_with_heatmap

if __name__ == "__main__":
    image_path = '/home/mstveras/360-indoor/images/7fB0x.jpg'
    annotation_path = '/home/mstveras/360-indoor/annotations/7fB0x.json'

    new_h, new_w = 460, 920

    image, h, w, boxes, classes = load_image_and_annotations(image_path, annotation_path)
    color_map = generate_color_map()

    image = cv2.resize(image, (new_w,new_h))

    for i in range(len(boxes)):
        box = boxes[i]
        u00, v00, _, _, a_long, a_lat, _ = box
        u00 = u00/w*new_w
        v00 = v00/h*new_h
        a_lat = np.radians(a_lat)
        a_long = np.radians(a_long)
        color = color_map.get(classes[i], (255, 255, 255))
        image = plot_bfov(image, v00, u00, a_lat, a_long, color, new_h, new_w)

    cv2.imwrite('final_image.png', image)

    image_with_heatmap = visualize_kent_distribution(image, new_h, new_w)
    cv2.imwrite('bfov_kent_overlay.png', image_with_heatmap)

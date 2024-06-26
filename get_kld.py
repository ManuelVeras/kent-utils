import pdb
import numpy as np
import json
from numpy import rad2deg

def angles2Q(alpha, beta, eta):
    
    gamma_1 = np.array([
    [np.cos(alpha)],
    [np.sin(alpha) * np.cos(eta)],
    [np.sin(alpha) * np.sin(eta)]
    ])

    gamma_2 = np.array([
    [-np.cos(beta) * np.sin(alpha)],
    [np.cos(beta) * np.cos(alpha) * np.cos(eta) - np.sin(beta) * np.sin(eta)],
    [np.cos(beta) * np.cos(alpha) * np.sin(eta) + np.sin(beta) * np.cos(eta)]
    ])

    gamma_3 = np.array([
    [np.sin(beta) * np.sin(alpha)],
    [-np.sin(beta) * np.cos(alpha) * np.cos(eta) - np.cos(beta) * np.sin(eta)],
    [-np.sin(beta) * np.cos(alpha) * np.sin(eta) + np.cos(beta) * np.cos(eta)]
    ])

    return np.concatenate((gamma_1, gamma_2, gamma_3), axis =1)

def c_approx(kappa: float, beta: float) -> float:
    """Approximation for the normalization constant c(kappa, beta)."""
    return 2 * np.pi * np.exp(kappa) * ((kappa - 2 * beta) * (kappa + 2 * beta))**(-0.5)

def del_kappa(kappa: float, beta: float) -> float:
    """Calculates the first derivative wrt kappa."""

    # Handle potential division by zero
    if kappa - 2 * beta == 0 or kappa + 2 * beta == 0:
        raise ZeroDivisionError("Division by zero occurred in the expression.")

    # Factor out a negative sign from the numerator
    numerator = -2 * np.pi * (4 * beta**2 + kappa - kappa**2) * np.exp(kappa)  

    # Simplify the denominator using the difference of squares
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2)

    result = numerator / denominator
    return result

def del_2_kappa(kappa: float, beta: float) -> float:
    """Calculates the second derivative wrt kappa."""

    # Handle potential division by zero
    if kappa - 2 * beta == 0 or kappa + 2 * beta == 0:
        raise ZeroDivisionError("Division by zero occurred in the expression.")

    # Calculate the numerator
    numerator = 2 * np.pi * (kappa**4 - 2 * kappa**3 + (2 - 8 * beta**2) * kappa**2 + 8 * beta**2 * kappa + 16 * beta**4 + 4 * beta**2) * np.exp(kappa)

    # Calculate the denominator
    denominator = (kappa - 2 * beta)**(5/2) * (kappa + 2 * beta)**(5/2)

    result = numerator / denominator
    return result

def del_beta(kappa, beta):
  """Calculates the third npematical expression with inputs kappa and beta."""

  # Handle potential division by zero
  if kappa - 2 * beta == 0 or 2 * beta + kappa == 0:
    raise ZeroDivisionError("Division by zero occurred in the expression.")

  # Calculate the numerator
  numerator = 8 * np.pi * np.exp(kappa) * beta

  # Calculate the denominator using the difference of squares pattern
  denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta) ** (3/2)

  result = numerator / denominator
  return result

def E_x(Q_matrix: np.ndarray, kappa: float, beta: float) -> np.ndarray:

    return del_kappa(kappa, beta) / c_approx(kappa, beta) * Q_matrix[:,0]

def ExxT(
    Q_matrix: np.ndarray, kappa: float, beta: float
) -> np.ndarray:
    
    # Helper functions (presumably defined elsewhere) to calculate compliance matrix derivatives
    c = c_approx(kappa, beta)
    c_k = del_kappa(kappa, beta)
    c_kk = del_2_kappa(kappa, beta)
    c_beta = del_beta(kappa, beta)

    # Calculate the lambda values
    lambda_1 = c_k / c
    lambda_2 = (c - c_kk + c_beta) / (2 * c)
    lambda_3 = (c - c_kk - c_beta) / (2 * c)

    # Construct the lambda matrix
    lambda_matrix = np.array([[lambda_1, 0, 0], [0, lambda_2, 0], [0, 0, lambda_3]])

    # Calculate and return ExxT
    return Q_matrix @ lambda_matrix @ Q_matrix.T

def kld(
    kappa_a: float, beta_a: float, Q_matrix_a: np.ndarray,
    kappa_b: float, beta_b: float, Q_matrix_b: np.ndarray
) -> float:
    """Calculates the expected log-likelihood ratio between two Kent distributions.

    This function assumes the expectations under both Kent distributions have been pre-calculated.
    The rotation matrices are passed as arrays, and their components are extracted within the function.

    Args:
        Ex_a: Expected value of x under Kent distribution 'a' (1D NumPy array).
        ExxT_a: Expected value of x * x.T under Kent distribution 'a' (2D NumPy array).
        ca: Normalization constant for Kent distribution 'a'.
        kappa_a, beta_a: Concentration and ovalness parameters for Kent distribution 'a'.
        Q_matrix_a: Rotation matrix for Kent distribution 'a' (3x3 NumPy array).
        Ex_b, ExxT_b, cb, kappa_b, beta_b, Q_matrix_b: Same as above, but for Kent distribution 'b'.

    Returns:
        The expected log-likelihood ratio (float).
    """
    gamma_a1, gamma_a2, gamma_a3 = Q_matrix_a[:, 0], Q_matrix_a[:, 1], Q_matrix_a[:, 2] # Extract the first and second column vectors of Gamma_a
    gamma_b1, gamma_b2, gamma_b3 = Q_matrix_b[:, 0], Q_matrix_b[:, 1], Q_matrix_a[:, 2] # Extract the first and second column vectors of Gamma_b

    ca = c_approx(kappa_a, beta_a)
    cb = c_approx(kappa_b, beta_b)

    ExxT_a = ExxT(Q_matrix_a, kappa_a, beta_a)

    Ex_a = E_x(Q_matrix_a, kappa_a, beta_a)

    #term2 = (kappa_a * gamma_a1.T - kappa_b * gamma_b1.T) @ Ex_a
    #term3 =  (beta_a * gamma_a2.T @ ExxT_a @ gamma_a2)


    result = (
        np.log(cb / ca) + (kappa_a * gamma_a1.T - kappa_b * gamma_b1.T) @ Ex_a 
        + (beta_a * gamma_a2.T @ ExxT_a @ gamma_a2) - (beta_b * gamma_b2.T @ ExxT_a @ gamma_b2)
        - (beta_a * gamma_a3.T @ ExxT_a @ gamma_a3) + (beta_b * gamma_b3.T @ ExxT_a @ gamma_b3)
    )

    return result.item() 

def check_orthonormality_and_beta(A, kappa, beta):

    # Check beta condition
    assert 0 <= beta < kappa / 2, f"Beta condition not met: 0 < {beta} < {kappa/2} is False"

    # Check orthonormality of
    identity = np.eye(3)
    assert np.allclose(A.T @ A, identity), "Columns of A are not orthonormal"

if __name__ == "__main__":
    
    input_filename =  'datasets/360INDOOR/annotations/instances_val2017_transformed.json'

    #first distribution   
    #kappa_a = 10
    #beta_a = 2
    #Q_matrix_a = np.array([[1,0,0], [0,1,0], [0,0,1]])
    #Q_matrix_a = np.array([[np.sqrt(1/2), np.sqrt(1/2), 0], [np.sqrt(1/2), -np.sqrt(1/2), 0], [0, 0, 1]])

    #Second distribution
    # Q_matrix_b = np.array([[-1,0,0], [0,-1,0], [0,0,-1]])
    #Q_matrix_b = np.array([[0,0,-1], [0,1,0], [-1,0,0]])
    #Q_matrix_b = np.array([[0,0,1], [0,1,0], [1,0,0]])
    #Q_matrix_b = angles2Q(alpha = 90, beta = 90, eta = 90)

    #with open(input_filename, 'r') as file:
    #    json_data = json.load(file)

    # Process the annotations
    #annotations = json_data['annotations']
    #ann_example = annotations[0]['bbox']

    #Q_matrix_b = angles2Q(alpha = -90, beta = -90, eta = -90)
    #Q_matrix_b = angles2Q(alpha = rad2deg(ann_example[0]), beta = rad2deg(ann_example[1]), eta = rad2deg(ann_example[2]))

    #kappa_b = ann_example[3]
    #beta_b = ann_example[4]

    #check_orthonormality_and_beta(Q_matrix_a, kappa_a, beta_a)


    kappa_a = 2*10.1
    beta_a = 4.1
    phi, psi, eta = 20., 0., 0.
    Q_matrix_a = angles2Q(phi, psi,eta)

    # Second distribution
    kappa_b = 10.1
    beta_b = 4.1
    phi_b, psi_b, eta_b = 0., 0., 0.
    Q_matrix_b = angles2Q(phi_b, psi_b,eta_b)

    kld_value = kld(kappa_a, beta_a, Q_matrix_a, kappa_b, beta_b, Q_matrix_b)
    #kld_value = kld(kappa_b, beta_b, Q_matrix_b, kappa_a, beta_a, Q_matrix_a)

    print(f"KLD values is {kld_value}")
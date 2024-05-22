import math
import numpy as np

def del_k(k, beta):
    """Calculates the first derivative wrt kappa.

    Args:
        k: A numerical value representing the variable k.
        beta: A numerical value representing the variable beta.

    Returns:
        The calculated result of the expression.
    """

    # Handle potential division by zero
    if k - 2 * beta == 0 or k + 2 * beta == 0:
        raise ZeroDivisionError("Division by zero occurred in the expression.")

    # Factor out a negative sign from the numerator
    numerator = -2 * math.pi * (4 * beta**2 + k - k**2) * math.exp(k)  

    # Simplify the denominator using the difference of squares
    denominator = (k - 2 * beta)**(3/2) * (k + 2 * beta)**(3/2)

    result = numerator / denominator
    return result

def del_2_z(kappa, beta):
    """Calculates the second derivative wrt kappa.

    Args:
        kappa: A numerical value representing the variable kappa.
        beta: A numerical value representing the variable beta.

    Returns:
        The calculated result of the expression.
    """

    # Handle potential division by zero
    if kappa - 2 * beta == 0 or kappa + 2 * beta == 0:
        raise ZeroDivisionError("Division by zero occurred in the expression.")

    # Calculate the numerator
    numerator = 2 * math.pi * (kappa**4 - 2 * kappa**3 + (2 - 8 * beta**2) * kappa**2 + 8 * beta**2 * kappa + 16 * beta**4 + 4 * beta**2) * math.exp(kappa)

    # Calculate the denominator
    denominator = (kappa - 2 * beta)**(5/2) * (kappa + 2 * beta)**(5/2)

    result = numerator / denominator
    return result


def del_beta(kappa, beta):
  """Calculates the derivative wrt beta.

  Args:
      kappa: A numerical value representing the variable kappa.
      beta: A numerical value representing the variable beta.

  Returns:
      The calculated result of the expression.
  """

  # Handle potential division by zero
  if kappa - 2 * beta == 0 or 2 * beta + kappa == 0:
    raise ZeroDivisionError("Division by zero occurred in the expression.")

  # Calculate the numerator
  numerator = 8 * math.pi * math.exp(kappa) * beta

  # Calculate the denominator using the difference of squares pattern
  denominator = (kappa - 2 * beta) * (kappa + 2 * beta) ** (3/2)

  result = numerator / denominator
  return result

def kld(
    Ex_a: np.ndarray, ExxT_a: np.ndarray, ca: float, kappa_a: float, beta_a: float, Gamma_a: np.ndarray,
    Ex_b: np.ndarray, ExxT_b: np.ndarray, cb: float, kappa_b: float, beta_b: float, Gamma_b: np.ndarray
) -> float:
    """Calculates the expected log-likelihood ratio between two Kent distributions.

    This function assumes the expectations under both Kent distributions have been pre-calculated.
    The rotation matrices are passed as arrays, and their components are extracted within the function.

    Args:
        Ex_a: Expected value of x under Kent distribution 'a' (1D NumPy array).
        ExxT_a: Expected value of x * x.T under Kent distribution 'a' (2D NumPy array).
        ca: Normalization constant for Kent distribution 'a'.
        kappa_a, beta_a: Concentration and ovalness parameters for Kent distribution 'a'.
        Gamma_a: Rotation matrix for Kent distribution 'a' (3x3 NumPy array).
        Ex_b, ExxT_b, cb, kappa_b, beta_b, Gamma_b: Same as above, but for Kent distribution 'b'.

    Returns:
        The expected log-likelihood ratio (float).
    """
    gamma_a1, gamma_a2, gamma_a3 = Gamma_a[:, 0], Gamma_a[:, 1], Gamma_a[:, 2] # Extract the first and second column vectors of Gamma_a
    gamma_b1, gamma_b2, gamma_b3 = Gamma_b[:, 0], Gamma_b[:, 1], Gamma_a[:, 2] # Extract the first and second column vectors of Gamma_b

    result = (
        np.log(cb / ca) 
        + (kappa_a * gamma_a1.T - kappa_b * gamma_b1.T) @ Ex_a 
        + (beta_a * gamma_a2.T @ ExxT_a @ gamma_a2 
        - beta_b * gamma_b2.T @ ExxT_b @ gamma_b2)
        + beta_a * gamma_a3.T @ ExxT_a @ gamma_a3 
        - beta_b * gamma_b3.T @ ExxT_b @ gamma_b3
    )

    return result.item() 

if __name__ == "__main__":
    k_value = 5
    beta_value = 1
    result = del_k(k_value, beta_value)
    print(f"Result for k={k_value}, beta={beta_value}: {result}")
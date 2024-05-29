import numpy as np
from scipy.special import iv as I_ # For modified Bessel functions

def objective_function(k_i, k_j, mu_j):
    """
    Implements the objective function.

    Args:
        k_i: The variable to optimize.
        k_j: Reference value for k_i.
        mu_j: Mean or expected value associated with k_j.
    Returns:
        The value of the objective function     .
    """

    def A_d(k):
        return I_(3/2, k) / I_(1/2, k)

    def C_d(k):
        # You'll need to replace this with your actual cost function
        return 2 * np.pi * np.exp(k) * (k)**(-1)

    print(A_d(k_i))
    print(np.log(C_d(k_i) / C_d(k_j)))
    return np.log(C_d(k_i) / C_d(k_j)) + A_d(k_i) * (k_i - k_j * mu_i @ mu_j.T)

# Example usage
k_i = 10  
k_j = 20
mu_j = np.array([0.,1,0])
mu_i = np.array([0,1,0])

result = objective_function(k_i, k_j, mu_j)
print(f"Objective function value: {result}")

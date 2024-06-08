import torch
import pdb

def angles2Q(alpha, beta, eta):
    alpha, beta, eta = torch.tensor(alpha), torch.tensor(beta), torch.tensor(eta)  # Convert to tensors

    gamma_1 = torch.tensor([
        [torch.cos(alpha)],
        [torch.sin(alpha) * torch.cos(eta)],
        [torch.sin(alpha) * torch.sin(eta)]
    ], dtype=torch.float32)

    gamma_2 = torch.tensor([
        [-torch.cos(beta) * torch.sin(alpha)],
        [torch.cos(beta) * torch.cos(alpha) * torch.cos(eta) - torch.sin(beta) * torch.sin(eta)],
        [torch.cos(beta) * torch.cos(alpha) * torch.sin(eta) + torch.sin(beta) * torch.cos(eta)]
    ], dtype=torch.float32)

    gamma_3 = torch.tensor([
        [torch.sin(beta) * torch.sin(alpha)],
        [-torch.sin(beta) * torch.cos(alpha) * torch.cos(eta) - torch.cos(beta) * torch.sin(eta)],
        [-torch.sin(beta) * torch.cos(alpha) * torch.sin(eta) + torch.cos(beta) * torch.cos(eta)]
    ], dtype=torch.float32)

    return torch.cat((gamma_1, gamma_2, gamma_3), axis=1)


def c_approx(kappa, beta):
    return 2 * torch.pi * torch.exp(kappa) * ((kappa - 2 * beta) * (kappa + 2 * beta))**(-0.5)

def del_kappa(kappa, beta):
    numerator = -2 * torch.pi * (4 * beta**2 + kappa - kappa**2) * torch.exp(kappa)
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2)
    return numerator / denominator

def del_2_kappa(kappa, beta):
    numerator = 2 * torch.pi * (kappa**4 - 2 * kappa**3 + (2 - 8 * beta**2) * kappa**2 + 8 * beta**2 * kappa + 16 * beta**4 + 4 * beta**2) * torch.exp(kappa)
    denominator = (kappa - 2 * beta)**(5/2) * (kappa + 2 * beta)**(5/2)
    return numerator / denominator

def del_beta(kappa, beta):
    numerator = 8 * torch.pi * torch.exp(kappa) * beta
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2)
    return numerator / denominator

def E_x(Q_matrix, kappa, beta):
    return del_kappa(kappa, beta) / c_approx(kappa, beta) * Q_matrix[:, 0]

def ExxT(Q_matrix, kappa, beta):
    c = c_approx(kappa, beta)
    c_k = del_kappa(kappa, beta)
    c_kk = del_2_kappa(kappa, beta)
    c_beta = del_beta(kappa, beta)

    lambda_1 = c_k / c
    lambda_2 = (c - c_kk + c_beta) / (2 * c)
    lambda_3 = (c - c_kk - c_beta) / (2 * c)

    lambda_matrix = torch.diag(torch.tensor([lambda_1, lambda_2, lambda_3]))  # Use torch.diag for efficiency
    return torch.matmul(Q_matrix, torch.matmul(lambda_matrix, Q_matrix.T))

def kld(kappa_a, beta_a, Q_matrix_a, kappa_b, beta_b, Q_matrix_b):
    gamma_a1, gamma_a2, gamma_a3 = Q_matrix_a[:, 0], Q_matrix_a[:, 1], Q_matrix_a[:, 2]
    gamma_b1, gamma_b2, gamma_b3 = Q_matrix_b[:, 0], Q_matrix_b[:, 1], Q_matrix_b[:, 2]

    ca = c_approx(kappa_a, beta_a)
    cb = c_approx(kappa_b, beta_b)

    ExxT_a = ExxT(Q_matrix_a, kappa_a, beta_a)
    Ex_a = E_x(Q_matrix_a, kappa_a, beta_a)

    result = (
        torch.log(cb / ca) 
        + torch.matmul((kappa_a * gamma_a1.T - kappa_b * gamma_b1.T), Ex_a)
        + (beta_a * torch.matmul(torch.matmul(gamma_a2.T, ExxT_a), gamma_a2)) 
        - (beta_b * torch.matmul(torch.matmul(gamma_b2.T, ExxT_a), gamma_b2))
        - (beta_a * torch.matmul(torch.matmul(gamma_a3.T, ExxT_a), gamma_a3)) 
        + (beta_b * torch.matmul(torch.matmul(gamma_b3.T, ExxT_a), gamma_b3))
    )

    return result.item()


def check_orthonormality_and_beta(A, B, kappa_a, beta_a, kappa_b, beta_b):
    assert 0 <= beta_a < kappa_a / 2, f"Beta condition not met: 0 < {beta_a} < {kappa_a/2} is False"
    assert 0 <= beta_b < kappa_b / 2, f"Beta condition not met: 0 < {beta_b} < {kappa_b/2} is False"

    identity = torch.eye(3)
    assert torch.allclose(torch.matmul(A.T, A), identity), "Columns of A are not orthonormal"
    assert torch.allclose(torch.matmul(B.T, B), identity), "Columns of B are not orthonormal"


if __name__ == "__main__":

    #First Distribution
    kappa_a = 10.1
    beta_a = 2.1
    Q_matrix_a = torch.tensor([[1,0,0], [0,1,0], [0,0,1]], dtype=torch.float32)
    #Q_matrix_a = np.array([[np.sqrt(1/2), np.sqrt(1/2), 0], [np.sqrt(1/2), -np.sqrt(1/2), 0], [0, 0, 1]])

    #Second distribution
    kappa_b = 10.1
    beta_b = 4.2

    #Q_matrix_b = angles2Q(alpha = -90, beta = -90, eta = -90)
    Q_matrix_b = torch.tensor([[1,0,0], [0,1,0], [0,0,1]], dtype=torch.float32)


    # Ensure that the data is converted to PyTorch tensors
    kappa_a, beta_a = torch.tensor(kappa_a), torch.tensor(beta_a)
    kappa_b, beta_b = torch.tensor(kappa_b), torch.tensor(beta_b)

    kappa_a, beta_a = kappa_a.float(), beta_a.float()
    kappa_b, beta_b = kappa_b.float(), beta_b.float()


    Q_matrix_a, Q_matrix_b = torch.tensor(Q_matrix_a), torch.tensor(Q_matrix_b) 
  
    check_orthonormality_and_beta(Q_matrix_a, Q_matrix_b, kappa_a, beta_a, kappa_b, beta_b)

    kld_value = kld(kappa_a, beta_a, Q_matrix_a, kappa_b, beta_b, Q_matrix_b)
    print(f"KLD value is {kld_value}")
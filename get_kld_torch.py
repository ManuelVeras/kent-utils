import torch

def angles_to_Q(alpha: torch.Tensor, beta: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    N = alpha.size(0)
    alpha = alpha.view(N, 1)
    beta = beta.view(N, 1)
    eta = eta.view(N, 1)
    
    gamma_1 = torch.cat([
        torch.cos(alpha),
        torch.sin(alpha) * torch.cos(eta),
        torch.sin(alpha) * torch.sin(eta)
    ], dim=1).unsqueeze(2)

    gamma_2 = torch.cat([
        -torch.cos(beta) * torch.sin(alpha),
        torch.cos(beta) * torch.cos(alpha) * torch.cos(eta) - torch.sin(beta) * torch.sin(eta),
        torch.cos(beta) * torch.cos(alpha) * torch.sin(eta) + torch.sin(beta) * torch.cos(eta)
    ], dim=1).unsqueeze(2)

    gamma_3 = torch.cat([
        torch.sin(beta) * torch.sin(alpha),
        -torch.sin(beta) * torch.cos(alpha) * torch.cos(eta) - torch.cos(beta) * torch.sin(eta),
        -torch.sin(beta) * torch.cos(alpha) * torch.sin(eta) + torch.cos(beta) * torch.cos(eta)
    ], dim=1).unsqueeze(2)

    gamma = torch.cat((gamma_1, gamma_2, gamma_3), dim=2)
    return gamma

def c_approximation(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    return 2 * torch.pi * torch.exp(kappa) * ((kappa - 2 * beta) * (kappa + 2 * beta))**(-0.5)

def delta_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    numerator = -2 * torch.pi * (4 * beta**2 + kappa - kappa**2) * torch.exp(kappa)
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2)
    return numerator / denominator

def delta_2_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    numerator = 2 * torch.pi * (kappa**4 - 2 * kappa**3 + (2 - 8 * beta**2) * kappa**2 + 8 * beta**2 * kappa + 16 * beta**4 + 4 * beta**2) * torch.exp(kappa)
    denominator = (kappa - 2 * beta)**(5/2) * (kappa + 2 * beta)**(5/2)
    return numerator / denominator

def delta_beta(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    numerator = 8 * torch.pi * torch.exp(kappa) * beta
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2)
    return numerator / denominator

def expected_x(gamma_a1: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    const = (c_k / c).view(-1, 1)
    return const * gamma_a1

def expected_xxT(kappa: torch.Tensor, beta: torch.Tensor, Q_matrix: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    c_kk = delta_2_kappa(kappa, beta)
    c_beta = delta_beta(kappa, beta)

    lambda_1 = c_k / c
    lambda_2 = (c - c_kk + c_beta) / (2 * c)
    lambda_3 = (c - c_kk - c_beta) / (2 * c)

    lambdas = torch.stack([lambda_1, lambda_2, lambda_3], dim=-1)  # Shape: [N, 3]
    lambda_matrix = torch.diag_embed(lambdas)  # Shape: [N, 3, 3]

    Q_matrix_T = Q_matrix.transpose(-1, -2)  # Transpose the last two dimensions: [N, 3, 3]
    result = torch.matmul(Q_matrix, torch.matmul(lambda_matrix, Q_matrix_T))  # Shape: [N, 3, 3]

    return result

def beta_gamma_exxt_gamma(beta: torch.Tensor, gamma: torch.Tensor, ExxT: torch.Tensor) -> torch.Tensor:
    gamma_unsqueezed = gamma.unsqueeze(1)  # Shape: (N, 1, 3)
    intermediate_result = torch.bmm(gamma_unsqueezed, ExxT)  # Shape: (N, 1, 3)
    gamma_unsqueezed_2 = gamma.unsqueeze(2)  # Shape: (N, 3, 1)
    result = torch.bmm(intermediate_result, gamma_unsqueezed_2).squeeze()  # Shape: (N,)
    return beta * result  # Shape: (N,)

def compute_kld_terms(kappa_a: torch.Tensor, beta_a: torch.Tensor, gamma_a1: torch.Tensor, gamma_a2: torch.Tensor, gamma_a3: torch.Tensor,
                      kappa_b: torch.Tensor, beta_b: torch.Tensor, gamma_b1: torch.Tensor, gamma_b2: torch.Tensor, gamma_b3: torch.Tensor,
                      Ex_a: torch.Tensor, ExxT_a: torch.Tensor, c_a: torch.Tensor, c_b: torch.Tensor, c_ka: torch.Tensor) -> torch.Tensor:
    
    log_term = torch.log(c_b / c_a)  # Shape: (N,)
    kappa_a_term = kappa_a.view(-1, 1) * gamma_a1  # Shape: (N, 3)
    kappa_b_term = kappa_b.view(-1, 1) * gamma_b1  # Shape: (N, 3)
    diff_kappa_term = kappa_a_term - kappa_b_term  # Shape: (N, 3)
    import pdb
    pdb.set_trace()
    ex_a_term = torch.sum(diff_kappa_term * Ex_a, dim=1)  # Shape: (N,)

    beta_a_term_1 = beta_gamma_exxt_gamma(beta_a, gamma_a2, ExxT_a)  # Shape: (N,)
    beta_b_term_1 = beta_gamma_exxt_gamma(beta_b, gamma_b2, ExxT_a)  # Shape: (N,)
    beta_a_term_2 = beta_gamma_exxt_gamma(beta_a, gamma_a3, ExxT_a)  # Shape: (N,)
    beta_b_term_2 = beta_gamma_exxt_gamma(beta_b, gamma_b3, ExxT_a)  # Shape: (N,)

    kld = (
        log_term + ex_a_term + beta_a_term_1 - beta_b_term_1 - beta_a_term_2 + beta_b_term_2
    )  # Shape: (N,)

    return kld

def get_kld(kent_a: torch.Tensor, kent_b: torch.Tensor) -> torch.Tensor:
    kappa_a, beta_a, phi_a, psi_a, eta_a = kent_a[:, 0], kent_a[:, 1], kent_a[:, 2], kent_a[:, 3], kent_a[:, 4]
    Q_matrix_a = angles_to_Q(phi_a, psi_a, eta_a)

    kappa_b, beta_b, phi_b, psi_b, eta_b = kent_b[:, 0], kent_b[:, 1], kent_b[:, 2], kent_b[:, 3], kent_b[:, 4]
    Q_matrix_b = angles_to_Q(phi_b, psi_b, eta_b)

    gamma_a1, gamma_a2, gamma_a3 = Q_matrix_a[:, :, 0], Q_matrix_a[:, :, 1], Q_matrix_a[:, :, 2]
    gamma_b1, gamma_b2, gamma_b3 = Q_matrix_b[:, :, 0], Q_matrix_b[:, :, 1], Q_matrix_b[:, :, 2]

    c_a = c_approximation(kappa_a, beta_a)
    c_b = c_approximation(kappa_b, beta_b)
    c_ka = delta_kappa(kappa_a, beta_a)

    ExxT_a = expected_xxT(kappa_a, beta_a, Q_matrix_a, c_a, c_ka)
    Ex_a = expected_x(gamma_a1, c_a, c_ka)

    kld = compute_kld_terms(kappa_a, beta_a, gamma_a1, gamma_a2, gamma_a3,
                            kappa_b, beta_b, gamma_b1, gamma_b2, gamma_b3,
                            Ex_a, ExxT_a, c_a, c_b, c_ka)
    return kld

def kent_loss(kld: torch.Tensor, const: float = 2.0) -> torch.Tensor:
    return 1 - 1 / (const + torch.sqrt(kld))

def kent_iou(kent_a: torch.Tensor, kent_b: torch.Tensor) -> torch.Tensor:
    kld = get_kld(kent_a, kent_b)
    return 1 / (1 + torch.sqrt(kld))

if __name__ == "__main__":

    kent_a1 = [20.2, 4.1, 0, 0, 0] 
    kent_a2 = [10.1, 4.1, 0, 0, 0]
    kent_a3 = [10.1, 4.1, 0, 0, 0]
    
    kent_a = torch.tensor([kent_a1, kent_a2, kent_a3], dtype=torch.float32, requires_grad=True)

    kent_b1 = [10.2, 4.1, 0, 0, 0] 
    kent_b2 = [20.1, 4.1, 0, 0, 0]
    kent_b3 = [30.1, 4.1, 0, 0, 0]

    kent_b = torch.tensor([kent_b1, kent_b2, kent_b3], dtype=torch.float32, requires_grad=True)

    kld = get_kld(kent_a, kent_b)
    print(kld)
    
    #kld.sum().backward()

    #print("Gradients for kent_a:", kent_a.grad)
    #print("Gradients for kent_b:", kent_b.grad)

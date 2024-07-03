import torch
import pdb

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

def del_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    numerator = -2 * torch.pi * (4 * beta**2 + kappa - kappa**2) * torch.exp(kappa)
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2)
    return numerator / denominator

def del_2_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    numerator = 2 * torch.pi * (kappa**4 - 2 * kappa**3 + (2 - 8 * beta**2) * kappa**2 + 8 * beta**2 * kappa + 16 * beta**4 + 4 * beta**2) * torch.exp(kappa)
    denominator = (kappa - 2 * beta)**(5/2) * (kappa + 2 * beta)**(5/2)
    return numerator / denominator

def del_beta(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    numerator = 8 * torch.pi * torch.exp(kappa) * beta
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2)
    return numerator / denominator

#ONE FOR EACH KENT
def expected_x(gamma_a1: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    const = (c_k / c).view(-1, 1)
    return const * gamma_a1

def expected_xxT(kappa: torch.Tensor, beta: torch.Tensor, Q_matrix: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    c_kk = del_2_kappa(kappa, beta)
    c_beta = del_beta(kappa, beta)

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

def kld_matrix(kappa_a: torch.Tensor, beta_a: torch.Tensor, gamma_a1: torch.Tensor, gamma_a2: torch.Tensor, gamma_a3: torch.Tensor,
                      kappa_b: torch.Tensor, beta_b: torch.Tensor, gamma_b1: torch.Tensor, gamma_b2: torch.Tensor, gamma_b3: torch.Tensor,
                      Ex_a: torch.Tensor, ExxT_a: torch.Tensor, c_a: torch.Tensor, c_b: torch.Tensor, c_ka: torch.Tensor) -> torch.Tensor:
    
    # log(c_b / c_a) + (kappa_a * gamma_a1.T - kappa_b * gamma_b1.T) * Ex_a
    # + beta_a * gamma_a2.T * ExxT_a * gamma_a2 - beta_b * gamma_b2.T * ExxT_a * gamma_b2
    # - beta_a * gamma_a3.T * ExxT_a * gamma_a3 + beta_b * gamma_b3.T * ExxT_a * gamma_b3

    log_term = torch.log(c_b.view(-1, 1) / c_a.view(1, -1)) 
    
    kappa_a_gamma_a1 = kappa_a.view(-1, 1) * gamma_a1  # Shape: (N, 3)
    kappa_b_gamma_b1 = kappa_b.view(-1, 1) * gamma_b1  # Shape: (N, 3)

    kappa_a_gamma_a1_expanded = kappa_a_gamma_a1.unsqueeze(1)  # Shape: [n_a, 1, 3]
    kappa_b_gamma_b1_expanded = kappa_b_gamma_b1.unsqueeze(0)  # Shape: [1, n_b, 3]

    diff_kappa_term = kappa_a_gamma_a1_expanded - kappa_b_gamma_b1_expanded  # Shape: [n_a, n_b, 3]    
    Ex_a_expanded = Ex_a.unsqueeze(1)  # Shape: [n_a, 1, 3]
    Ex_a_expanded = Ex_a_expanded.expand(-1, diff_kappa_term.size(1), -1)  # Shape: [n_a, n_b, 3]
    ex_a_term = torch.sum(diff_kappa_term * Ex_a_expanded, dim=-1)  # Shape: [n_a, n_b]

    #ACREDITO QUE ESTEJA CORRETO ATE ESSE PONTO

    #ExxT_a is a 3x3 matrix for each kent

    # Compute the next term: beta_a * gamma_a2.T * ExxT_a * gamma_a2
    beta_a_gamma_a2 = beta_a.view(-1, 1) * gamma_a2  # Shape: (n_a, 3)
    beta_a_gamma_a2_expanded = beta_a_gamma_a2.unsqueeze(1)  # Shape: (n_a, 1, 3)
    intermediate_result_a2 = torch.bmm(beta_a_gamma_a2_expanded, ExxT_a)  # Shape: (n_a, 1, 3)
    beta_a_term_1 = torch.bmm(intermediate_result_a2, gamma_a2.unsqueeze(2)).squeeze()  # Shape: (n_a, 1)

    pdb.set_trace()

    # Expand beta_a_term_1 to match the shape [n_a, n_b]
    beta_a_term_1_expanded = beta_a_term_1.unsqueeze(1).expand(-1, beta_b.size(0))  # Shape: (n_a, n_b)
    #beta_a_term_1_expanded = beta_a_term_1.expand(-1, beta_b.size(0))  # Shape: (n_a, n_b)

    # Compute the term: -beta_b * gamma_b2.T * ExxT_a * gamma_b2
    beta_b_gamma_b2 = beta_b.view(-1, 1) * gamma_b2  # Shape: (n_b, 3)
    beta_b_gamma_b2_expanded = beta_b_gamma_b2.unsqueeze(0)  # Shape: (1, n_b, 3)
    intermediate_result_b2 = torch.bmm(beta_b_gamma_b2_expanded, ExxT_a.transpose(0, 1))  # Shape: (1, n_b, 3)
    beta_b_term_2 = torch.bmm(intermediate_result_b2, gamma_b2.unsqueeze(2)).squeeze()  # Shape: (1, n_b)

    # Expand beta_b_term_2 to match the shape [n_a, n_b]
    beta_b_term_2_expanded = beta_b_term_2.expand(kappa_a.size(0), -1)  # Shape: (n_a, n_b)

    # Compute the final term: beta_a_term_1_expanded - beta_b_term_2_expanded
    beta_term = beta_a_term_1_expanded - beta_b_term_2_expanded  # Shape: (n_a, n_b)

    # Repeat the process for gamma_a3 and gamma_b3
    # Compute the term: -beta_a * gamma_a3.T * ExxT_a * gamma_a3
    beta_a_gamma_a3 = beta_a.view(-1, 1) * gamma_a3  # Shape: (n_a, 3)
    beta_a_gamma_a3_expanded = beta_a_gamma_a3.unsqueeze(1)  # Shape: (n_a, 1, 3)
    intermediate_result_a3 = torch.bmm(beta_a_gamma_a3_expanded, ExxT_a)  # Shape: (n_a, 1, 3)
    beta_a_term_3 = torch.bmm(intermediate_result_a3, gamma_a3.unsqueeze(2)).squeeze()  # Shape: (n_a, 1)

    # Expand beta_a_term_3 to match the shape [n_a, n_b]
    beta_a_term_3_expanded = beta_a_term_3.expand(-1, beta_b.size(0))  # Shape: (n_a, n_b)

    # Compute the term: beta_b * gamma_b3.T * ExxT_a * gamma_b3
    beta_b_gamma_b3 = beta_b.view(-1, 1) * gamma_b3  # Shape: (n_b, 3)
    beta_b_gamma_b3_expanded = beta_b_gamma_b3.unsqueeze(0)  # Shape: (1, n_b, 3)
    intermediate_result_b3 = torch.bmm(beta_b_gamma_b3_expanded, ExxT_a.transpose(0, 1))  # Shape: (1, n_b, 3)
    beta_b_term_4 = torch.bmm(intermediate_result_b3, gamma_b3.unsqueeze(2)).squeeze()  # Shape: (1, n_b)

    # Expand beta_b_term_4 to match the shape [n_a, n_b]
    beta_b_term_4_expanded = beta_b_term_4.expand(kappa_a.size(0), -1)  # Shape: (n_a, n_b)


    kld = (
        log_term + ex_a_term + beta_term - beta_term_2
    )  # Shape: (n_a, n_b)

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
    c_ka = del_kappa(kappa_a, beta_a)

    ExxT_a = expected_xxT(kappa_a, beta_a, Q_matrix_a, c_a, c_ka)
    Ex_a = expected_x(gamma_a1, c_a, c_ka)

    kld = kld_matrix(kappa_a, beta_a, gamma_a1, gamma_a2, gamma_a3,
                            kappa_b, beta_b, gamma_b1, gamma_b2, gamma_b3,
                            Ex_a, ExxT_a, c_a, c_b, c_ka)
    return kld

def kent_loss(kld: torch.Tensor, const: float = 2.0) -> torch.Tensor:
    return 1 - 1 / (const + torch.sqrt(kld))

def kent_iou(kent_a: torch.Tensor, kent_b: torch.Tensor) -> torch.Tensor:
    kld = get_kld(kent_a, kent_b)
    return 1 / (1 + torch.sqrt(kld))

if __name__ == "__main__":

    kent_a1 = [20.2, 4.1, 20, 0, 0] 
    kent_a2 = [9.1, 4.1, 10, 0, 0]
    kent_a3 = [10.1, 4.1, 0, 0, 0]
    kent_a4 = [10.1, 4.1, 0, 0, 0]
    
    kent_a = torch.tensor([kent_a1, kent_a2, kent_a3, kent_a4], dtype=torch.float32, requires_grad=True)

    kent_b1 = [10.2, 4.1, 0, 0, 0] 
    kent_b2 = [20.1, 4.1, 0, 0, 0]
    kent_b3 = [30.1, 4.1, 0, 0, 0]

    kent_b = torch.tensor([kent_b1, kent_b2, kent_b3], dtype=torch.float32, requires_grad=True)

    kld = get_kld(kent_a, kent_b)
    print(kld)
    
    #kld.sum().backward()

    #print("Gradients for kent_a:", kent_a.grad)
    #print("Gradients for kent_b:", kent_b.grad)


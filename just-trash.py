import torch

# Example sizes
n_a = 4
n_b = 3

# Generate test data
c_a = torch.randn(n_a)
c_b = torch.randn(n_b)

# Original implementation (element-wise)
original_log_terms = []
for i in range(n_b):
    for j in range(n_a):
        original_log_terms.append(torch.log(c_b[i] / c_a[j]))
original_log_terms = torch.tensor(original_log_terms).view(n_b, n_a)

# New implementation (broadcasting)
new_log_terms = torch.log(c_b.view(-1, 1) / c_a.view(1, -1))  # Shape: (n_b, n_a)

# Compare results
print("Original Log Terms:\n", original_log_terms)
print("New Log Terms:\n", new_log_terms)

# Check if they are close
if torch.allclose(original_log_terms, new_log_terms, atol=1e-6):
    print("The new implementation is correct.")
else:
    print("The new implementation is incorrect.")
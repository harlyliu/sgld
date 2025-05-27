import torch
import torch.nn as nn
import torch.nn.functional as F
from v6.beta_prior import gp_eigen_value, gp_eigen_funcs_fast

class SpatialSTGPInputLayer(nn.Module):
    def __init__(self, num_units, grids, poly_degree=10, a=0.01, b=1.0, d=2,
                 nu=0.1, sigma_lambda_squared=1.0, device='cpu'):
        super().__init__()
        self.device = device
        self.num_units = num_units
        self.nu = nu
        self.sigma_lambda_squared = nn.Parameter(torch.tensor(sigma_lambda_squared, dtype=torch.float32, device=self.device))

        # Initialize grids and other parameters
        self.grids = torch.tensor(grids.copy(), dtype=torch.float32).to(device)
        self.poly_degree = poly_degree
        self.a = a
        self.b = b
        self.d = d

        eigenvalues_np = gp_eigen_value(poly_degree, a, b, d)
        self.K = len(eigenvalues_np)
        self.eigenvalues = torch.tensor(eigenvalues_np, dtype=torch.float32, device=device)  # shape (K,)

        if isinstance(grids, torch.Tensor):
            grids_np = grids.cpu().numpy()
        else:
            grids_np = grids.copy()

        eigenfuncs_np = gp_eigen_funcs_fast(grids_np, poly_degree, a, b, orth=True)
        self.eigenfuncs = torch.tensor(eigenfuncs_np, dtype=torch.float32, device=device)  # shape (K, V)

        self.eigenfuncs = self.eigenfuncs.T  # now (V, K)

        std_dev = torch.sqrt(self.sigma_lambda_squared * self.eigenvalues)
        self.C = nn.Parameter(torch.randn(num_units, self.K, device=device) * std_dev.unsqueeze(0))

        self.xi = nn.Parameter(torch.zeros(num_units, device=device))

        self.nu_tilde = self.nu / (torch.sqrt(self.sigma_lambda_squared) + 1e-8)

    def soft_threshold(self, x):
        return torch.sign(x) * torch.clamp(torch.abs(x) - self.nu_tilde, min=0.0)

    def forward(self, X):
        beta = self.C @ self.eigenfuncs.T  # (num_units, V)
        beta = self.soft_threshold(beta)

        weighted_sum = X @ beta.T  # (B, num_units)
        z = weighted_sum + self.xi.unsqueeze(0)  # (B, num_units)

        activated = F.relu(z)
        return activated


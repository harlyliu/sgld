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
        self.sigma_lambda_squared = sigma_lambda_squared

        # grids: tensor shape (V, d), spatial locations
        self.grids = torch.tensor(grids.copy(), dtype=torch.float32).to(device)

        # Load eigenvalues and eigenfunctions (precomputed)

        eigenvalues_np = gp_eigen_value(poly_degree, a, b, d)
        self.K = len(eigenvalues_np)
        self.eigenvalues = torch.tensor(eigenvalues_np, dtype=torch.float32, device=device)  # shape (K,)
        if isinstance(grids, torch.Tensor):
            grids_np = grids.cpu().numpy()
        else:
            grids_np = grids.copy()

        eigenfuncs_np = gp_eigen_funcs_fast(grids_np, poly_degree, a, b, orth=False)
        self.eigenfuncs = torch.tensor(eigenfuncs_np, dtype=torch.float32, device=device)  # shape (K, V)

        # transpose eigenfuncs to shape (V, K) to match math notation
        self.eigenfuncs = self.eigenfuncs.T  # now (V, K)

        # Initialize learnable coefficients C ~ N(0, sigma_lambda^2 * Lambda)
        std_dev = torch.sqrt(self.sigma_lambda_squared * self.eigenvalues)
        # shape (num_units, K), each row c_u is coeffs for unit u
        self.C = nn.Parameter(torch.randn(num_units, self.K, device=device) * std_dev.unsqueeze(0))

        # Bias per unit
        self.xi = nn.Parameter(torch.zeros(num_units, device=device))

        # Precompute threshold nu_tilde = nu / sigma_lambda
        sigma_lambda_tensor = torch.tensor(self.sigma_lambda_squared, dtype=torch.float32, device=self.device)
        self.nu_tilde = self.nu / (torch.sqrt(sigma_lambda_tensor) + 1e-8)

    def soft_threshold(self, x):
        return torch.sign(x) * torch.clamp(torch.abs(x) - self.nu_tilde, min=0.0)

    def forward(self, X):
        # X shape: (B, V), batch size B, V voxels (spatial locations)

        # Compute spatially varying weights beta_u(s_j)
        # self.C shape: (num_units, K), self.eigenfuncs.T shape: (K, V)
        # Resulting beta shape: (num_units, V)
        beta = self.C @ self.eigenfuncs.T  # (num_units, V)

        # Apply soft-thresholding on the spatially varying weights
        beta = self.soft_threshold(beta)
        #print(f"X.shape: {X.shape}")  # Should be (1000, in_feature)
        #print(f"self.eigenfuncs.shape: {self.eigenfuncs.shape}")  # Should match in_feature (i.e., (in_feature, K))
        # Now compute weighted sum z_i,u = xi_u + sum_j beta_u(s_j) * X_i(s_j)
        # X shape: (B, V), beta shape: (num_units, V)
        # Perform batch matrix multiplication (B, V) @ (V, num_units).T => (B, num_units)
        weighted_sum = X @ beta.T  # (B, num_units)

        # Add bias for each unit (broadcasted)
        z = weighted_sum + self.xi.unsqueeze(0)  # (B, num_units)

        # Apply activation (e.g., ReLU)
        activated = F.relu(z)
        return activated


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from SGLD_v6 import sample_inverse_gamma
from v6.beta_prior import gp_eigen_value, gp_eigen_funcs_fast


class SpatialSTGPInputLayer(nn.Module):
    def __init__(self, in_feature, num_of_units_in_top_layer_of_fully_connected_layers, grids, poly_degree=10, a=0.01, b=1.0, dimensions=2,
                 nu=0.1,a_theta=2.0,b_theta=1.0, a_lambda=2.0, b_lambda=1.0, device='cpu'):
        """
        :param num_of_units_in_top_layer_of_fully_connected_layers: number of neurons in this layer of the neural network
        :param grids: tensor that serves as a skeleton for the image, tensor of coordinates
        :param poly_degree: the degree to which the eigen functions and eigen values must be calculated
        :param a: left bound of spatial domain(used for eigen)
        :param b: right bound of spatial domain(used for eigen)
        :param dimensions: amount of dimensions in GP(used for eigen)
        :param nu: soft threshold(determines sparsity)
        :param device: the device the neural network is on
        """

        super().__init__()
        self.in_feature = in_feature
        self.device = device
        self.num_of_units_in_top_layer_of_fully_connected_layers = num_of_units_in_top_layer_of_fully_connected_layers
        print(f'__init__:: self.num_of_units_in_top_layer_of_fully_connected_layers={self.num_of_units_in_top_layer_of_fully_connected_layers}')
        self.nu = nu
        self.a_lambda = a_lambda
        self.b_lambda = b_lambda
        self.sigma_lambda_squared = sample_inverse_gamma(self.a_lambda, self.b_lambda, size=1)
        print(f'__init__:: sigma_lambda_squared={self.sigma_lambda_squared}')

        # Initialize grids and other parameters
        self.grids = torch.tensor(grids.copy(), dtype=torch.float32).to(device)
        self.poly_degree = poly_degree
        self.a = a
        self.b = b
        self.dimensions = dimensions

        eigenvalues_np = gp_eigen_value(poly_degree, a, b, dimensions)
        self.K = len(eigenvalues_np)
        self.eigenvalues = torch.tensor(eigenvalues_np, dtype=torch.float32, device=device)  # shape (K,)

        if isinstance(grids, torch.Tensor):
            grids_np = grids.cpu().numpy()
        else:
            grids_np = grids.copy()

        eigenfuncs_np = gp_eigen_funcs_fast(grids_np, poly_degree, a, b, orth=True)
        self.eigenfuncs = torch.tensor(eigenfuncs_np, dtype=torch.float32, device=device)  # shape (K, V)

        self.eigenfuncs = self.eigenfuncs.T  # now (V, K)

        self.Cu = None  # Cu in equation 34
        self.sample_Cu()

        # initialize self.theta
        self.beta = None
        self.beta = torch.matmul(self.Cu, self.eigenfuncs.T)


        # ksi is the bias term for the input layer in equation 31. ksi is the pronunciation of the greek letter
        # self.ksi is a vector, the size is num_of_units_in_top_layer_of_fully_connected_layers
        self.ksi = nn.Parameter(torch.zeros(num_of_units_in_top_layer_of_fully_connected_layers, device=device))
        self.initializeKsi(a_theta, b_theta)
        # self.ksi = torch.zeros(num_of_units_in_top_layer_of_fully_connected_layers, device=device) # comment out later
        #self.ksi = torch.zeros(num_of_units_in_top_layer_of_fully_connected_layers, device=device)  # delete later
        # note: nu is the thing that looks like v but isn't. nu is inversely related to variance(sigma_lambda squared
        # the larger the variance, the lower the threshold. when variance is small, higher threshold, greater sparsity
        # used to normalize thresholding relative to variance. + 1e-8 prevents division by 0 error.
        # This prevents division by zero or numerical instability if sigma_lambda_squared is very small.
        # function in the line above equation 33. nu~ =v/sigma_lambda
        self.nu_tilde = None
        self.set_nu_tilde()

    def initializeKsi(self, a_theta=2.0, b_theta=1.0):
        """
        Initialize ksi vector using an inverse-gamma prior, matching FC layer bias init.
        Draws sigma_theta_squared ~ InvGamma(a_theta, b_theta), then sets
        ksi.data ~ Normal(0, sigma_theta).
        """
        # --- Sample sigma_theta_squared ---
        sigma_theta_squared = sample_inverse_gamma(a_theta, b_theta)
        sigma_theta = math.sqrt(sigma_theta_squared)

        # --- Initialize ksi values ---
        self.ksi.data = torch.normal(
            mean=0.0,
            std=sigma_theta,
            size=self.ksi.size(),
            device=self.device
        )

    def sample_sigma_lambda_squared(self):
        with torch.no_grad():
            total_entries = self.Cu.numel()
            squared_norm = torch.sum(self.Cu ** 2)
            new_a_lambda = self.a_lambda + total_entries / 2
            new_b_lambda = self.b_lambda + squared_norm / 2
            self.sigma_lambda_squared = sample_inverse_gamma(new_a_lambda, new_b_lambda, size=1)
            # print(f'_sample_sigma_lambda_squared:: sigma_lambda_squared={self.sigma_lambda_squared}')
            return self.sigma_lambda_squared

    def set_nu_tilde(self):
        self.nu_tilde = torch.abs(self.nu / (torch.sqrt(self.sigma_lambda_squared) + 1e-8))
        """
        sigma_lambda_scalar = torch.sqrt(self.sigma_lambda_squared).item()
        # Safe bounding of nu_tilde to avoid trivial solutions
        self.nu_tilde = torch.tensor(
            min(1.0, max(1e-3, self.nu / (sigma_lambda_scalar + 1e-8))),
            device=self.device
        )
        """

    def sample_Cu(self):
        """
        Resample Cu ∼ N(0, σ²_lambda ⋅ Λ) after sigma_lambda_squared is updated.
        """
        std_dev = torch.sqrt(self.sigma_lambda_squared * self.eigenvalues)  # std_dev.shape = (self.K, )
        # print(f'sample_Cu:: self.sigma_lambda_squared={self.sigma_lambda_squared} std_dev={std_dev}')
        self.Cu = torch.randn(self.num_of_units_in_top_layer_of_fully_connected_layers, self.K, device=self.device) * std_dev  # Cu in equation 34
        # print(f'sample_Cu:: self.Cu={self.Cu}')
        self.beta = torch.matmul(self.Cu, self.eigenfuncs.T)

    def soft_threshold(self, x):

        magnitude = torch.abs(x) - self.nu_tilde
        thresholded = F.relu(magnitude)
        return thresholded * torch.sign(x)

    def forward(self, X):
        """

        :param X: represents the image, intensity of each voxel
        :return: the input for the fully connected hidden layers
        """
        print(f'SpatialSTGPInputLayer::forward: X.shape={X.shape} X[0]={X[0]} self.beta[0]={self.beta[0]}')
        # function 34

        # self.beta = torch.matmul(self.Cu, self.eigenfuncs.T)  # uncomment out later (num_of_units_in_top_layer_of_fully_connected_layers, number of voxels in an image)

        # print(f'forward::theta before threshold{self.theta}')
        # print(f'forward:: nu={self.nu} nu_tilde={self.nu_tilde} sigma_lambda_squared={self.sigma_lambda_squared} torch.sqrt(self.sigma_lambda_squared)={torch.sqrt(self.sigma_lambda_squared)}')

        # self.beta = self.soft_threshold(self.beta) # uncomment out later
        # print(f'forward:: X[0]={X[0]} {self.beta[0]} {torch.matmul(X[0], self.beta[2])}')
        # function 31
        # print(f'self.beta.shape{self.beta.shape}')
        # print(f'self.beta={self.beta}')
        z = torch.matmul(X, self.beta.T) + self.ksi  # (B, num_of_units_in_top_layer_of_fully_connected_layers)
        # print(f'forward::x.shape{X.shape}')
        # print(f'forward::self.theta after threshold{self.theta}')
        # print(f'forward:: z.shape={z.shape}')
        # print(f'z{z}')
        # exit()
        #print(f'forward:: z{z}')
        # activated = F.relu(z)
        return z



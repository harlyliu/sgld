import torch
import torch.nn as nn
from STGP_input_layer import SpatialSTGPInputLayer
from SGLD_v6 import sample_inverse_gamma
import math


class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features, a_beta, b_beta):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        sigma_beta_squared = sample_inverse_gamma(a_beta, b_beta, size=1)
        self._initialize_beta(sigma_beta_squared)

    def forward(self, x):
        return self.linear(x)

    def _initialize_beta(self, sigma_beta_squared, device='cpu'):
        sigma_beta = math.sqrt(sigma_beta_squared)
        self.linear.weight.data = torch.normal(mean=0, std=sigma_beta, size=self.linear.weight.size(), device=device,
                                               requires_grad=True)
        if self.linear.bias is not None:
            self.linear.bias.data = torch.normal(mean=0, std=sigma_beta, size=self.linear.bias.size(), device=device,
                                                 requires_grad=True)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_unit_list=(20, 30, 1), a_theta=2.0, b_theta=1.0):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_unit_list = hidden_unit_list

        layers = []
        current_input_size = input_size

        for i, units in enumerate(hidden_unit_list):
            linear_layer = nn.Linear(current_input_size, units)
            layers.append(linear_layer)

            if i < len(hidden_unit_list) - 1:
                layers.append(nn.ReLU())

            current_input_size = units

        self.sequential = nn.Sequential(*layers)

        sigma_theta_squared = sample_inverse_gamma(a_theta, b_theta, size=1)
        self._initialize(sigma_theta_squared)

    def _initialize(self, sigma_theta_squared):
        sigma_theta = math.sqrt(sigma_theta_squared)
        for m in self.sequential.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=sigma_theta)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data, mean=0, std=sigma_theta)

    def forward(self, x):
        return self.sequential(x)


# class STGPNeuralNetwork(nn.Module):
#     def __init__(self, grids, fully_connected_layers,
#                  poly_degree=10, a=0.01, b=1.0, dimensions=2,
#                  nu=0.1, a_theta=2.0, b_theta=1.0, a_lambda=2.0, b_lambda=1.0, device='cpu'):
#         """
#         Neural Network Class that implements Spatially Varying Weights
#         :param grids: tensor that holds the coordinates of each where input coordinates are defined
#         :param fully_connected_layers:how many units must be in each layer
#         :param poly_degree:how many degrees the eigen value should be calculated to
#         :param a:left endpoint of the spatial domain
#         :param b:right endpoint of the spatial domain
#         :param dimensions:determines the amount of dimensions in GP
#         :param nu:soft threshold
#         :param device:which device the neural network is ran on
#         """
#         super().__init__()
#         self.device = device
#
#         # Initialize SpatialSTGPInputLayer with sigma_lambda_squared and other parameters
#         self.input_layer = SpatialSTGPInputLayer(
#             num_of_units_in_top_layer_of_fully_connected_layers=fully_connected_layers[0],
#             grids=grids,
#             poly_degree=poly_degree,
#             a=a,
#             b=b,
#             dimensions=dimensions,
#             nu=nu,
#             a_lambda=a_lambda,
#             b_lambda=b_lambda,
#             device=device
#         )
#
#         layers = []
#         layers.append(self.input_layer)
#         input_size = fully_connected_layers[0]
#
#         for units in fully_connected_layers[1:]:
#             layers.append(nn.Linear(input_size, units))
#             layers.append(nn.ReLU())
#             input_size = units
#
#         layers.pop()  # Remove the final ReLU for the output layer
#
#         self.all_layers = nn.Sequential(*layers).to(device)
#
#         self._initializeSigmaThetaSquared(a_theta, b_theta)
#
#     def _initializeSigmaThetaSquared(self, a_theta=2.0, b_theta=1.0):
#         # --- Sample sigma_theta_squared ---
#         sigma_theta_squared = sample_inverse_gamma(a_theta, b_theta)
#         sigma_theta = math.sqrt(sigma_theta_squared)
#
#         # --- Initialize weights of all nn.Linear layers ---
#         for m in self.all_layers.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight.data, mean=0, std=sigma_theta)
#                 if m.bias is not None:
#                     nn.init.normal_(m.bias.data, mean=0, std=sigma_theta)
#         self.input_layer.ksi.data = torch.normal(mean=0, std=sigma_theta, size=self.input_layer.ksi.size(), device=self.device)
#
#     def forward(self, X):
#         # Forward pass through the layers
#         return self.all_layers(X)


class STGPNeuralNetworkDummy(nn.Module):
    def __init__(
            self,
            in_feature,
            grids,
            fully_connected_layers,
            poly_degree=10,
            a=0.01,
            b=1.0,
            dimensions=2,
            nu=0.1,
            a_theta=2.0,
            b_theta=1.0,
            a_lambda=2.0,
            b_lambda=1.0,
            device='cpu'
    ):
        """
        Combines a fixed STGP input transform with a standard FC network.
        """
        super().__init__()
        self.device = device
        self.input_layer = SpatialSTGPInputLayer(
            in_feature=in_feature,
            num_of_units_in_top_layer_of_fully_connected_layers=4,
            grids=grids,
            poly_degree=poly_degree,
            a=a,
            b=b,
            dimensions=dimensions,
            nu=nu,
            a_theta=a_theta,
            b_theta=b_theta,
            a_lambda=a_lambda,
            b_lambda=b_lambda,
            device=device
        )
        # Build the rest using NeuralNetwork
        self.fc = NeuralNetwork(
            input_size=4,
            hidden_unit_list=tuple(fully_connected_layers),
            a_theta=a_theta,
            b_theta=b_theta
        ).to(device)

    def forward(self, X):
        print(f'STGPNeuralNetworkDummy::forward: X.shape={X.shape}')
        # print(f'forward:: X={X}')
        z = self.input_layer(X)  # applies fixed theta & ksi, plus ReLU
        # print(f'forward:: z={z}')
        return self.fc(z)
        # return self.fc(X)


import torch
import torch.nn as nn
import STGP_input_layer
from SGLD_v5 import sample_inverse_gamma


class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features, a_beta, b_beta):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        sigma_beta_squared = sample_inverse_gamma(a_beta, b_beta, size=1).squeeze()
        self._initialize_beta(sigma_beta_squared)

    def forward(self, x):
        return self.linear(x)

    def _initialize_beta(self, sigma_beta_squared, device='cpu'):
        sigma_beta = torch.sqrt(sigma_beta_squared.clone().detach())
        self.linear.weight.data = torch.normal(mean=0, std=sigma_beta, size=self.linear.weight.size(), device=device,
                                               requires_grad=True)
        if self.linear.bias is not None:
            self.linear.bias.data = torch.normal(mean=0, std=sigma_beta, size=self.linear.bias.size(), device=device,
                                                 requires_grad=True)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_unit_list=[20, 30, 1], a_beta=2.0, b_beta=1.0):
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

        sigma_beta_squared = sample_inverse_gamma(a_beta, b_beta, size=1).squeeze()
        self._initialize(sigma_beta_squared)

    def _initialize(self, sigma_beta_squared):
        sigma_beta = torch.sqrt(sigma_beta_squared.clone().detach())
        for m in self.sequential.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=sigma_beta)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data, mean=0, std=sigma_beta)

    def forward(self, x):
        return self.sequential(x)


class STGPNeuralNetwork(nn.Module):
    def __init__(self, grids, hidden_unit_list,
                 poly_degree=10, a=0.01, b=1.0, d=2,
                 nu=0.1, sigma_lambda_squared=1.0, device='cpu'):
        super().__init__()
        self.device = device

        self.input_layer = STGP_input_layer.SpatialSTGPInputLayer(
            num_units=hidden_unit_list[0],
            grids=grids,
            poly_degree=poly_degree,
            a=a,
            b=b,
            d=d,
            nu=nu,
            sigma_lambda_squared=sigma_lambda_squared,
            device=device
        )

        layers = []
        layers.append(self.input_layer)
        input_size = hidden_unit_list[0]

        for units in hidden_unit_list[1:]:
            layers.append(nn.Linear(input_size, units))
            layers.append(nn.ReLU())
            input_size = units

        layers.pop()  # remove final ReLU for output layer

        self.all_layers = nn.Sequential(*layers).to(device)

    def forward(self, x):
        #x = self.input_layer(x)
        return self.all_layers(x)

import random

import numpy as np
import torch

from SGLD_v5 import SgldBayesianRegression as V5
from model import STGPNeuralNetwork, NeuralNetwork, LinearRegression
from utils import generate_linear_data
from GP_comp.GP import generate_grids, gp_eigen_funcs_fast, gp_eigen_value


# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def test_stgp(in_feature):
    # Step 1: Generate synthetic data (input_size = number of voxels)
    X, y, true_weights, true_bias = generate_linear_data(n=1000, in_features=in_feature, noise_std=1.0)
    print(f"Shape of X: {X.shape}")
    # Step 2: Generate spatial grid matching input features (voxels)
    # Example: 1D grid with `in_feature` points from -1 to 1
    grids = generate_grids(d=1, num_grids=in_feature, grids_lim=(-1, 1))

    # Step 3: Define STGP parameters
    poly_degree = 15
    a = 0.01
    b = 1.0
    d = 1
    nu = 0.1
    sigma_lambda_squared = 1.1

    # Step 4: Instantiate STGP Neural Network (eigen computations done internally)
    model = STGPNeuralNetwork(
        grids=grids,
        hidden_unit_list=[5, 1],
        poly_degree=poly_degree,
        a=a,
        b=b,
        d=d,
        nu=nu,
        sigma_lambda_squared=sigma_lambda_squared,
        device='cpu'
    )

    # Step 5: Train using SGLD
    c = V5(
        a=2.0,
        b=1.0,
        a_beta=2.0,
        b_beta=1.0,
        step_size=0.0001,
        num_epochs=300,
        burn_in_epochs=100,
        batch_size=100,
        device='cpu',
        model=model
    )
    c.train(X, y)

    # Step 6: Make a prediction for a test input (all ones)
    inputs = torch.ones(in_feature)
    print(f"true_weight={true_weights} true_bias={true_bias}")
    print(f"X={inputs} Y(predicted)={c.predict(inputs)} Y(expected)={sum(true_weights) + true_bias}")
    print(f"X={inputs} Y(predicted)={c.predict(inputs, method='param_avg')} Y(expected)={sum(true_weights) + true_bias}")

def test_neural(in_feature):
    # Generate synthetic data
    X, y, true_weights, true_bias = generate_linear_data(n=1000, in_features=in_feature, noise_std=1.0)

    a_beta = 2.0
    b_beta = 1.0

    # Run SGLD
    c = V5(
        a=2.0,
        b=1.0,
        a_beta=a_beta,
        b_beta=b_beta,
        step_size=0.0001,
        num_epochs=300,
        burn_in_epochs=100,
        batch_size=100,
        device='cpu',
        model=NeuralNetwork(input_size=in_feature, hidden_unit_list=[5,1], a_beta=a_beta, b_beta=b_beta)
    )
    c.train(X, y)
    inputs = torch.ones(in_feature)
    print(f'true_weight={true_weights} true_bias={true_bias}')
    print(f'X={inputs} Y(predicted)={c.predict(inputs)} Y(expected)={sum(true_weights) + true_bias}')
    print(f'X={inputs} Y(predicted)={c.predict(inputs,method="param_avg")} Y(expected)={sum(true_weights) + true_bias}')

def test_linear(in_feature):
    # Generate synthetic data
    X, y, true_weights, true_bias = generate_linear_data(n=1000, in_features=in_feature, noise_std=1.0)

    a_beta = 2.0
    b_beta = 1.0
    out_feature = 1

    # Run SGLD
    c = V5(
        a=2.0,
        b=1.0,
        a_beta=a_beta,
        b_beta=b_beta,
        model=LinearRegression(in_features=in_feature, out_features=out_feature, a_beta=a_beta, b_beta=b_beta),
        step_size=0.001,
        num_epochs=150,
        burn_in_epochs=100,
        batch_size=100,
        device='cpu',
    )
    c.train(X, y)

    inputs = torch.ones(in_feature)
    print(f'true_weight={true_weights} true_bias={true_bias}')
    print(f'X={inputs} Y(predicted)={c.predict(inputs)} Y(expected)={sum(true_weights) + true_bias}')
    print(f'X={inputs} Y(predicted)={c.predict(inputs,method="param_avg")} Y(expected)={sum(true_weights) + true_bias}')


# Generate data and run SGLD
if __name__ == "__main__":
    #test_neural(3)
    #test_linear(3)
    test_stgp(3)

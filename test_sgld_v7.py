import random

import numpy as np
import torch

from SGLD_v7 import SgldBayesianRegression as V7
from model import STGPNeuralNetwork
from utils import generate_linear_data
from GP_comp.GP import generate_grids


# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def test_stgp(in_feature):

    # Step 1: Define parameters
    fully_connected_layers = [in_feature, 10, 1]
    poly_degree = 22
    a = 0.01
    b = 1.0
    dimensions = 1
    nu = 0.001
    a_theta = 2.0
    b_theta = 1.0
    a_lambda = 2.0
    b_lambda = 1.0
    device = 'cpu'

    # Step 2: Generate synthetic data
    if dimensions == 1:
        X, y, true_weights, true_bias = generate_linear_data(n=1000, in_features=in_feature, noise_std=1.0)
        print(f"Shape of X: {X.shape}")

        # Step 3: Generate spatial grid
        grids = generate_grids(dimensions=dimensions, num_grids=in_feature, grids_lim=(-1, 1))
    else:
        exit("Need to replace testing data if dimension > 1")

    # Step 4: Build model
    model = STGPNeuralNetwork(
        in_feature=in_feature,
        grids=grids,
        fully_connected_layers=fully_connected_layers,
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

    # Step 5: Train with SGLD
    trainer = V7(
        a=a,
        b=b,
        a_theta=a_theta,
        b_theta=b_theta,
        step_size=0.0001,
        num_epochs=300,  # fix this back to 300 later
        burn_in_epochs=100,
        batch_size=100,
        device=device,
        model=model
    )
    trainer.train(X, y)

    # Step 6: Test prediction
    inputs = torch.ones(in_feature)
    expected_y = sum(true_weights) + true_bias

    print(f"true_weight={true_weights} true_bias={true_bias}")
    print(f"X={inputs} Y(predicted)={trainer.predict(inputs)} Y(expected)={expected_y}")
    # print(f"X={inputs} Y(predicted)={trainer.predict(inputs, method='param_avg')} Y(expected)={expected_y}")


# Generate data and run SGLD
if __name__ == "__main__":
    test_stgp(3)

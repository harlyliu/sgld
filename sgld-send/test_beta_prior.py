import numpy as np

from GP_comp.GP import generate_grids
from v6.beta_prior import sample_from_normal_distribution, SVPrior
import torch
import random

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def test_sample_from_normal_distribution():
    # Example usage:
    variance_np_scalar = np.array(0.5)
    samples_scalar = sample_from_normal_distribution(variance_np_scalar)
    print(f"Samples (scalar variance): {samples_scalar}")

    variance_np_vector = np.array([0.1, 1.0, 2.5])
    samples_vector = sample_from_normal_distribution(variance_np_vector)
    print(f"Samples (vector variance): {samples_vector}")

    variance_np_matrix = np.array([[0.2, 0.8], [1.5, 3.0]])
    samples_matrix = sample_from_normal_distribution(variance_np_matrix)
    print(f"Samples (matrix variance):\n{samples_matrix}")


def test_create_c_u():
    s = SVPrior()
    ans = s._create_c_u()
    print(ans)


def test_SVPrior():
    s = SVPrior()
    grids = generate_grids(d=2, num_grids=3, grids_lim=(-1, 1), random=False)
    ans = s.create_beta(grids)
    print(ans.shape)


if __name__ == '__main__':
    # test_sample_from_normal_distribution()
    # test_create_c_u()
    test_SVPrior()

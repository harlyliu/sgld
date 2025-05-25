import torch
import numpy as np

from GP_comp.GP import gp_eigen_value, gp_eigen_funcs_fast


def sample_from_normal_distribution(eigen_values: np.ndarray, sigma_lambda_squared=1.0) -> torch.Tensor:
    """
    Samples from a normal distribution with a mean of zero and a specified variance.

    Args:
        eigen_values: A NumPy ndarray. Function 34.
        sigma_lambda_squared: Function 34.

    Returns:
        A PyTorch tensor containing the samples from the normal distribution(s).

    """
    eigen_values_tensor = torch.from_numpy(eigen_values).float()
    mean = torch.zeros_like(eigen_values_tensor)
    samples = torch.normal(mean, eigen_values_tensor * sigma_lambda_squared)
    return samples


class SVPrior:
    def __init__(self, poly_degree=10, a=1, b=1, d=2, sigma_lambda_squared=1.0):
        self.poly_degree = poly_degree
        self.a = a
        self.b = b
        self.d = d
        self.sigma_lambda_squared = sigma_lambda_squared

    def _create_c_u(self):
        """function 34"""
        eigen_values = gp_eigen_value(self.poly_degree, self.a, self.b, self.d)
        ans = sample_from_normal_distribution(eigen_values, self.sigma_lambda_squared)
        # print(f'_create_c_u: ans.shape={ans.shape} type(ans)={type(ans)}')
        # print(ans)
        return ans

    def _create_eigen_func(self, grids):
        """function 34"""
        ans = gp_eigen_funcs_fast(grids, self.poly_degree, self.a, self.b, orth=False)
        ans = torch.from_numpy(ans).float()
        # print(f'_create_eigen_func: ans.shape={ans.shape} type(ans)={type(ans)}')
        # print(ans)
        return ans

    def create_beta(self, grids):
        ans = self._create_c_u().t()@self._create_eigen_func(grids)
        # print(ans)
        return ans

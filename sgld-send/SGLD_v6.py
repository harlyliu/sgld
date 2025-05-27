import torch
from torch.distributions import Gamma
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def sample_inverse_gamma(shape_param, rate_param, size=1):
    """
    Sample from an Inverse-Gamma distribution.

    Args:
        shape_param (float or torch.Tensor): Shape parameter α (must be positive).
        rate_param (float or torch.Tensor): Scale parameter β (must be positive).
        size (int): Number of samples or shape of the output tensor.
    Returns:
        torch.Tensor: Samples from the Inverse-Gamma distribution, shape determined by size.
    """
    gamma_dist = Gamma(shape_param, rate_param)
    gamma_samples = gamma_dist.sample(torch.Size((size,)))
    inverse_gamma_samples = 1.0 / gamma_samples
    return inverse_gamma_samples


class SgldBayesianRegression:
    """
    Stochastic Gradient Langevin Dynamics (SGLD) for Bayesian Linear Regression.
    """

    def __init__(
            self,
            a: float,  # shape of inverse gamma prior of sigma^2
            b: float,  # rate of inverse gamma prior of sigma^2
            a_beta: float,  # shape of inverse gamma prior of sigma^2_beta
            b_beta: float,  # rate of inverse gamma prior of sigma^2_beta
            model,
            step_size: float,
            num_epochs: int,
            burn_in_epochs: int,
            batch_size: int,  # number of samples in a single batch
            device: str
    ):
        self.device = device
        self.a = torch.tensor(a, dtype=torch.float32, device=self.device)
        self.b = torch.tensor(b, dtype=torch.float32, device=self.device)
        self.a_beta = torch.tensor(a_beta, dtype=torch.float32, device=self.device)
        self.b_beta = torch.tensor(b_beta, dtype=torch.float32, device=self.device)
        self.step_size = step_size
        self.num_epochs = num_epochs
        self.burn_in_epochs = burn_in_epochs
        self.batch_size = batch_size
        self.model = model

        self.samples = {'params': [], 'sigma': [], 'sigma_beta': [], 'residue':[]}

    def train(self, X_train, y_train):
        X = X_train.to(self.device)
        y = y_train.to(self.device)
        n = X.shape[0]

        # Initial sampling for sigma^2
        sigma_squared = self._sample_sigma_squared(X, y)
        print(f'Initial sigma^2: {sigma_squared}')

        # Initial sample for sigma_beta^2
        sigma_beta_squared = self._sample_sigma_beta_squared()

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")

            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                total_grad = self._calculate_total_grad(X_batch, y_batch, sigma_squared, sigma_beta_squared, n)

                with torch.no_grad():
                    param_list = [p for p in self.model.parameters()]
                    for i, param in enumerate(param_list):
                        start_idx = sum(p.numel() for p in param_list[:i])
                        end_idx = start_idx + param.numel()

                        grad_slice = total_grad[start_idx:end_idx].reshape(param.shape)

                        param_noise = torch.randn_like(param) * torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device))

                        param.add_(self.step_size * grad_slice + param_noise)

                # Sample variances per batch
                sigma_squared = self._sample_sigma_squared(X_batch, y_batch)
                sigma_beta_squared = self._sample_sigma_beta_squared()

                params_flat = torch.cat([p.detach().flatten() for p in self.model.parameters()]).cpu().numpy()
                if epoch >= self.burn_in_epochs:
                    self.samples['params'].append(params_flat)
                    self.samples['sigma'].append(sigma_squared.item())
                    self.samples['sigma_beta'].append(sigma_beta_squared.item())

    def _calculate_total_grad(self, X_batch, y_batch, sigma_squared, sigma_beta_squared, n):
        """
        Compute the total gradient combining the log-likelihood and log-prior gradients.
        """
        # Compute the likelihood gradient using the original method
        likelihood_grad = self._get_gradient_of_log_prob_likelihood(X_batch, y_batch, sigma_squared)

        # Scale the likelihood gradient based on the batch size
        likelihood_grad_scaled = (n / self.batch_size) * likelihood_grad

        # Compute the prior gradient using the original method
        prior_grad = self._get_gradient_of_log_prob_prior(sigma_beta_squared)

        # Combine the gradients
        total_grad = likelihood_grad_scaled + prior_grad
        return total_grad

    def _log_prob_of_prior(self, beta, sigma_beta_squared):
        """
        Compute the log-probability of the prior for model parameters.
        """
        # Ensure beta is a 2D tensor (even if it's a 1D tensor)
        if len(beta.shape) == 1:
            beta = beta.unsqueeze(0)  # Add a batch dimension, making it (1, N)

        p = beta.shape[1]  # Now it's guaranteed to have shape (B, N)
        sigma_beta_squared = sigma_beta_squared.to(self.device).clone().detach()
        beta_squared_sum = torch.sum(beta ** 2, dim=1)
        log_norm = -(p / 2) * torch.log(2 * torch.pi * sigma_beta_squared)
        quadratic_term = -(1 / (2 * sigma_beta_squared)) * beta_squared_sum
        return log_norm + quadratic_term


    def _sample_sigma_squared(self, X, y):
        """
        Sample σ^2 from an Inverse-Gamma distribution using equation (20).
        """
        with torch.no_grad():
            n = y.shape[0]

            # Compute predictions using the model
            y_pred = self.model(X)
            residuals = y - y_pred  # Shape: (n, 1)
            residual_squared_sum = torch.sum(residuals**2) / 2
            self.samples['residue'].append(residual_squared_sum)

            new_a = self.a + n / 2
            new_b = self.b + residual_squared_sum

            # Handle potential NaNs in the parameters
            if torch.isnan(new_a) or torch.isnan(new_b):
                if torch.isnan(new_a):
                    new_a = torch.tensor(1e-5, dtype=torch.float32, device=self.device)  # Default small value
                if torch.isnan(new_b):
                    new_b = torch.tensor(1e-5, dtype=torch.float32, device=self.device)  # Default small value

            # Sample sigma_squared from Inverse-Gamma
            sigma_squared = sample_inverse_gamma(new_a, new_b, size=1).squeeze()

            return sigma_squared

    def _sample_sigma_beta_squared(self):
        """
        Sample σβ^2 from an Inverse-Gamma distribution.
        """
        with torch.no_grad():
            p = sum(p.numel() for p in self.model.parameters())
            beta = torch.cat([p.flatten() for p in self.model.parameters()]).to(self.device)

            beta_squared_sum = torch.sum(beta**2) / 2

            new_a_beta = self.a_beta + p / 2
            new_b_beta = self.b_beta + beta_squared_sum

            # Handle potential NaNs in the parameters
            if torch.isnan(new_a_beta) or torch.isnan(new_b_beta):
                if torch.isnan(new_a_beta):
                    new_a_beta = torch.tensor(1e-5, dtype=torch.float32, device=self.device)  # Default small value
                if torch.isnan(new_b_beta):
                    new_b_beta = torch.tensor(1e-5, dtype=torch.float32, device=self.device)  # Default small value

            sigma_beta_squared = sample_inverse_gamma(new_a_beta, new_b_beta, size=1).squeeze()

            return sigma_beta_squared

    def predict(self, X, method="sample_avg"):
        with torch.no_grad():
            param_list = [p for p in self.model.parameters()]
            # For each sample, make new prediction and then average predictions (Monte Carlo)
            if method == "sample_avg":
                all_predictions = []
                for sample_params in self.samples['params']:
                    start_idx = 0
                    for i, param in enumerate(param_list):
                        end_idx = start_idx + param.numel()
                        param_slice = torch.tensor(sample_params[start_idx:end_idx], device=X.device).reshape(param.shape)
                        param.set_(param_slice)
                        start_idx = end_idx

                    prediction = self.model(X)
                    all_predictions.append(prediction.cpu().numpy())

                mean_prediction = np.mean(all_predictions, axis=0)
                variance_prediction = np.std(all_predictions, axis=0)
                return torch.tensor(mean_prediction, device=X.device)

            # Get the all parameters for all samples, take average, and then make prediction
            elif method == "param_avg":
                param_accumulator = [torch.zeros_like(p, device=X.device) for p in param_list]

                for sample_params in self.samples['params']:
                    start_idx = 0
                    for i, param in enumerate(param_list):
                        end_idx = start_idx + param.numel()
                        param_slice = torch.tensor(sample_params[start_idx:end_idx], device=X.device).reshape(param.shape)
                        param_accumulator[i] += param_slice
                        start_idx = end_idx

                num_samples = len(self.samples['params'])
                for i, param in enumerate(param_list):
                    param.set_(param_accumulator[i] / num_samples)

                prediction = self.model(X)
                return prediction
    def _get_gradient_of_log_prob_prior(self, sigma_beta_squared):
        """
        Compute the gradient of the log-prior with respect to model parameters.

        Args:
            sigma_beta_squared (float or torch.Tensor): Standard deviation squared of the prior for β.

        Returns:
            torch.Tensor: Gradient of the log-prior with respect to model parameters.
        """
        params = torch.cat([p.flatten() for p in self.model.parameters()])
        params = params.clone().detach().requires_grad_(True).to(self.device)

        # print(f'get_model_prior_gradient: params={params} sigma_beta={sigma_beta}')
        log_prior = self._log_prob_of_prior(params, sigma_beta_squared)
        log_prior.backward()

        total_grad = params.grad.clone()
        params.grad.zero_()

        return total_grad
    def _get_gradient_of_log_prob_likelihood(self, X, y, sigma_squared):
        """
        Compute the gradient of the log-likelihood with respect to model parameters.

        Args:
            X (torch.Tensor): Design matrix of shape (batch_size, p).
            y (torch.Tensor): Target vector of shape (batch_size,) or (batch_size, 1).
            sigma_squared (float or torch.Tensor): The noise variance parameter (sigma^2).

        Returns:
            torch.Tensor: Gradient of the log-likelihood with respect to model parameters.
        """
        # Zero the gradients before the backward pass
        self.model.zero_grad()

        for param in self.model.parameters():
            param.requires_grad_(True)

        log_likelihood = self._log_prob_of_likelihood(y, X, sigma_squared)
        log_likelihood.backward(retain_graph=True)  # Retain graph after first backward pass

        total_grad = torch.cat([p.grad.flatten() for p in self.model.parameters()])

        # Zero the gradients after extracting total_grad
        for param in self.model.parameters():
            param.grad.zero_()

        return total_grad


    def _log_prob_of_likelihood(self, y, X, sigma_square):
        """
        Compute the log-likelihood of y under a Gaussian likelihood model p(y; model, σ^2, X).

        Args:
            y (torch.Tensor): Target vector of shape (n,) or (n, 1).
            X (torch.Tensor): Design matrix of shape (n, p) containing predictors.
            sigma_square (float or torch.Tensor): Standard deviation squared of the Gaussian noise (default=1.0).
        Returns:
            torch.Tensor: torch.Size([]), a real number.
        """
        sigma_square = sigma_square.to(self.device)
        y = y.to(self.device)
        y_pred = self.model(X)
        residuals = y - y_pred
        # print('_log_prob_of_likelihood::X', X.shape)
        # print('_log_prob_of_likelihood::y_pred', y_pred.shape)
        # print('_log_prob_of_likelihood::residuals', residuals.shape)

        n = y.shape[0]
        residual_squared_sum = torch.sum(residuals**2)

        # The following implementation has to use torch.log because the value is extremely small when n is big
        # and the calculation without log is very unstable
        log_norm = -(n / 2) * torch.log(2 * torch.pi * sigma_square)
        quadratic_term = -(1 / (2 * sigma_square)) * residual_squared_sum
        log_likelihood = log_norm + quadratic_term
        # print(f'log_likelihood={log_likelihood} log_norm={log_norm} quadratic_term={quadratic_term}')
        # print('_log_prob_of_likelihood::residual_squared_sum', residual_squared_sum.shape)
        # print('_log_prob_of_likelihood::quadratic_term', quadratic_term.shape)
        # print(f'_log_prob_of_likelihood::log_likelihood={log_likelihood}', log_likelihood.shape)
        return log_likelihood

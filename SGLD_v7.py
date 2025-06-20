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
            a_theta: float,  # shape of inverse gamma prior of sigma^2_beta
            b_theta: float,  # rate of inverse gamma prior of sigma^2_beta
            model,
            step_size: float,
            num_epochs: int,
            burn_in_epochs: int,
            batch_size: int,  # number of samples in a single batch
            device: str
    ):
        """
        Initializes the SgldBayesianRegression model.

        Args:
            a: Shape parameter for the inverse gamma prior of sigma^2.
            b: Rate parameter for the inverse gamma prior of sigma^2.
            a_theta: Shape parameter for the inverse gamma prior of sigma^2_theta.
            b_theta: Rate parameter for the inverse gamma prior of sigma^2_theta.
            step_size: Step size for the SGLD algorithm.
            num_epochs: Total number of epochs.
            burn_in_epochs: Number of burn-in epochs.
            batch_size: Number of samples in each batch.
            device: Device to use for computation (e.g., 'cpu', 'cuda').
        """
        self.device = device
        self.a = torch.tensor(a, dtype=torch.float32, device=self.device)
        self.b = torch.tensor(b, dtype=torch.float32, device=self.device)
        self.a_theta = torch.tensor(a_theta, dtype=torch.float32, device=self.device)
        self.b_theta = torch.tensor(b_theta, dtype=torch.float32, device=self.device)
        self.step_size = step_size
        self.num_epochs = num_epochs
        self.burn_in_epochs = burn_in_epochs
        self.batch_size = batch_size
        self.model = model

        self.samples = {'params': [], 'sigma': [], 'sigma_theta': [], 'sigma_lambda': [], 'nu_tilde': [], 'beta': [],
                        'residue': []}

    def train(self, X_train, y_train):
        X = X_train.to(self.device)
        y = y_train.to(self.device)
        n = X.shape[0]
        sigma_squared = self._sample_sigma_squared(X, y)
        print(f'initial sigma^2={sigma_squared}')
        sigma_theta_squared = self._sample_sigma_theta_squared()

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            # print(f'epoch={epoch}')
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")

            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                # print(f'train:: batch_idx={batch_idx} X_batch.shape={X_batch.shape}, y_batch.shape={y_batch.shape}')

                # total_grad is 1D vector, concatenated by all parameters in the model, which is a common practice
                total_grad = self._calculate_total_grad(X_batch, y_batch, sigma_squared, sigma_theta_squared, n, self.model.input_layer.beta, self.model.input_layer.sigma_lambda_squared)

                with torch.no_grad():
                    # for name, param in self.model.named_parameters():
                    #     print(f"Layer: {name}")
                    #     print(f"Shape: {param.shape}")
                    #     print(f"Values:\n{param.data}")
                    #     print("-" * 30)
                    param_list = [p for p in self.model.parameters()]
                    # print(param_list)
                    # exit()
                    for i, param in enumerate(param_list):
                        # print(f'train:: i={i} param={param}')
                        start_idx = sum(p.numel() for p in param_list[:i])
                        end_idx = start_idx + param.numel()

                        # Extract the corresponding gradient slice and reshape it to match param's shape
                        grad_slice = total_grad[start_idx:end_idx].reshape(param.shape)

                        # Generate noise with the same shape as param
                        param_noise = torch.randn_like(param) * torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device))

                        # Update param in-place
                        param.add_(self.step_size * grad_slice + param_noise)
                # Sample variances per batch
                sigma_squared = self._sample_sigma_squared(X_batch, y_batch)
                sigma_theta_squared = self._sample_sigma_theta_squared()
                sigma_lambda_squared = self.model.input_layer.sample_sigma_lambda_squared()

                params_flat = torch.cat([p.detach().flatten() for p in self.model.parameters()]).cpu().numpy()
                if epoch >= self.burn_in_epochs:
                    self.samples['params'].append(params_flat)
                    self.samples['sigma'].append(sigma_squared.item())
                    self.samples['sigma_theta'].append(sigma_theta_squared.item())
                    self.samples['sigma_lambda'].append(sigma_lambda_squared)
                    # self.samples['nu_tilde'].append(self.model.input_layer.nu_tilde)
                    # self.samples['beta'].append(self.model.input_layer.beta)

    def _log_prob_of_prior(self, theta, sigma_theta_squared, beta, sigma_lambda_squared):
        """
        Compute the probability of theta under a Gaussian prior N(0, sigma_beta^2 I).
        """
        # print(f'theta.shap={theta.shape}')
        if len(theta.shape) == 1:
            theta = theta.unsqueeze(0)  # torch.Size([2]) --> torch.Size([1, 2])
        # print(f'theta.shap={theta.shape}')
        # print(f'_log_prob_of_prior::type(sigma_lambda_squared)={type(sigma_lambda_squared)} {sigma_lambda_squared.shape} type(sigma_theta_squared)={type(sigma_theta_squared)} {sigma_theta_squared.shape}')
        p = theta.shape[1]
        sigma_lambda_squared = sigma_lambda_squared.squeeze()
        sigma_theta_squared = sigma_theta_squared.to(self.device).clone().detach()
        sigma_lambda_squared = sigma_lambda_squared.to(self.device).clone().detach()
        theta_squared_sum = torch.sum(theta ** 2, dim=1)
        # print(f'theta {theta.shape}{theta.size()}')
        beta = beta.clone().detach().flatten().unsqueeze(0)
        # print(f'after beta {beta.shape}{beta.size()}')
        lambda_squared_sum = torch.sum(beta ** 2, dim=1)

        log_norm_theta = -(p / 2) * torch.log(2 * torch.pi * sigma_theta_squared)
        # print(f'_log_prob_of_prior::type(beta.numel())={type(beta.numel())} type(p)={type(p)}')
        log_norm_lambda = -(beta.numel() / 2) * torch.log(2 * torch.pi * sigma_lambda_squared)
        # print(f'_log_prob_of_prior::log_norm_lambda={log_norm_lambda} log_norm_theta={log_norm_theta }')
        quadratic_term_theta = -(1 / (2 * sigma_theta_squared)) * theta_squared_sum
        quadratic_term_lambda = -(1 / (2 * sigma_lambda_squared)) * lambda_squared_sum
        # print(f'_log_prob_of_prior::quadratic_term_theta={quadratic_term_theta.shape}{quadratic_term_theta.size()} quadratic_term_lambda={quadratic_term_lambda.shape}{quadratic_term_lambda.size()}')
        return log_norm_theta + log_norm_lambda + quadratic_term_theta + quadratic_term_lambda

    def _calculate_total_grad(self, X_batch, y_batch, sigma_squared, sigma_theta_squared, n, beta, sigma_lambda_squared):
        likelihood_grad = self._get_gradient_of_log_prob_likelihood(X_batch, y_batch, sigma_squared)
        likelihood_grad_scaled = (n / self.batch_size) * likelihood_grad
        prior_grad = self._get_gradient_of_log_prob_prior(sigma_theta_squared, beta, sigma_lambda_squared)
        total_grad = likelihood_grad_scaled + prior_grad
        return total_grad

    def _sample_sigma_squared(self, X, y):
        """
        Sample σ^2 from an Inverse-Gamma distribution using equation (20).

        Args:
            X (torch.Tensor): Design matrix of shape (n, p) containing predictors.
            y (torch.Tensor): Target vector of shape (n,) or (n, 1).
        Returns:
            torch.Tensor: Sampled σ^2 from the Inverse-Gamma distribution.
        """
        with torch.no_grad():
            # Number of observations (n)
            n = y.shape[0]

            # Compute predictions (y_pred) using the model
            y_pred = self.model(X)
            residuals = y - y_pred  # Shape: (n, 1)
            residual_squared_sum = torch.sum(residuals**2) / 2  # (y - Xβ)^T (y - Xβ) / 2
            self.samples['residue'].append(residual_squared_sum)

            # New shape and scale parameters for σ^2 (equation 20)
            new_a = self.a + n / 2
            new_b = self.b + residual_squared_sum

            # Sample σ^2 from Inverse-Gamma(new_shape_sigma, new_scale_sigma)
            sigma_squared = sample_inverse_gamma(new_a, new_b, size=1).squeeze()
            # print(f"Batch residual sum of squares / 2: {residual_squared_sum.item()}")
            # print(f"shape: {new_shape_sigma.item()}, scale: {new_scale_sigma.item()} a={a} n={n}")
            # print(f"Sampled σ^2: {sigma_squared.item()}")

            return sigma_squared

    def _sample_sigma_theta_squared(self):
        """
        Sample σβ^2 from an Inverse-Gamma distribution using equation (21), with data dependency via β^T β.
        Returns:
            torch.Tensor: Sampled σβ^2 from the Inverse-Gamma distribution.
        """
        with torch.no_grad():
            # Number of features (p) including bias
            p = sum(p.numel() for p in self.model.parameters())  # Total number of parameters (weights + bias)

            # Compute β^T β / 2 from current model parameters
            theta = torch.cat([p.flatten() for p in self.model.parameters()]).to(self.device)  # Flatten all parameters into a vector
            theta_squared_sum = torch.sum(theta**2) / 2  # β^T β / 2
            # print(f"β^T β / 2: {theta_squared_sum.item()}")

            # New shape and scale parameters for σβ^2 (equation 21)
            new_a_theta = self.a_theta + p / 2
            new_b_theta = self.b_theta + theta_squared_sum

            # Sample σβ^2 from Inverse-Gamma(new_shape_beta, new_scale_beta)
            sigma_theta_squared = sample_inverse_gamma(new_a_theta, new_b_theta, size=1).squeeze()

            # Optional: Print diagnostics for debugging
            # print(f"β^T β / 2: {theta_squared_sum.item()}")
            # print(f"Shape: {new_shape_beta.item()}, Scale: {new_scale_beta.item()}")
            # print(f"Sampled σβ^2: {sigma_theta_squared.item()}")

            return sigma_theta_squared

    def predict(self, X, method="sample_avg"):
        with torch.no_grad():
            param_list = [p for p in self.model.parameters()]
            # For each sample, make new prediction and then average predictions.(Monte Carlo)
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
                print(f'predict (sample_avg)::variance_prediction={variance_prediction}')
                return torch.tensor(mean_prediction, device=X.device)
            # get the all parameters for all samples, take average and then make prediction
            elif method == "param_avg":
                # Initialize accumulators
                param_accumulator = [torch.zeros_like(p, device=X.device) for p in param_list]

                # Sum all sampled parameters
                for sample_params in self.samples['params']:
                    start_idx = 0
                    for i, param in enumerate(param_list):
                        end_idx = start_idx + param.numel()
                        param_slice = torch.tensor(sample_params[start_idx:end_idx], device=X.device).reshape(param.shape)
                        param_accumulator[i] += param_slice
                        start_idx = end_idx

                # Average the parameters
                num_samples = len(self.samples['params'])
                for i, param in enumerate(param_list):
                    param.set_(param_accumulator[i] / num_samples)

                # Predict using the mean parameters
                prediction = self.model(X)
                print('predict (param_avg)::done')
                return prediction

    def _get_gradient_of_log_prob_likelihood(self, X, y, sigma_squared):
        """
        Compute the gradient of the log-likelihood with respect to model parameters.

        Args:
            X (torch.Tensor): Design matrix of shape (batch_size, p).
            y (torch.Tensor): Target vector of shape (batch_size,) or (batch_size, 1).
            sigma_squared (float or torch.Tensor): Standard deviation squared of the Gaussian noise.

        Returns:
            torch.Tensor: Gradient of the log-likelihood with respect to model parameters.
        """
        # Zero the gradients before the backward pass
        self.model.zero_grad()

        # for param in self.model.parameters():
        #   param.requires_grad_(True)
        log_likelihood = self._log_prob_of_likelihood(y, X, sigma_squared)
        log_likelihood.backward()
        total_grad = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.requires_grad])

        # Note: Zeroing gradients after extracting total_grad might be redundant
        # if this function is called once per optimization step. However, it's
        # generally good practice to ensure gradients are zeroed for the next step.
        # self.model.zero_grad()

        return total_grad

    def _get_gradient_of_log_prob_prior(self, sigma_theta_squared, beta, sigma_lambda_squared):
        """
        Compute the gradient of the log-prior with respect to model parameters.

        Args:
            sigma_theta_squared (float or torch.Tensor): Standard deviation squared of the prior for β.

        Returns:
            torch.Tensor: Gradient of the log-prior with respect to model parameters.
        """
        params = torch.cat([p.flatten() for p in self.model.parameters()])
        params = params.clone().detach().requires_grad_(True).to(self.device)

        # print(f'get_model_prior_gradient: params={params} sigma_beta={sigma_beta}')
        log_prior = self._log_prob_of_prior(params, sigma_theta_squared, beta, sigma_lambda_squared)
        # print(f'get_model_prior_gradient:log_prior{log_prior}')
        log_prior.backward()

        total_grad = params.grad.clone()
        params.grad.zero_()

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

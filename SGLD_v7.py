import torch
from torch.distributions import Gamma
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


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


def select_significant_voxels(beta_samples, gamma):
    """
    Implements Section 3.3 Bayesian FDR selection.

    Args:
      beta_samples: list of T NumPy arrays, each shape (U1, V)
      gamma:        float, desired false discovery rate threshold

    Returns:
      mask:  np.ndarray of bool, shape (V,), True = selected voxel
      p_hat: np.ndarray of float, shape (V,), inclusion probabilities
      delta: float, cutoff probability
      r:     int, number of voxels selected
    """
    # 1) Stack into array of shape (T, U1, V)
    beta_arr = np.stack(beta_samples, axis=0)
    # 2) For each draw t and voxel j, flag if any unit weight ≠ 0 → shape (T, V)
    any_nz = np.any(beta_arr != 0, axis=1)
    # 3) Average over T draws to get p_hat[j] ∈ [0,1] → shape (V,)
    p_hat = any_nz.astype(float).mean(axis=0)
    # 4) Sort p_hat descending
    order = np.argsort(-p_hat)       # indices that sort high→low
    p_sorted = p_hat[order]             # sorted probabilities
    # 5) Compute running FDR for top k voxels
    fdr = np.cumsum(1 - p_sorted) / np.arange(1, len(p_sorted) + 1)
    # print("fdr:", fdr)
    # 6) Find largest k with FDR(k) ≤ γ
    valid = np.where(fdr <= gamma)[0]
    if valid.size > 0:
        r = int(valid[-1] + 1)
        delta = float(p_sorted[r - 1])
    else:
        r, delta = 0, 1.0

    # 7) Build final mask
    mask = p_hat >= delta
    return mask, p_hat, delta, r


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
        # print(f'initial sigma^2={sigma_squared}')
        sigma_theta_squared = self._sample_sigma_theta_squared()

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            # print(f'epoch={epoch}')
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")

            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                # print(f'train:: batch_idx={batch_idx} X_batch.shape={X_batch.shape}, y_batch.shape={y_batch.shape}')

                print(self.model.input_layer.beta)
                beta = self.model.input_layer.beta
                beta_thresh = self.model.input_layer.soft_threshold(beta)
                beta.data.copy_(beta_thresh)
                print(self.model.input_layer.beta)
                exit()
                # total_grad is 1D vector, concatenated by all parameters in the model, which is a common practice
                total_grad = self._calculate_total_grad(X_batch, y_batch, sigma_squared, sigma_theta_squared, n, self.model.input_layer.beta, self.model.input_layer.sigma_lambda_squared)

                with torch.no_grad():
                    # for name, param in self.model.named_parameters():
                    #     print(f"Layer: {name}")
                    #     print(f"Shape: {param.shape}")
                    #     print(f"Values:\n{param.data}")
                    #     print("-" * 30)
                    param_list = [p for p in self.model.parameters()]
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
    """
    def select_significant_voxels(self, gamma):

        # list that will hold all beta samples
        beta_samples = []
        dev = self.model.input_layer.beta.device

        param_shape = self.model.input_layer.beta.shape
        U1, V = param_shape
        # U1= amount of units
        # V = amount of voxels
        for flat_params in self.samples['params']:
            # find and reconstruct only the beta parameter
            idx = 0
            for p in self.model.parameters():
                numel = p.numel()
                if p is self.model.input_layer.beta:
                    chunk = flat_params[idx:idx+numel]
                    beta = torch.from_numpy(chunk.reshape(param_shape)).to(dev)
                    break
                idx += numel

            # apply the same soft‐threshold you use in train
            b = self.model.input_layer.soft_threshold(beta)
            beta_samples.append(b.detach().cpu().numpy())  # (U1, V)
        # build array (length of beta samples, U1, V) and compute inclusion flags
        print(len(beta_samples))

        # assume beta_samples is your list of length T=2000,
        # each element a NumPy array of shape (U1=5, V=25)
        beta_arr = np.stack(beta_samples, axis=0)       # shape: (2000, 5, 25)

        # for each sample t and voxel j, check if any of the 5 units is nonzero:
        #   any_nz[t, j] == True  if ∃u such that beta_arr[t, u, j] != 0
        any_nz = np.any(beta_arr != 0, axis=1)          # shape: (2000, 25), dtype=bool

        # convert to 0/1 and average over the 2000 samples:
        #   p_hat[j] = (1/2000) * sum_t any_nz[t, j]
        p_hat = any_nz.astype(float).mean(axis=0)       # shape: (25,), dtype=float          # (V,) float
        # print(p_hat)
        # FDR thresholding
        order = np.argsort(-p_hat) # (V,) int, sort all values of -phat
        p_sorted = p_hat[order]
        print(p_sorted)
        # compute running fdr for top k voxels
        fdr = np.cumsum(1 - p_sorted) / np.arange(1, V+1)     # (V,) float
        # at each prefix length k, frd[k-1] tells expected proportion of false positives for that many voxels
        print(fdr)
        valid = np.where(fdr <= gamma)[0] # all positions that are under gamma
        print(valid)
        if valid.size:
            r = int(valid[-1] + 1) # number of voxels to select, k= index + 1
            delta = float(p_sorted[r - 1]) # probability threshold
        else:
            r, delta = 0, 1.0
        print(r)
        print(delta)

        mask = p_hat >= delta  # (V,) bool
        print(p_hat)
        return mask, p_hat, delta, r
    """
    def predict(self, X, gamma=None):
        if gamma is not None:
            beta_samples = []
            dev = self.model.input_layer.beta.device

            param_shape = self.model.input_layer.beta.shape
            for flat_params in self.samples['params']:
                # find and reconstruct only the beta parameter
                idx = 0
                for p in self.model.parameters():
                    numel = p.numel()
                    if p is self.model.input_layer.beta:
                        chunk = flat_params[idx:idx+numel]
                        beta = torch.from_numpy(chunk.reshape(param_shape)).to(dev)
                        # apply the same soft‐threshold you use in train
                        b = self.model.input_layer.soft_threshold(beta)
                        beta_samples.append(b.detach().cpu().numpy())  # (U1, V)
                        break
                    idx += numel

            mask, p_hat, delta, r = select_significant_voxels(beta_samples, gamma)
            print(f"Threshold δ={delta:.3f}, selecting r={r} voxels at FDR={gamma}")

            # 2) zero out all beta weights for voxels where mask[j] is False
            bool_mask = torch.tensor(mask, device=self.model.input_layer.beta.device)
            # assume beta shape is (U1, V), so we mask along the V dimension
            self.model.input_layer.beta.data[:, ~bool_mask] = 0
            p_masked = p_hat * mask.astype(float)
            #   now p_masked[j] == p_hat[j] if mask[j]==True, else 0.0

            # 2) reshape into your image grid
            dim = int(np.sqrt(p_masked.size))
            prob_img_masked = p_masked.reshape(dim, dim)

            # 3) display
            plt.figure()
            plt.imshow(prob_img_masked, interpolation='nearest')
            plt.colorbar(label="Inclusion probability (masked)")
            plt.title(f"Thresholded p̂ (δ={delta:.3f}, r={r})")
            plt.tight_layout()
            plt.show()

        with torch.no_grad():
            param_list = [p for p in self.model.parameters()]
            # For each sample, make new prediction and then average predictions.(Monte Carlo)
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

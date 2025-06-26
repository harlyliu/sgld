import torch


def generate_linear_data(n=1000, in_features=3, noise_std=1.0):
    """
    Generate synthetic data for multivariate linear regression: y = X @ w + b + noise.

    Args:
        n (int): Number of samples.
        in_features (int): Number of input features.
        noise_std (float): Standard deviation of Gaussian noise.

    Returns:
        tuple: (X, y) – Design matrix and target vector.
    """
    # True parameters
    # true_weights = torch.FloatTensor(in_features).uniform_(-2, 2)  # Random weights for each feature
    weight = [x + 1.0 for x in range(in_features)]
    true_weights = torch.tensor(weight)
    true_bias = torch.tensor(-1.0)  # Single bias term
    # print(f"True weights: {true_weights}")  # Example: tensor([1.234, -0.567, 0.890])
    # print(f"True bias: {true_bias}")      # tensor(-1.0)

    # Generate X (uniformly distributed between -5 and 5)
    X = torch.FloatTensor(n, in_features).uniform_(-5, 5)

    # Generate y = X @ w + b + noise
    # X @ w performs matrix multiplication between (n, in_features) and (in_features,) -> (n,)
    # We add the bias and noise afterward
    noise = torch.normal(mean=0, std=noise_std, size=(n,))
    y = X @ true_weights + true_bias + noise  # Shape: (n,)
    y = y.unsqueeze(1)  # torch.Size([1000]) -> torch.Size([1000, 1])
    return X, y, true_weights, true_bias


# Example usage:
if __name__ == "__main__":
    # Generate data with 3 features
    X, y, true_weights, true_bias = generate_linear_data(n=1000, in_features=1, noise_std=0.5)
    print(f"X shape: {X.shape}")  # torch.Size([1000, 3])
    print(f"y shape: {y.shape}")  # torch.Size([1000])
    print(true_weights[0] + true_bias)


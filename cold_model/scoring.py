import numpy as np

def mse_per_sequence(X_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.mean((X_hat - X) ** 2, axis=(1, 2))

def compute_tau(val_mse: np.ndarray, q: float) -> float:
    return float(np.quantile(val_mse, q))

def anomaly_mask(seq_mse: np.ndarray, tau: float) -> np.ndarray:
    return seq_mse > tau

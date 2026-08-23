from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ThresholdResult:
    tau: float
    seq_mse: np.ndarray
    is_anom: np.ndarray  # boolean per sequence


def compute_seq_mse(X_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.mean((X_hat - X) ** 2, axis=(1, 2))


def estimate_tau(
    tau_fixed: Optional[float],
    tau_source: str,
    tau_quantile: float,
    seq_mse: np.ndarray,
    train_mse: Optional[np.ndarray],
) -> float:
    if tau_fixed is not None:
        return float(tau_fixed)

    if tau_source == "train_idle" and train_mse is not None and len(train_mse) > 0:
        return float(np.quantile(train_mse, tau_quantile))

    return float(np.quantile(seq_mse, tau_quantile))


def smooth_min_normal_run(is_anom: np.ndarray, min_normal_seq_run: int) -> np.ndarray:
    """
    Converts short normal runs into anomalies.
    Same logic as your script.
    """
    if min_normal_seq_run <= 1:
        return is_anom

    flags_normal = (~is_anom).astype(int)  # 1=normal, 0=anom
    cleaned_normal = flags_normal.copy()
    run_start = None

    for i, f in enumerate(flags_normal):
        if f and run_start is None:
            run_start = i

        is_last = (i == len(flags_normal) - 1)
        if (not f or is_last) and run_start is not None:
            run_end = i if not f else i
            length = run_end - run_start + 1
            if length < min_normal_seq_run:
                cleaned_normal[run_start:run_end + 1] = 0
            run_start = None

    return (cleaned_normal == 0)

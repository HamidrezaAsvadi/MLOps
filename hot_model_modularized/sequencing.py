from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SequenceResult:
    X_all: np.ndarray       # shape (n_seq, seq_len, n_feat), float32
    idx_end: np.ndarray     # shape (n_seq,), int
    normal_mask_seq: Optional[np.ndarray]  # boolean mask over sequences if row_mask provided


def build_sequences(
    X_scaled: np.ndarray,
    seq_len: int,
    step: int,
    row_mask: Optional[np.ndarray] = None,
) -> SequenceResult:
    T = X_scaled.shape[0]
    xs = []
    idx_end = []

    for end in range(seq_len - 1, T, step):
        start = end - (seq_len - 1)
        xs.append(X_scaled[start:end + 1])
        idx_end.append(end)

    if not xs:
        raise RuntimeError("No sequences could be built (check SEQ_LEN / data length).")

    X_all = np.stack(xs).astype(np.float32)
    idx_end = np.array(idx_end, dtype=int)

    normal_mask_seq = None
    if row_mask is not None:
        if len(row_mask) != T:
            raise ValueError("row_mask length must match X_scaled rows.")
        normal_mask_seq = row_mask[idx_end]

    return SequenceResult(X_all=X_all, idx_end=idx_end, normal_mask_seq=normal_mask_seq)

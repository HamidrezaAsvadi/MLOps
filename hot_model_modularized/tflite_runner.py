from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tensorflow.lite.python.interpreter import Interpreter

@dataclass
class TFLiteRunner:
    model_path: str
    use_flex_if_found: bool = True
    interpreter: Optional[Interpreter] = None
    inp_index: Optional[int] = None
    out_index: Optional[int] = None
    flex_delegate: Optional[object] = None

    def __post_init__(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"TFLite model not found at: {self.model_path}")

        self.flex_delegate = None

        if self.flex_delegate is not None:
            self.interpreter = Interpreter(
                model_path=str(self.model_path),
                experimental_delegates=[self.flex_delegate],
            )
        else:
            self.interpreter = Interpreter(model_path=str(self.model_path))

        self.interpreter.allocate_tensors()
        inp = self.interpreter.get_input_details()[0]
        out_d = self.interpreter.get_output_details()[0]
        self.inp_index = inp["index"]
        self.out_index = out_d["index"]

    def reconstruct(self, X: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Runs reconstruction in batches.
        X must be float32 of shape (n_seq, seq_len, n_feat).
        """
        if self.interpreter is None or self.inp_index is None or self.out_index is None:
            raise RuntimeError("Interpreter not initialized.")

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        n_seq = X.shape[0]
        X_hat = np.zeros_like(X, dtype=np.float32)

        for i0 in range(0, n_seq, batch_size):
            i1 = min(i0 + batch_size, n_seq)
            xb = X[i0:i1]

            self.interpreter.resize_tensor_input(self.inp_index, xb.shape, strict=True)
            self.interpreter.allocate_tensors()

            self.interpreter.set_tensor(self.inp_index, xb)
            self.interpreter.invoke()
            yb = self.interpreter.get_tensor(self.out_index).astype(np.float32)

            X_hat[i0:i1] = yb

        return X_hat

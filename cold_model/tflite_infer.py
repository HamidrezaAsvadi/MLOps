import io
import time
import contextlib
import numpy as np
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

def tflite_predict_sequences(tflite_path: str, X: np.ndarray, batch_size: int = 250) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    start = time.time()
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
    print(f"Interpreter {(time.time() - start):.4f} seconds")

    start = time.time()
    input_details = interpreter.get_input_details()
    print(f"get_input_details {(time.time() - start):.4f} seconds")
    start = time.time()
    output_details = interpreter.get_output_details()
    print(f"get_output_details {(time.time() - start):.4f} seconds")
    in_idx = input_details[0]["index"]
    out_idx = output_details[0]["index"]

    N = X.shape[0]
    Y = np.empty_like(X, dtype=np.float32)

    start = time.time()
    cnt = 0
    for i in range(0, N, batch_size):
        cnt += 1
        # cycle_start = time.time()
        xb = X[i:i + batch_size]
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            interpreter.resize_tensor_input(in_idx, xb.shape, strict=False)
            interpreter.allocate_tensors()
            interpreter.set_tensor(in_idx, xb)
            interpreter.invoke()
            yb = interpreter.get_tensor(out_idx)
        Y[i:i + len(xb)] = yb
        # print(f"cycle {(time.time() - cycle_start):.4f} seconds")
    print(f"prediction over {cnt} steps {batch_size} batch_size {N} total {(time.time() - start):.4f} seconds")
    return Y

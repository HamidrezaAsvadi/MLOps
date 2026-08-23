import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

def select_feature_columns(out: pd.DataFrame) -> list[str]:
    feat_cols = []
    if "Temperature" in out.columns:
        feat_cols.append("Temperature")
    if "SetPointR" in out.columns:
        feat_cols.append("SetPointR")
    if "State" in out.columns and pd.api.types.is_numeric_dtype(out["State"]):
        feat_cols.append("State")
    if not feat_cols:
        raise ValueError("No usable features (Temperature/SetPointR/State) found.")
    return feat_cols

def resample_and_interpolate(out: pd.DataFrame, feat_cols: list[str], resample_freq: str) -> pd.DataFrame:
    time_col = "time"
    df_feat = out[[time_col] + feat_cols].copy()
    df_feat[time_col] = pd.to_datetime(df_feat[time_col], errors="coerce")
    df_feat = df_feat.dropna(subset=[time_col]).sort_values(time_col)

    df_res = df_feat.set_index(time_col).resample(resample_freq).mean()
    df_res[feat_cols] = (
        df_res[feat_cols]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    df_res = df_res.dropna(subset=feat_cols, how="all")
    if df_res.empty:
        raise ValueError("Resampled dataframe is empty after interpolation/fill.")
    return df_res.reset_index()

def scale_features(df_res: pd.DataFrame, feat_cols: list[str]):
    X_mat = df_res[feat_cols].values
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_mat)
    return X_scaled, scaler

def build_sequences(X_scaled: np.ndarray, seq_len: int, step: int):
    T = X_scaled.shape[0]
    xs = []
    idx_end = []
    for end in range(seq_len - 1, T, step):
        start = end - (seq_len - 1)
        xs.append(X_scaled[start:end + 1])
        idx_end.append(end)
    if not xs:
        raise ValueError("Not enough points to form sequences.")
    return np.stack(xs).astype(np.float32), np.array(idx_end, dtype=int)

def pick_normal_sequences(df_res: pd.DataFrame, idx_end: np.ndarray, X_all: np.ndarray, device_name: str, normal_windows: dict):
    time_col = "time"
    dev_windows = normal_windows.get(device_name)
    if dev_windows is None:
        dev_windows = normal_windows.get("__default__")

    normal_mask_seq = None
    if dev_windows:
        t_series = pd.to_datetime(df_res[time_col].values)
        row_mask = np.zeros(len(df_res), dtype=bool)
        for a, b in dev_windows:
            a_ts = pd.to_datetime(a)
            b_ts = pd.to_datetime(b)
            if b_ts < a_ts:
                a_ts, b_ts = b_ts, a_ts
            row_mask |= (t_series >= a_ts) & (t_series <= b_ts)
        normal_mask_seq = row_mask[idx_end]

    if normal_mask_seq is not None and normal_mask_seq.any():
        X_normal = X_all[normal_mask_seq]
    else:
        split_i = max(int(0.6 * len(X_all)), 1)
        X_normal = X_all[:split_i]

    if X_normal.shape[0] < 10:
        raise ValueError("Not enough normal sequences to score. Add windows or reduce SEQ_LEN.")
    return X_normal

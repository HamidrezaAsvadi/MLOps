from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


@dataclass
class PreprocessResult:
    df_res: pd.DataFrame
    X_scaled: np.ndarray
    row_mask: Optional[np.ndarray]  # boolean mask on df_res rows for train windows (or None)
    scaler: RobustScaler


def resample_interpolate_and_scale(
    df_temp: pd.DataFrame,
    feat_cols: List[str],
    resample_rule: str,
    idle_train_windows: Optional[List[Tuple[str, str]]],
    time_col: str = "time",
) -> PreprocessResult:
    df_feat = df_temp[[time_col] + feat_cols].copy()
    df_feat[time_col] = pd.to_datetime(df_feat[time_col], errors="coerce")
    df_feat = df_feat.dropna(subset=[time_col]).sort_values(time_col)
    if df_feat.empty:
        raise RuntimeError("No valid timestamps after parsing.")

    df_res = df_feat.set_index(time_col).resample(resample_rule.lower()).mean()

    df_res[feat_cols] = (
        df_res[feat_cols]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )

    df_res = df_res.dropna(subset=feat_cols, how="all")
    if df_res.empty:
        raise RuntimeError("Resampled dataframe is empty after interpolation.")

    df_res = df_res.reset_index()

    X_mat = df_res[feat_cols].values

    row_mask = None
    if idle_train_windows:
        t_series = pd.to_datetime(df_res[time_col].values)
        row_mask = np.zeros(len(df_res), dtype=bool)

        norm_wins = []
        for a, b in idle_train_windows:
            a_ts = pd.to_datetime(a)
            b_ts = pd.to_datetime(b)
            if b_ts < a_ts:
                a_ts, b_ts = b_ts, a_ts
            norm_wins.append((a_ts, b_ts))

        for a_ts, b_ts in norm_wins:
            row_mask |= (t_series >= a_ts) & (t_series <= b_ts)

    scaler = RobustScaler()
    if row_mask is not None and row_mask.any():
        scaler.fit(X_mat[row_mask])
    else:
        scaler.fit(X_mat)

    X_scaled = scaler.transform(X_mat)
    return PreprocessResult(df_res=df_res, X_scaled=X_scaled, row_mask=row_mask, scaler=scaler)

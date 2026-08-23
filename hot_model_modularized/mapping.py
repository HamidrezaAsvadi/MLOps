from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class MappingResult:
    per_resampled: pd.Series  # bool per df_res row
    per_sample: pd.Series     # bool per df_temp row


def map_normal_endpoints_to_original_samples(
    df_res: pd.DataFrame,
    idx_end: np.ndarray,
    is_normal_seq: np.ndarray,
    df_temp: pd.DataFrame,
    time_col: str = "time",
) -> MappingResult:
    per_resampled = pd.Series(False, index=df_res.index)

    idx_end = np.array(idx_end, dtype=int)
    normal_end_idxs = idx_end[is_normal_seq]
    if len(normal_end_idxs) > 0:
        per_resampled.iloc[normal_end_idxs] = True

    per_sample = pd.Series(False, index=df_temp.index)

    if per_resampled.any():
        df_res_normal = df_res.loc[per_resampled.values, [time_col]].copy().sort_values(time_col)

        df_orig_times = (
            df_temp[[time_col]]
            .copy()
            .dropna()
            .astype({time_col: "datetime64[ns]"})
            .reset_index()
            .rename(columns={"index": "orig_idx"})
            .sort_values(time_col)
        )

        mapped = pd.merge_asof(
            df_res_normal,
            df_orig_times,
            on=time_col,
            direction="backward",
        )

        mapped = mapped.dropna(subset=["orig_idx"])
        mapped_idxs = mapped["orig_idx"].astype(int).unique().tolist()
        per_sample.iloc[mapped_idxs] = True

    return MappingResult(per_resampled=per_resampled, per_sample=per_sample)

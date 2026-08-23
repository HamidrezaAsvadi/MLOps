import numpy as np
import pandas as pd

def map_anomalies_to_regions(out: pd.DataFrame, df_res: pd.DataFrame, idx_end: np.ndarray, is_anom: np.ndarray,
                            exclusion_mask: np.ndarray, ignore_defrost: bool, defrost_windows: list):
    time_col = "time"

    per_resampled = np.zeros(len(df_res), dtype=bool)
    anom_end_idxs = idx_end[is_anom]
    per_resampled[anom_end_idxs] = True

    per_sample_mask = np.zeros(len(out), dtype=bool)

    if per_resampled.any():
        df_res_anom = df_res.loc[per_resampled, [time_col]].copy().sort_values(time_col)

        df_orig_times = (
            out[[time_col]]
            .copy()
            .dropna()
            .astype({time_col: "datetime64[ns]"})
            .reset_index()
            .rename(columns={"index": "orig_idx"})
            .sort_values(time_col)
        )

        mapped2 = pd.merge_asof(df_res_anom, df_orig_times, on=time_col, direction="backward")
        mapped2 = mapped2.dropna(subset=["orig_idx"])
        mapped_idxs = mapped2["orig_idx"].astype(int).unique().tolist()
        per_sample_mask[mapped_idxs] = True

    if ignore_defrost and len(defrost_windows) > 0:
        per_sample_mask = per_sample_mask & (~exclusion_mask)

    filtered_times = out.loc[per_sample_mask, time_col].values

    anom_regions = []
    if len(filtered_times) > 0:
        start = filtered_times[0]
        prev = filtered_times[0]

        median_step = out[time_col].diff().median()
        if pd.isna(median_step):
            median_step = pd.Timedelta(seconds=0)

        max_gap = median_step * 1.5 if median_step > pd.Timedelta(0) else pd.Timedelta(seconds=0)

        for t in filtered_times[1:]:
            if max_gap and (t - prev) > max_gap:
                anom_regions.append((pd.to_datetime(start), pd.to_datetime(prev)))
                start = t
            prev = t
        anom_regions.append((pd.to_datetime(start), pd.to_datetime(prev)))

    return anom_regions

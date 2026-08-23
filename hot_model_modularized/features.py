from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd


@dataclass
class DeviceFeatureBuildResult:
    df_temp: pd.DataFrame
    feat_cols: List[str]
    base_timeline: pd.DataFrame
    temp_measure: Optional[str]
    sp_measure: Optional[str]


def build_device_features(
    raw_all: pd.DataFrame,
    device_id: str,
    state_feature_scale: float,
    quiet: bool = True,
    device_name: str = "",
) -> DeviceFeatureBuildResult:
    time_col = "time"
    meas_col = "measure"
    val_double = "value_double"
    val_bigint = "value_bigint"
    val_bool = "value_boolean"

    df_dev = raw_all[raw_all["device"] == device_id].copy()
    if df_dev.empty:
        raise ValueError(f"No data for device_id={device_id!r}")

    df_dev[time_col] = pd.to_datetime(df_dev[time_col], errors="coerce")
    df_dev = df_dev.dropna(subset=[time_col]).sort_values(time_col)

    base_timeline = (
        df_dev[[time_col]]
        .dropna()
        .drop_duplicates()
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    out = base_timeline.copy()

    measures = df_dev[meas_col].astype(str).unique().tolist()

    temp_candidates = ["ProbeR", "CabinetProbe", "TempR", "Temperature", "Temp"]
    temp_measure = next((m for m in temp_candidates if m in measures), None)

    sp_candidates = ["SetPointR", "TemperatureSetPoint", "SetPoint", "Setpoint"]
    sp_measure = next((m for m in sp_candidates if m in measures), None)

    # Pick best value column for temperature
    temp_col = None
    if temp_measure is not None:
        subset = df_dev[df_dev[meas_col] == temp_measure]
        if not subset.empty:
            candidates = [c for c in [val_double, val_bigint, val_bool] if c in subset.columns]
            counts = {c: subset[c].notna().sum() for c in candidates}
            if counts:
                best_col = max(counts.items(), key=lambda kv: kv[1])[0]
                if counts[best_col] > 0:
                    temp_col = best_col

    # Pick best value column for setpoint
    sp_col = None
    if sp_measure is not None:
        subset = df_dev[df_dev[meas_col] == sp_measure]
        if not subset.empty:
            candidates = [c for c in [val_double, val_bigint, val_bool] if c in subset.columns]
            counts = {c: subset[c].notna().sum() for c in candidates}
            if counts:
                best_col = max(counts.items(), key=lambda kv: kv[1])[0]
                if counts[best_col] > 0:
                    sp_col = best_col

    # Temperature
    if temp_measure is not None and temp_col is not None:
        df_temp_series = (
            df_dev[df_dev[meas_col] == temp_measure][[time_col, temp_col]]
            .rename(columns={temp_col: "Temperature"})
            .sort_values(time_col)
        )
        df_temp_series["Temperature"] = pd.to_numeric(df_temp_series["Temperature"], errors="coerce")
        df_temp_series.loc[df_temp_series["Temperature"] <= -1000, "Temperature"] = np.nan
        out = out.merge(df_temp_series, on=time_col, how="left")
        out["Temperature"] = out["Temperature"].ffill()
    else:
        out["Temperature"] = np.nan

    # SetPoint
    if sp_measure is not None and sp_col is not None:
        df_sp_series = (
            df_dev[df_dev[meas_col] == sp_measure][[time_col, sp_col]]
            .rename(columns={sp_col: "SetPointR"})
            .sort_values(time_col)
        )
        df_sp_series["SetPointR"] = pd.to_numeric(df_sp_series["SetPointR"], errors="coerce")
        df_sp_series.loc[df_sp_series["SetPointR"] <= -1000, "SetPointR"] = np.nan
        out = out.merge(df_sp_series, on=time_col, how="left")
        out["SetPointR"] = out["SetPointR"].ffill()
    else:
        out["SetPointR"] = np.nan

    # State (merge_asof backward)
    df_state_raw = (
        df_dev[df_dev[meas_col] == "State"][[time_col, val_bigint]]
        .rename(columns={val_bigint: "State"})
        .sort_values(time_col)
    )
    if not df_state_raw.empty:
        df_state_raw["State"] = pd.to_numeric(df_state_raw["State"], errors="coerce")
        left = base_timeline.sort_values(time_col)
        right = df_state_raw.sort_values(time_col)
        merged_asof = pd.merge_asof(left, right, on=time_col, direction="backward")
        out = out.merge(merged_asof, on=time_col, how="left")
    else:
        out["State"] = np.nan

    # Optional Output1 / HeatRelay
    for out_measure, col_name in [("Output1", "Output1"), ("HeatRelay", "HeatRelay", "ReleState")]:
        df_out_raw = df_dev[df_dev[meas_col] == out_measure][[time_col, val_bigint, val_bool]].copy()
        df_out_raw = df_out_raw.sort_values(time_col)

        if not df_out_raw.empty:
            cnt_big = df_out_raw[val_bigint].notna().sum()
            cnt_bool = df_out_raw[val_bool].notna().sum()
            src_col = val_bigint if cnt_big >= cnt_bool else val_bool

            df_out_series = df_out_raw[[time_col, src_col]].rename(columns={src_col: col_name})
            df_out_series[col_name] = pd.to_numeric(df_out_series[col_name], errors="coerce")

            left = base_timeline.sort_values(time_col)
            right = df_out_series.sort_values(time_col)
            merged_asof = pd.merge_asof(left, right, on=time_col, direction="backward")

            out = out.merge(merged_asof, on=time_col, how="left")
            out[col_name] = out[col_name].fillna(0)
        else:
            out[col_name] = 0.0

    df_temp = out.copy()

    # Derived features
    df_temp["temp_minus_sp"] = df_temp["Temperature"] - df_temp["SetPointR"]
    df_temp["abs_temp_minus_sp"] = df_temp["temp_minus_sp"].abs()

    # Scale state-like features (must match training)
    for col in ["State", "Output1", "HeatRelay"]:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col] * state_feature_scale

    feat_cols = [
        c for c in [
            "Temperature",
            "SetPointR",
            "temp_minus_sp",
            "abs_temp_minus_sp",
            "State",
            "Output1",
            "HeatRelay",
        ] if c in df_temp.columns
    ]
    if not feat_cols:
        raise RuntimeError(f"No usable features for {device_name or device_id}")

    return DeviceFeatureBuildResult(
        df_temp=df_temp,
        feat_cols=feat_cols,
        base_timeline=base_timeline,
        temp_measure=temp_measure,
        sp_measure=sp_measure,
    )

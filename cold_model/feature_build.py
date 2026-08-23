import numpy as np
import pandas as pd
from datetime import timedelta

def pick_device(df_all: pd.DataFrame, device_filters: list[str]) -> str:
    unique_devices = sorted(df_all["device"].dropna().astype(str).unique().tolist())
    needles = [s.lower() for s in device_filters]
    devices = [d for d in unique_devices if any(n in d.lower() for n in needles)]
    if not devices:
        raise ValueError("No matching device found.")
    return devices[0]

def filter_device(df_all: pd.DataFrame, device_name: str) -> pd.DataFrame:
    df = df_all[df_all["device"] == device_name].copy()
    if df.empty:
        raise ValueError(f"No data for device {device_name!r}")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

def build_aligned_table(df: pd.DataFrame) -> pd.DataFrame:
    time_col = "time"
    val_col = "value_double"
    big_col = "value_bigint"
    bool_col = "value_boolean"
    meas_col = "measure"

    df_temp = (
        df[df[meas_col] == "ProbeR"][[time_col, val_col]]
        .rename(columns={val_col: "Temperature"})
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    df_sp = (
        df[df[meas_col] == "SetPointR"][[time_col, val_col]]
        .rename(columns={val_col: "SetPointR"})
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    df_state_raw = (
        df[df[meas_col] == "State"][[time_col, big_col]]
        .rename(columns={big_col: "State"})
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    df_def_raw = (
        df[df[meas_col] == "Defrost"][[time_col, bool_col]]
        .rename(columns={bool_col: "Defrost"})
        .sort_values(time_col)
        .reset_index(drop=True)
    )

    if not df_def_raw.empty:
        mapping = {
            "true": True, "false": False, "1": True, "0": False,
            "t": True, "f": False, "yes": True, "no": False,
        }
        df_def_raw["Defrost"] = df_def_raw["Defrost"].astype(str).str.strip().str.lower().map(mapping)
        df_def_raw = df_def_raw.dropna(subset=["Defrost"]).copy()
        df_def_raw["Defrost"] = df_def_raw["Defrost"].astype(bool)

    base_timeline = df_temp[[time_col]].copy() if not df_temp.empty else df[[time_col]].drop_duplicates().sort_values(time_col).reset_index(drop=True)
    out = base_timeline.copy()

    if not df_temp.empty:
        out = out.merge(df_temp, on=time_col, how="left")

    if not df_sp.empty:
        out = out.merge(
            pd.merge_asof(base_timeline, df_sp.sort_values(time_col), on=time_col, direction="backward"),
            on=time_col,
            how="left",
        )

    if not df_state_raw.empty:
        df_state_raw["State"] = pd.to_numeric(df_state_raw["State"], errors="coerce")
        out = out.merge(
            pd.merge_asof(base_timeline, df_state_raw.sort_values(time_col), on=time_col, direction="backward"),
            on=time_col,
            how="left",
        )
    else:
        out["State"] = np.nan

    if not df_def_raw.empty:
        out = out.merge(
            pd.merge_asof(base_timeline, df_def_raw.sort_values(time_col), on=time_col, direction="backward"),
            on=time_col,
            how="left",
        )
        out["Defrost"] = out["Defrost"].fillna(False).astype(bool)
    else:
        out["Defrost"] = False

    return out

def build_defrost_exclusion(out: pd.DataFrame, ignore: bool, cooldown_min: int):
    time_col = "time"
    defrost_windows = []

    if (not out.empty) and ("Defrost" in out.columns):
        s = out[[time_col, "Defrost"]].sort_values(time_col).reset_index(drop=True).copy()
        s["Defrost"] = s["Defrost"].fillna(False).astype(bool)

        in_on = False
        start = None
        for i in range(len(s)):
            flag = bool(s.loc[i, "Defrost"])
            t = s.loc[i, time_col]
            prev = bool(s.loc[i - 1, "Defrost"]) if i > 0 else False

            if flag and (i == 0 or not prev) and not in_on:
                start = t
                in_on = True
            if (not flag) and (i > 0 and prev) and in_on:
                defrost_windows.append((start, t))
                start = None
                in_on = False

        if in_on and start is not None:
            defrost_windows.append((start, s[time_col].iloc[-1]))

    exclusion_mask = np.zeros(len(out), dtype=bool)
    if ignore and defrost_windows:
        for a, b in defrost_windows:
            a = pd.to_datetime(a)
            b = pd.to_datetime(b) + timedelta(minutes=cooldown_min)
            exclusion_mask |= out[time_col].between(a, b, inclusive="left").values

    return defrost_windows, exclusion_mask

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ============================================================
# Configuration
# ============================================================


@dataclass
class InferenceConfig:
    HEATING_DEVICE_NAMES: List[str] = field(
        default_factory=lambda: [
            "Frontcooking Grill Left",
            "Frontcooking Grill Right",
        ]
    )

    RESAMPLE_RULE: str = "1S"
    SEQ_LEN: int = 30
    STEP: int = 1
    BATCH: int = 64

    TAU_SOURCE: str = "train_idle"  # train_idle or global
    TAU_QUANTILE: float = 0.99
    MIN_NORMAL_SEQ_RUN: int = 10
    MIN_ANOM_SEQ_RUN: int = 5
    STATE_FEATURE_SCALE: float = 0.3
    IDLE_CONFIRM_MINUTES: int = 5

    EPS_FLAT: float = 0.02
    DROP_THR: float = -0.10
    DEEP_DROP_THR_5: float = -1.0
    DEEP_DROP_THR_10: float = -2.0
    BELOW_SP_THR: float = -1.0

    SCRIPT_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    METADATA_PATH: str = ""
    COMBINED_CSV_PATH: str = ""
    TFLITE_MODEL_PATH: str = ""

    IDLE_TRAIN_WINDOWS: Dict[str, Optional[List[Tuple[str, str]]]] = field(
        default_factory=lambda: {
            "Frontcooking Grill Left": [
                ("2025-04-04 16:00:00", "2025-04-04 17:00:00"),
                ("2025-04-08 16:00:00", "2025-04-08 17:00:00"),
                ("2025-04-11 16:00:00", "2025-04-11 17:00:00"),
                ("2025-04-14 14:45:00", "2025-04-14 17:00:00"),
                ("2025-04-19 15:45:00", "2025-04-19 16:45:00"),
                ("2025-04-20 16:00:00", "2025-04-20 16:45:00"),
                ("2025-04-24 16:08:00", "2025-04-24 17:30:00"),
                ("2025-04-27 09:15:00", "2025-04-27 10:15:00"),
                ("2025-04-29 15:30:00", "2025-04-29 17:30:00"),
            ],
            "__default__": None,
        }
    )


def make_default_config() -> InferenceConfig:
    cfg = InferenceConfig()
    root = cfg.SCRIPT_DIR.parents[0]
    cfg.METADATA_PATH = str(root / "assets" / "metadata.json")
    cfg.COMBINED_CSV_PATH = str(root / "assets" / "combined_2025-04-14_15_to_17.csv")
    cfg.TFLITE_MODEL_PATH = str(root / "assets" / "hot_model.v1.tflite")
    return cfg


# ============================================================
# IO helpers
# ============================================================


def is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


@dataclass
class DataSource:
    s3_client: Optional[object] = None

    @staticmethod
    def from_paths(*paths: str) -> "DataSource":
        use_s3 = any(is_s3_uri(p) for p in paths if isinstance(p, str))
        if use_s3:
            import boto3
            return DataSource(s3_client=boto3.client("s3"))
        return DataSource(s3_client=None)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        if is_s3_uri(path):
            if self.s3_client is None:
                raise RuntimeError("S3 path provided but s3_client is None.")
            u = urlparse(path)
            bucket = u.netloc
            key = u.path.lstrip("/")
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read().decode(encoding)
        with open(path, "r", encoding=encoding) as f:
            return f.read()

    def read_json(self, path: str):
        return json.loads(self.read_text(path))

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        if is_s3_uri(path):
            if self.s3_client is None:
                raise RuntimeError("S3 path provided but s3_client is None.")
            u = urlparse(path)
            bucket = u.netloc
            key = u.path.lstrip("/")
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return pd.read_csv(obj["Body"], **kwargs)
        return pd.read_csv(path, **kwargs)


# ============================================================
# Shared data structures
# ============================================================


@dataclass
class DeviceCalibration:
    scaler: RobustScaler
    tau: float
    feat_cols: List[str]


@dataclass
class DeviceResult:
    device: str
    device_id: str
    timestamp: pd.Timestamp
    status: str
    tau: float
    latest_score: float
    anomaly_sequences: int
    idle_samples_mapped: int


# ============================================================
# Model runner
# ============================================================


@dataclass
class TFLiteRunner:
    model_path: str
    interpreter: Optional[Interpreter] = None
    inp_index: Optional[int] = None
    out_index: Optional[int] = None

    def __post_init__(self):
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"TFLite model not found at: {self.model_path}")
        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]
        self.inp_index = inp["index"]
        self.out_index = out["index"]

    def reconstruct(self, x: np.ndarray, batch_size: int) -> np.ndarray:
        if x.dtype != np.float32:
            x = x.astype(np.float32)

        y = np.zeros_like(x, dtype=np.float32)
        for i0 in range(0, len(x), batch_size):
            i1 = min(i0 + batch_size, len(x))
            xb = x[i0:i1]
            self.interpreter.resize_tensor_input(self.inp_index, xb.shape, strict=True)
            self.interpreter.allocate_tensors()
            self.interpreter.set_tensor(self.inp_index, xb)
            self.interpreter.invoke()
            y[i0:i1] = self.interpreter.get_tensor(self.out_index).astype(np.float32)
        return y


# ============================================================
# Feature engineering
# ============================================================


def build_name_to_id(metadata_obj, heating_device_names: List[str]) -> Dict[str, str]:
    if not isinstance(metadata_obj, list):
        raise ValueError("metadata.json must be a JSON list of device objects.")

    name_to_id_all = {
        d["name"]: d["id"]
        for d in metadata_obj
        if isinstance(d, dict) and "name" in d and "id" in d
    }

    missing = [n for n in heating_device_names if n not in name_to_id_all]
    if missing:
        raise ValueError(f"These device names were not found in metadata.json: {missing}")

    return {name: name_to_id_all[name] for name in heating_device_names}


@dataclass
class DeviceFeatureBuildResult:
    df_temp: pd.DataFrame
    feat_cols: List[str]


@dataclass
class PreprocessResult:
    df_res: pd.DataFrame
    x_scaled: np.ndarray
    row_mask: Optional[np.ndarray]
    scaler: RobustScaler


@dataclass
class SequenceResult:
    x_all: np.ndarray
    idx_end: np.ndarray
    time_end: np.ndarray
    normal_mask_seq: Optional[np.ndarray]


@dataclass
class StateMachineResult:
    confirmed_idle: np.ndarray
    anomaly_final: np.ndarray
    idle_samples_mapped: int


FEATURE_COLUMNS = [
    "temp_minus_sp",
    "temp_diff_5",
    "State",
    "Output1",
    "relay_on_temp_dropping",
    "deep_temp_drop_5",
    "deep_temp_drop_10",
]

CONTINUOUS_FEATURES = {
    "Temperature",
    "SetPointR",
    "temp_minus_sp",
    "abs_temp_minus_sp",
    "temp_diff_1",
    "temp_diff_5",
    "temp_diff_10",
    "temp_slope_5",
    "temp_slope_10",
    "temp_std_10",
    "temp_std_30",
}

STATE_LIKE_FEATURES = {"State", "Output1", "HeatRelay"}

BINARY_FLAG_FEATURES = {
    "relay_on",
    "relay_on_temp_not_rising",
    "relay_on_temp_dropping",
    "deep_temp_drop_5",
    "deep_temp_drop_10",
    "below_sp_while_relay_on",
}



def _pick_best_value_col(df_dev: pd.DataFrame, measure: str) -> Optional[str]:
    subset = df_dev[df_dev["measure"] == measure]
    if subset.empty:
        return None
    candidates = [c for c in ["value_double", "value_bigint", "value_boolean"] if c in subset.columns]
    counts = {c: subset[c].notna().sum() for c in candidates}
    if not counts:
        return None
    best_col = max(counts.items(), key=lambda kv: kv[1])[0]
    return best_col if counts[best_col] > 0 else None



def build_device_features(raw_all: pd.DataFrame, device_id: str, cfg: InferenceConfig) -> DeviceFeatureBuildResult:
    time_col = "time"
    meas_col = "measure"

    df_dev = raw_all[raw_all["device"] == device_id].copy()
    if df_dev.empty:
        raise ValueError(f"No data for device_id={device_id!r}")

    df_dev[time_col] = pd.to_datetime(df_dev[time_col], errors="coerce")
    df_dev = df_dev.dropna(subset=[time_col]).sort_values(time_col)
    if df_dev.empty:
        raise ValueError(f"No valid timestamps for device_id={device_id!r}")

    base_timeline = (
        df_dev[[time_col]]
        .dropna()
        .drop_duplicates()
        .sort_values(time_col)
        .reset_index(drop=True)
    )
    out = base_timeline.copy()

    measures = df_dev[meas_col].astype(str).unique().tolist()
    temp_measure = next((m for m in ["ProbeR", "CabinetProbe", "TempR", "Temperature", "Temp"] if m in measures), None)
    sp_measure = next((m for m in ["SetPointR", "TemperatureSetPoint", "SetPoint", "Setpoint"] if m in measures), None)

    temp_col = _pick_best_value_col(df_dev, temp_measure) if temp_measure else None
    sp_col = _pick_best_value_col(df_dev, sp_measure) if sp_measure else None

    if temp_measure and temp_col:
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

    if sp_measure and sp_col:
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

    df_state_raw = (
        df_dev[df_dev[meas_col] == "State"][[time_col, "value_bigint"]]
        .rename(columns={"value_bigint": "State"})
        .sort_values(time_col)
    )
    if not df_state_raw.empty:
        df_state_raw["State"] = pd.to_numeric(df_state_raw["State"], errors="coerce")
        out = out.merge(
            pd.merge_asof(base_timeline.sort_values(time_col), df_state_raw, on=time_col, direction="backward"),
            on=time_col,
            how="left",
        )
    else:
        out["State"] = np.nan

    for out_measure, col_name in [
        ("Output1", "Output1"),
        ("HeatRelay", "HeatRelay"),
        ("ReleState", "Output1"),
    ]:
        df_out_raw = df_dev[df_dev[meas_col] == out_measure][[time_col, "value_bigint", "value_boolean"]].copy()
        df_out_raw = df_out_raw.sort_values(time_col)
        if df_out_raw.empty:
            if col_name not in out.columns:
                out[col_name] = 0.0
            continue

        cnt_big = df_out_raw["value_bigint"].notna().sum()
        cnt_bool = df_out_raw["value_boolean"].notna().sum()
        src_col = "value_bigint" if cnt_big >= cnt_bool else "value_boolean"

        df_out_series = df_out_raw[[time_col, src_col]].rename(columns={src_col: col_name})
        df_out_series[col_name] = pd.to_numeric(df_out_series[col_name], errors="coerce")
        merged_asof = pd.merge_asof(base_timeline.sort_values(time_col), df_out_series, on=time_col, direction="backward")

        if col_name in out.columns:
            out[col_name] = out[col_name].combine_first(merged_asof[col_name])
        else:
            out = out.merge(merged_asof, on=time_col, how="left")
        out[col_name] = out[col_name].fillna(0)

    df_temp = out.copy()
    df_temp["temp_minus_sp"] = df_temp["Temperature"] - df_temp["SetPointR"]
    df_temp["abs_temp_minus_sp"] = df_temp["temp_minus_sp"].abs()

    if "Output1" in df_temp.columns:
        relay_raw = pd.to_numeric(df_temp["Output1"], errors="coerce").fillna(0)
    elif "HeatRelay" in df_temp.columns:
        relay_raw = pd.to_numeric(df_temp["HeatRelay"], errors="coerce").fillna(0)
    else:
        relay_raw = pd.Series(0, index=df_temp.index, dtype=float)
    relay_on_raw = (relay_raw > 0).astype(int)

    df_temp["temp_diff_1"] = df_temp["Temperature"].diff(1)
    df_temp["temp_diff_5"] = df_temp["Temperature"].diff(5)
    df_temp["temp_diff_10"] = df_temp["Temperature"].diff(10)
    df_temp["temp_slope_5"] = df_temp["Temperature"].diff(5) / 5.0
    df_temp["temp_slope_10"] = df_temp["Temperature"].diff(10) / 10.0
    df_temp["temp_std_10"] = df_temp["Temperature"].rolling(10, min_periods=1).std()
    df_temp["temp_std_30"] = df_temp["Temperature"].rolling(30, min_periods=1).std()

    df_temp["relay_on"] = relay_on_raw.astype(float)
    df_temp["relay_on_temp_not_rising"] = (
        (relay_on_raw == 1) & (df_temp["temp_slope_5"] <= cfg.EPS_FLAT)
    ).astype(float)
    df_temp["relay_on_temp_dropping"] = (
        (relay_on_raw == 1) & (df_temp["temp_slope_5"] < cfg.DROP_THR)
    ).astype(float)
    df_temp["deep_temp_drop_5"] = (df_temp["temp_diff_5"] <= cfg.DEEP_DROP_THR_5).astype(float)
    df_temp["deep_temp_drop_10"] = (df_temp["temp_diff_10"] <= cfg.DEEP_DROP_THR_10).astype(float)
    df_temp["below_sp_while_relay_on"] = (
        (relay_on_raw == 1) & (df_temp["temp_minus_sp"] < cfg.BELOW_SP_THR)
    ).astype(float)

    engineered_cols = [
        "temp_diff_1",
        "temp_diff_5",
        "temp_diff_10",
        "temp_slope_5",
        "temp_slope_10",
        "temp_std_10",
        "temp_std_30",
        "relay_on",
        "relay_on_temp_not_rising",
        "relay_on_temp_dropping",
        "deep_temp_drop_5",
        "deep_temp_drop_10",
        "below_sp_while_relay_on",
    ]
    for col in engineered_cols:
        df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce").fillna(0.0)

    for col in ["State", "Output1", "HeatRelay"]:
        if col in df_temp.columns:
            df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce").fillna(0.0)
            df_temp[col] = df_temp[col] * cfg.STATE_FEATURE_SCALE

    feat_cols = [c for c in FEATURE_COLUMNS if c in df_temp.columns]
    if not feat_cols:
        raise RuntimeError(f"No usable features for device_id={device_id!r}")

    return DeviceFeatureBuildResult(df_temp=df_temp, feat_cols=feat_cols)



def resample_interpolate_and_scale(
    df_temp: pd.DataFrame,
    feat_cols: List[str],
    resample_rule: str,
    scaler: Optional[RobustScaler] = None,
    idle_train_windows: Optional[List[Tuple[str, str]]] = None,
) -> PreprocessResult:
    time_col = "time"
    df_feat = df_temp[[time_col] + feat_cols].copy()
    df_feat[time_col] = pd.to_datetime(df_feat[time_col], errors="coerce")
    df_feat = df_feat.dropna(subset=[time_col]).sort_values(time_col)
    if df_feat.empty:
        raise RuntimeError("No valid timestamps after parsing.")

    continuous_cols = [c for c in feat_cols if c in CONTINUOUS_FEATURES]
    state_like_cols = [c for c in feat_cols if c in STATE_LIKE_FEATURES]
    binary_flag_cols = [c for c in feat_cols if c in BINARY_FLAG_FEATURES]
    other_cols = [c for c in feat_cols if c not in continuous_cols and c not in state_like_cols and c not in binary_flag_cols]
    continuous_cols.extend(other_cols)

    parts = []
    indexed = df_feat.set_index(time_col)

    if continuous_cols:
        df_cont = indexed[continuous_cols].resample(resample_rule.lower()).mean()
        df_cont[continuous_cols] = (
            df_cont[continuous_cols].interpolate(method="time", limit_direction="both").ffill().bfill()
        )
        parts.append(df_cont)

    if state_like_cols:
        df_state = indexed[state_like_cols].resample(resample_rule.lower()).last()
        df_state[state_like_cols] = df_state[state_like_cols].ffill().bfill().fillna(0)
        parts.append(df_state)

    if binary_flag_cols:
        df_bin = indexed[binary_flag_cols].resample(resample_rule.lower()).max()
        df_bin[binary_flag_cols] = df_bin[binary_flag_cols].ffill().bfill().fillna(0)
        parts.append(df_bin)

    if not parts:
        raise RuntimeError("No feature groups available after resampling.")

    df_res = pd.concat(parts, axis=1)
    df_res = df_res[feat_cols].dropna(subset=feat_cols, how="all")
    if df_res.empty:
        raise RuntimeError("Resampled dataframe is empty after interpolation.")
    df_res = df_res.reset_index()

    row_mask = None
    if idle_train_windows:
        t_series = pd.to_datetime(df_res[time_col].values)
        row_mask = np.zeros(len(df_res), dtype=bool)
        for a, b in idle_train_windows:
            a_ts = pd.to_datetime(a)
            b_ts = pd.to_datetime(b)
            if b_ts < a_ts:
                a_ts, b_ts = b_ts, a_ts
            row_mask |= (t_series >= a_ts) & (t_series <= b_ts)

    x_mat = df_res[feat_cols].values

    scaler_to_use = scaler or RobustScaler()
    if scaler is None:
        if row_mask is not None and row_mask.any():
            scaler_to_use.fit(x_mat[row_mask])
        else:
            scaler_to_use.fit(x_mat)

    x_scaled = scaler_to_use.transform(x_mat)
    return PreprocessResult(df_res=df_res, x_scaled=x_scaled, row_mask=row_mask, scaler=scaler_to_use)



def build_sequences(x_scaled: np.ndarray, seq_len: int, step: int, df_res: pd.DataFrame, row_mask: Optional[np.ndarray]) -> SequenceResult:
    t = x_scaled.shape[0]
    xs = []
    idx_end = []
    for end in range(seq_len - 1, t, step):
        start = end - (seq_len - 1)
        xs.append(x_scaled[start:end + 1])
        idx_end.append(end)
    if not xs:
        raise RuntimeError("No sequences could be built (check SEQ_LEN / data length).")

    x_all = np.stack(xs).astype(np.float32)
    idx_end = np.array(idx_end, dtype=int)
    time_end = df_res.loc[idx_end, "time"].values
    normal_mask_seq = row_mask[idx_end] if row_mask is not None else None
    return SequenceResult(x_all=x_all, idx_end=idx_end, time_end=time_end, normal_mask_seq=normal_mask_seq)



def compute_seq_mse(x_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.mean((x_hat - x) ** 2, axis=(1, 2))



def estimate_tau(
    tau_source: str,
    tau_quantile: float,
    seq_mse: np.ndarray,
    train_mse: Optional[np.ndarray],
) -> float:
    if tau_source == "train_idle" and train_mse is not None and len(train_mse) > 0:
        return float(np.quantile(train_mse, tau_quantile))
    return float(np.quantile(seq_mse, tau_quantile))



def smooth_min_normal_run(is_anom: np.ndarray, min_normal_seq_run: int) -> np.ndarray:
    if min_normal_seq_run <= 1:
        return is_anom
    flags_normal = (~is_anom).astype(int)
    cleaned_normal = flags_normal.copy()
    run_start = None
    for i, f in enumerate(flags_normal):
        if f == 1 and run_start is None:
            run_start = i
        end_of_run = run_start is not None and ((f == 0) or (i == len(flags_normal) - 1))
        if end_of_run:
            run_end = i - 1 if f == 0 else i
            run_len = run_end - run_start + 1
            if run_len < min_normal_seq_run:
                cleaned_normal[run_start:run_end + 1] = 0
            run_start = None
    return cleaned_normal == 0



def smooth_min_anomaly_run(is_anom: np.ndarray, min_anom_seq_run: int) -> np.ndarray:
    if min_anom_seq_run <= 1:
        return is_anom
    flags_anom = is_anom.astype(int)
    cleaned_anom = flags_anom.copy()
    run_start = None
    for i, f in enumerate(flags_anom):
        if f == 1 and run_start is None:
            run_start = i
        end_of_run = run_start is not None and ((f == 0) or (i == len(flags_anom) - 1))
        if end_of_run:
            run_end = i - 1 if f == 0 else i
            run_len = run_end - run_start + 1
            if run_len < min_anom_seq_run:
                cleaned_anom[run_start:run_end + 1] = 0
            run_start = None
    return cleaned_anom == 1



def apply_confirmed_idle_state_machine(
    df_res: pd.DataFrame,
    idx_end: np.ndarray,
    time_end: np.ndarray,
    is_anom: np.ndarray,
    df_temp: pd.DataFrame,
    idle_confirm_minutes: int,
) -> StateMachineResult:
    is_normal_seq_raw = ~is_anom

    score_df = pd.DataFrame(
        {
            "time_end": pd.to_datetime(time_end),
            "is_normal_raw": is_normal_seq_raw.astype(int),
        }
    ).sort_values("time_end").reset_index(drop=True)

    confirm_delta = pd.Timedelta(minutes=idle_confirm_minutes)
    confirmed_idle = np.zeros(len(score_df), dtype=bool)
    idle_mode_active = False
    normal_run_start = None

    times_arr = pd.to_datetime(score_df["time_end"]).values
    raw_normal_arr = score_df["is_normal_raw"].values.astype(bool)

    for i in range(len(score_df)):
        t = pd.to_datetime(times_arr[i])
        is_normal_now = bool(raw_normal_arr[i])

        if idle_mode_active:
            if is_normal_now:
                confirmed_idle[i] = True
            else:
                confirmed_idle[i] = False
                idle_mode_active = False
                normal_run_start = None
        else:
            if is_normal_now:
                if normal_run_start is None:
                    normal_run_start = t
                run_duration = t - normal_run_start
                if run_duration >= confirm_delta:
                    idle_mode_active = True
                    confirmed_idle[i] = True
                else:
                    confirmed_idle[i] = False
            else:
                confirmed_idle[i] = False
                normal_run_start = None

    per_resampled_idle = pd.Series(False, index=df_res.index)
    idle_end_idxs = idx_end[confirmed_idle]
    if len(idle_end_idxs) > 0:
        per_resampled_idle.iloc[idle_end_idxs] = True

    per_sample_idle = pd.Series(False, index=df_temp.index)
    if per_resampled_idle.any():
        df_res_idle = df_res.loc[per_resampled_idle.values, ["time"]].copy().sort_values("time")
        df_orig_times = (
            df_temp[["time"]]
            .copy()
            .dropna()
            .astype({"time": "datetime64[ns]"})
            .reset_index()
            .rename(columns={"index": "orig_idx"})
            .sort_values("time")
        )
        mapped = pd.merge_asof(df_res_idle, df_orig_times, on="time", direction="backward")
        mapped = mapped.dropna(subset=["orig_idx"])
        mapped_idxs = mapped["orig_idx"].astype(int).unique().tolist()
        per_sample_idle.iloc[mapped_idxs] = True

    return StateMachineResult(
        confirmed_idle=confirmed_idle,
        anomaly_final=(~confirmed_idle),
        idle_samples_mapped=int(per_sample_idle.sum()),
    )



def _readings_to_dataframe(readings: List[dict]) -> pd.DataFrame:
    dataset = {
        "installation": ["NONE"] * len(readings),
        "device": [d.get("deviceName") for d in readings],
        "measure": [d.get("resourceName") for d in readings],
        "time": [d.get("origin") for d in readings],
        "unit": ["NONE"] * len(readings),
        "value_double": [d.get("value") for d in readings],
        "value_bigint": [d.get("value") for d in readings],
        "value_boolean": [d.get("value") for d in readings],
        "value_varchar": [d.get("value") for d in readings],
        "source_file": ["NONE"] * len(readings),
        "device_id": [d.get("deviceName") for d in readings],
    }
    return pd.DataFrame(dataset)


# ============================================================
# Public model class
# ============================================================


class HotModel:
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.cfg = config or make_default_config()
        self.ds = DataSource.from_paths(
            self.cfg.METADATA_PATH,
            self.cfg.COMBINED_CSV_PATH,
            self.cfg.TFLITE_MODEL_PATH,
        )
        metadata_obj = self.ds.read_json(self.cfg.METADATA_PATH)
        self.name_to_id = build_name_to_id(metadata_obj, self.cfg.HEATING_DEVICE_NAMES)
        self.device_id_to_name = {v: k for k, v in self.name_to_id.items()}
        self.device_ids = set(self.name_to_id.values())
        self.runner = TFLiteRunner(model_path=self.cfg.TFLITE_MODEL_PATH)
        self.calibrations: Dict[str, DeviceCalibration] = {}
        self._try_build_historical_calibration()

    def _try_build_historical_calibration(self) -> None:
        try:
            raw_hist = self.ds.read_csv(self.cfg.COMBINED_CSV_PATH, low_memory=False)
            if "device" not in raw_hist.columns:
                return

            raw_hist = raw_hist[raw_hist["device"].isin(self.device_ids)].copy()
            for device_name, device_id in self.name_to_id.items():
                if raw_hist[raw_hist["device"] == device_id].empty:
                    continue

                feat_res = build_device_features(raw_hist, device_id, self.cfg)
                windows = self.cfg.IDLE_TRAIN_WINDOWS.get(
                    device_name,
                    self.cfg.IDLE_TRAIN_WINDOWS.get("__default__"),
                )
                prep = resample_interpolate_and_scale(
                    feat_res.df_temp,
                    feat_res.feat_cols,
                    self.cfg.RESAMPLE_RULE,
                    scaler=None,
                    idle_train_windows=windows,
                )
                seq = build_sequences(
                    prep.x_scaled,
                    self.cfg.SEQ_LEN,
                    self.cfg.STEP,
                    prep.df_res,
                    prep.row_mask,
                )
                x_hat = self.runner.reconstruct(seq.x_all, self.cfg.BATCH)
                seq_mse = compute_seq_mse(x_hat, seq.x_all)

                train_mse = None
                if seq.normal_mask_seq is not None and seq.normal_mask_seq.any():
                    x_normal = seq.x_all[seq.normal_mask_seq]
                    x_norm_hat = self.runner.reconstruct(x_normal, self.cfg.BATCH)
                    train_mse = compute_seq_mse(x_norm_hat, x_normal)

                tau = estimate_tau(
                    self.cfg.TAU_QUANTILE,
                    self.cfg.TAU_SOURCE,
                    seq_mse,
                    train_mse,
                )
                self.calibrations[device_id] = DeviceCalibration(
                    scaler=prep.scaler,
                    tau=tau,
                    feat_cols=feat_res.feat_cols,
                )
        except Exception:
            self.calibrations = {}

    def _fallback_live_calibration(self, device_id: str, feat_res: DeviceFeatureBuildResult) -> DeviceCalibration:
        prep = resample_interpolate_and_scale(
            feat_res.df_temp,
            feat_res.feat_cols,
            self.cfg.RESAMPLE_RULE,
            scaler=None,
            idle_train_windows=None,
        )
        seq = build_sequences(prep.x_scaled, self.cfg.SEQ_LEN, self.cfg.STEP, prep.df_res, prep.row_mask)
        x_hat = self.runner.reconstruct(seq.x_all, self.cfg.BATCH)
        seq_mse = compute_seq_mse(x_hat, seq.x_all)
        tau = float(np.quantile(seq_mse, self.cfg.TAU_QUANTILE))
        scaler = RobustScaler().fit(prep.df_res[feat_res.feat_cols].values)
        calib = DeviceCalibration(scaler=scaler, tau=tau, feat_cols=feat_res.feat_cols)
        self.calibrations[device_id] = calib
        return calib

    def _infer_one_device(self, raw_live: pd.DataFrame, device_name: str, device_id: str) -> DeviceResult:
        feat_res = build_device_features(raw_live, device_id, self.cfg)
        calib = self.calibrations.get(device_id)
        if calib is None:
            calib = self._fallback_live_calibration(device_id, feat_res)

        prep = resample_interpolate_and_scale(
            feat_res.df_temp,
            calib.feat_cols,
            self.cfg.RESAMPLE_RULE,
            scaler=calib.scaler,
            idle_train_windows=None,
        )
        seq = build_sequences(prep.x_scaled, self.cfg.SEQ_LEN, self.cfg.STEP, prep.df_res, prep.row_mask)
        x_hat = self.runner.reconstruct(seq.x_all, self.cfg.BATCH)
        seq_mse = compute_seq_mse(x_hat, seq.x_all)
        print(f"[DEBUG] device={device_name} | tau={calib.tau:.6f} | latest_score={seq_mse[-1]:.6f}")
        is_anom = seq_mse > calib.tau
        is_anom = smooth_min_normal_run(is_anom, self.cfg.MIN_NORMAL_SEQ_RUN)
        is_anom = smooth_min_anomaly_run(is_anom, self.cfg.MIN_ANOM_SEQ_RUN)

        sm = apply_confirmed_idle_state_machine(
            prep.df_res,
            seq.idx_end,
            seq.time_end,
            is_anom,
            feat_res.df_temp,
            self.cfg.IDLE_CONFIRM_MINUTES,
        )

        latest_idle = bool(sm.confirmed_idle[-1]) if len(sm.confirmed_idle) else False
        status = "idle" if latest_idle else "no idle detected"
        timestamp = pd.to_datetime(feat_res.df_temp["time"].iloc[-1])

        return DeviceResult(
            device=device_name,
            device_id=device_id,
            timestamp=timestamp,
            status=status,
            tau=calib.tau,
            latest_score=float(seq_mse[-1]),
            anomaly_sequences=int(sm.anomaly_final.sum()),
            idle_samples_mapped=sm.idle_samples_mapped,
        )

    def infer(self, readings: List[dict]) -> List[DeviceResult]:
        raw_live = _readings_to_dataframe(readings)
        if "device" not in raw_live.columns:
            return []

        raw_live["device"] = raw_live["device"].map(lambda x: self.name_to_id.get(x, x))
        raw_live = raw_live[raw_live["device"].isin(self.device_ids)].copy()
        if raw_live.empty:
            return []

        results: List[DeviceResult] = []
        for device_name, device_id in self.name_to_id.items():
            if raw_live[raw_live["device"] == device_id].empty:
                continue
            try:
                result = self._infer_one_device(raw_live, device_name, device_id)
                print(f"{result.device}:")
                print(f"{result.timestamp}, {result.status}")
                results.append(result)
            except Exception:
                continue
        return results

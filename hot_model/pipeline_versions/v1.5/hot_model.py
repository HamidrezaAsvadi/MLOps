# hot_model_all_in_one.py
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# sklearn is used in preprocessing (RobustScaler)
from sklearn.preprocessing import RobustScaler

# tensorflow lite interpreter (your current runner uses TF)
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ============================================================
# config.py  (merged)
# ============================================================

@dataclass
class InferenceConfig:
    # -------- Output controls --------
    QUIET: bool = True               # True => suppress all debug/info/summary prints
    PRINT_IDLE_HEADER: bool = False  # True => prints "[device] IDLE OUTPUT:" header
    PRINT_NO_IDLE: bool = True      # True => prints "(no idle detected)" when idle_count==0
    DEBUG_EXCEPTIONS: bool = False  # True => print exception text even if QUIET

    # -------- Core settings --------
    HEATING_DEVICE_NAMES: List[str] = field(
        default_factory=lambda: ["Frontcooking Grill Left", "Frontcooking Grill Right"]
    )
    MODE: str = "idle"

    RESAMPLE_RULE: str = "10S"
    SEQ_LEN: int = 30
    STEP: int = 1
    BATCH: int = 256

    # Thresholding
    TAU = 0.95
    TAU_SOURCE = "train_idle"   # "train_idle" or "global"
    TAU_QUANTILE = 0.95

    # Smoothing
    MIN_NORMAL_SEQ_RUN: int = 2

    # Feature scaling (must match training)
    STATE_FEATURE_SCALE: float = 0.3

    # -------- Paths --------
    METADATA_PATH: str = ""
    COMBINED_CSV_PATH: str = ""
    TFLITE_MODEL_PATH: str = ""

    # For robust default pathing relative to this file
    SCRIPT_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    # Training windows used for scaler/tau when TAU_SOURCE="train_idle"
    IDLE_TRAIN_WINDOWS: Dict[str, Optional[List[Tuple[str, str]]]] = field(
        default_factory=lambda: {
            "Frontcooking Grill Left": [
                ("2025-04-01 15:00:00", "2025-04-01 17:00:00"),
                ("2025-04-03 16:30:00", "2025-04-03 17:30:00"),
                ("2025-04-04 16:00:00", "2025-04-04 17:45:00"),
                ("2025-04-05 16:00:00", "2025-04-05 18:00:00"),
                ("2025-04-06 16:00:00", "2025-04-06 18:00:00"),
            ],
            "__default__": None,
        }
    )


def make_default_config() -> InferenceConfig:
    """
    Creates a config with manually specified paths (relative to project root).
    """
    cfg = InferenceConfig()
    ROOT = cfg.SCRIPT_DIR.parents[0]
    cfg.METADATA_PATH = str(ROOT / "assets" / "metadata.json")
    cfg.COMBINED_CSV_PATH = str(ROOT / "assets" / "combined_dataset_ice_on.csv")
    cfg.TFLITE_MODEL_PATH = str(ROOT / "assets" / "hot_model.v1.tflite")
    return cfg


# ============================================================
# io_sources.py (merged)
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
            if u.scheme != "s3" or not u.netloc or not u.path:
                raise ValueError(f"Not a valid s3 uri: {path}")
            bucket = u.netloc
            key = u.path.lstrip("/")
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read().decode(encoding)
        else:
            with open(path, "r", encoding=encoding) as f:
                return f.read()

    def read_json(self, path: str):
        return json.loads(self.read_text(path))

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        if is_s3_uri(path):
            if self.s3_client is None:
                raise RuntimeError("S3 path provided but s3_client is None.")
            u = urlparse(path)
            if u.scheme != "s3" or not u.netloc or not u.path:
                raise ValueError(f"Not a valid s3 uri: {path}")
            bucket = u.netloc
            key = u.path.lstrip("/")
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            return pd.read_csv(obj["Body"], **kwargs)
        return pd.read_csv(path, **kwargs)


# ============================================================
# metadata.py (merged)
# ============================================================

def build_name_to_id(metadata_obj, heating_device_names: List[str]) -> Dict[str, str]:
    if not isinstance(metadata_obj, list):
        raise ValueError(
            "metadata.json must be a JSON list of device objects like "
            "[{'id':..., 'name':...}, ...]"
        )

    name_to_id_all = {
        d["name"]: d["id"]
        for d in metadata_obj
        if isinstance(d, dict) and "name" in d and "id" in d
    }

    missing = [n for n in heating_device_names if n not in name_to_id_all]
    if missing:
        raise ValueError(f"These device names were not found in metadata.json: {missing}")

    return {name: name_to_id_all[name] for name in heating_device_names}


# ============================================================
# features.py (merged, with a small bugfix)
# ============================================================

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

    # Optional relay-like signals
    for out_measure, col_name in [
        ("Output1", "Output1"),
        ("HeatRelay", "HeatRelay"),
        ("ReleState", "Output1"),   # map incoming ReleState into Output1 feature
    ]:
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

            if col_name in out.columns:
                out[col_name] = out[col_name].combine_first(
                    merged_asof[col_name] if col_name in merged_asof.columns else pd.Series(index=out.index, dtype=float)
                )
            else:
                out = out.merge(merged_asof, on=time_col, how="left")

            out[col_name] = out[col_name].fillna(0)
        else:
            if col_name not in out.columns:
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


# ============================================================
# preprocessing.py (merged)
# ============================================================

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


# ============================================================
# sequencing.py (merged)
# ============================================================

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


# ============================================================
# tflite_runner.py (merged)
# ============================================================

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

        # In your original file this was always None; keeping same behavior.
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


# ============================================================
# thresholding.py (merged)
# ============================================================

def compute_seq_mse(X_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.mean((X_hat - X) ** 2, axis=(1, 2))


def estimate_tau(
    tau_fixed: Optional[float],
    tau_source: str,
    tau_quantile: float,
    seq_mse: np.ndarray,
    train_mse: Optional[np.ndarray],
) -> float:
    if tau_fixed is not None:
        return float(tau_fixed)

    if tau_source == "train_idle" and train_mse is not None and len(train_mse) > 0:
        return float(np.quantile(train_mse, tau_quantile))

    return float(np.quantile(seq_mse, tau_quantile))


def smooth_min_normal_run(is_anom: np.ndarray, min_normal_seq_run: int) -> np.ndarray:
    """
    Converts short normal runs into anomalies.
    Same logic as your script.
    """
    if min_normal_seq_run <= 1:
        return is_anom

    flags_normal = (~is_anom).astype(int)  # 1=normal, 0=anom
    cleaned_normal = flags_normal.copy()
    run_start = None

    for i, f in enumerate(flags_normal):
        if f and run_start is None:
            run_start = i

        is_last = (i == len(flags_normal) - 1)
        if (not f or is_last) and run_start is not None:
            run_end = i if not f else i
            length = run_end - run_start + 1
            if length < min_normal_seq_run:
                cleaned_normal[run_start:run_end + 1] = 0
            run_start = None

    return (cleaned_normal == 0)


# ============================================================
# mapping.py (merged)
# ============================================================

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


# ============================================================
# pipeline.py (merged)
# ============================================================

@dataclass
class DeviceResult:
    device: str
    device_id: str
    tau: float
    anomaly_sequences: int
    idle_samples_mapped: int


def _readings_to_dataframe(readings: List[dict]) -> pd.DataFrame:
    """
    Builds the same columns you used before, but ensures time column name matches pipeline.
    """
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
    return pd.DataFrame(data=dataset)


def run_pipeline(readings: List[dict]) -> List[DeviceResult]:
    cfg = make_default_config()

    # Data source (local or S3)
    ds = DataSource.from_paths(cfg.METADATA_PATH, cfg.COMBINED_CSV_PATH)

    # Load metadata and map names -> ids
    metadata_obj = ds.read_json(cfg.METADATA_PATH)
    name_to_id = build_name_to_id(metadata_obj, cfg.HEATING_DEVICE_NAMES)

    # Build raw dataset from readings
    raw_all = _readings_to_dataframe(readings)

    if "device" not in raw_all.columns:
        raise ValueError("Input data must contain a 'device' column")

    # IMPORTANT: map device names in readings -> device ids (so later filtering matches)
    # If readings already have ids, this keeps them.
    raw_all["device"] = raw_all["device"].map(lambda x: name_to_id.get(x, x))

    device_ids = list(name_to_id.values())
    raw_all = raw_all[raw_all["device"].isin(device_ids)].copy()

    # Load TFLite runner once
    runner = TFLiteRunner(model_path=cfg.TFLITE_MODEL_PATH, use_flex_if_found=True)

    results: List[DeviceResult] = []

    for device_name, device_id in name_to_id.items():
        try:
            # 1) Build per-device features on original timeline
            feat_res = build_device_features(
                raw_all=raw_all,
                device_id=device_id,
                state_feature_scale=cfg.STATE_FEATURE_SCALE,
                quiet=cfg.QUIET,
                device_name=device_name,
            )
            df_temp = feat_res.df_temp
            feat_cols = feat_res.feat_cols

            # 2) Resample/interpolate + RobustScaler
            dev_windows = cfg.IDLE_TRAIN_WINDOWS.get(device_name, cfg.IDLE_TRAIN_WINDOWS.get("__default__"))
            prep = resample_interpolate_and_scale(
                df_temp=df_temp,
                feat_cols=feat_cols,
                resample_rule=cfg.RESAMPLE_RULE,
                idle_train_windows=dev_windows,
                time_col="time",
            )

            # 3) Sequences
            seq = build_sequences(
                X_scaled=prep.X_scaled,
                seq_len=cfg.SEQ_LEN,
                step=cfg.STEP,
                row_mask=prep.row_mask,
            )

            # X_normal for tau estimation if available
            X_normal = None
            if seq.normal_mask_seq is not None and seq.normal_mask_seq.any():
                X_normal = seq.X_all[seq.normal_mask_seq]

            # 4) Reconstruct all sequences
            X_hat = runner.reconstruct(seq.X_all, cfg.BATCH)
            seq_mse = compute_seq_mse(X_hat, seq.X_all)
            #print(f"[DEBUG] {device_name}: n_rows={len(prep.df_res)}, n_seq={len(seq.X_all)}")
            
            # 5) Estimate tau
            train_mse = None
            if cfg.TAU is None and cfg.TAU_SOURCE == "train_idle" and X_normal is not None and len(X_normal) > 0:
                Xn_hat = runner.reconstruct(X_normal, cfg.BATCH)
                train_mse = compute_seq_mse(Xn_hat, X_normal)

            tau = estimate_tau(
                tau_fixed=cfg.TAU,
                tau_source=cfg.TAU_SOURCE,
                tau_quantile=cfg.TAU_QUANTILE,
                seq_mse=seq_mse,
                train_mse=train_mse,
            )

            is_anom = seq_mse > tau

            # 6) Smoothing: require minimum NORMAL sequence run
            is_anom = smooth_min_normal_run(is_anom, cfg.MIN_NORMAL_SEQ_RUN)
            #print(f"[DEBUG] {device_name}: tau={tau:.6f}, anomalies={int(np.sum(is_anom))}, normal={int(np.sum(~is_anom))}")
            print(f"{device_name}:")

            latest_anomaly = bool(is_anom[-1])
            timestamp = df_temp["time"].iloc[-1]

            if latest_anomaly:
                print(f"{timestamp}, no idle detected")
            else:
               print(f"{timestamp}, idle")
               
        except Exception as e:
            print(f"[Error] {device_name}: {e}")

    return results


# ============================================================
# hot_model.py (merged)
# ============================================================

class HotModel:
    def infer(self, readings: List[dict]) -> List[DeviceResult]:
        return run_pipeline(readings)


# Optional: quick manual test
if __name__ == "__main__":
    # Example shape only; replace with your real readings.
    readings = [
        {"deviceName": "Frontcooking Grill Left", "resourceName": "CabinetProbe", "origin": "2025-04-01 15:00:00", "value": 20.0},
    ]
    print(HotModel().infer(readings))
    
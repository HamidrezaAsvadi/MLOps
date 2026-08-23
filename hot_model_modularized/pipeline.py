from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from hot_model.config import InferenceConfig
from hot_model.io_sources import DataSource
from hot_model.metadata import build_name_to_id
from hot_model.features import build_device_features
from hot_model.preprocessing import resample_interpolate_and_scale
from hot_model.sequencing import build_sequences
from hot_model.tflite_runner import TFLiteRunner
from hot_model.thresholding import compute_seq_mse, estimate_tau, smooth_min_normal_run
from hot_model.mapping import map_normal_endpoints_to_original_samples
from hot_model.config import make_default_config

@dataclass
class DeviceResult:
    device: str
    device_id: str
    tau: float
    anomaly_sequences: int
    idle_samples_mapped: int


def run_pipeline(readings: list[dict]) -> List[DeviceResult]:
    dataset: dict ={
            "installation" : ["NONE"] * len(readings),
            "device" : [d['deviceName'] for d in readings],
            "measure" : [d['resourceName'] for d in readings],
            "time" : [d['origin'] for d in readings],
            "unit" : ["NONE"] * len(readings),
            "value_double" : [d['value'] for d in readings],
            "value_bigint" : [d['value'] for d in readings],
            "value_boolean" : [d['value'] for d in readings],
            "value_varchar" : [d['value'] for d in readings],
            "source_file" : ["NONE"] * len(readings),
            "device_id" : [d["deviceName"] for d in readings],
    }



    cfg = make_default_config()
    # Data source (local or S3)
    ds = DataSource.from_paths(cfg.METADATA_PATH, cfg.COMBINED_CSV_PATH)

    # Load metadata and map names -> ids
    metadata_obj = ds.read_json(cfg.METADATA_PATH)
    name_to_id = build_name_to_id(metadata_obj, cfg.HEATING_DEVICE_NAMES)

    # Load raw dataset and filter for these device IDs
    raw_all = pd.DataFrame(data=dataset)
    # raw_all = ds.read_csv(cfg.COMBINED_CSV_PATH, low_memory=False)
    if "device" not in raw_all.columns:
        raise ValueError("combined_dataset.csv must contain a 'device' column")

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
            X_hat = runner.reconstruct(seq.seq_all if hasattr(seq, "seq_all") else seq.X_all, cfg.BATCH)  # safe
            seq_mse = compute_seq_mse(X_hat, seq.X_all)

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

            # 7) Map normal endpoints back to original samples => idle points
            is_normal_seq = ~is_anom
            mapping = map_normal_endpoints_to_original_samples(
                df_res=prep.df_res,
                idx_end=seq.idx_end,
                is_normal_seq=is_normal_seq,
                df_temp=df_temp,
                time_col="time",
            )

            df_temp["is_idle"] = mapping.per_sample.values.astype(bool)

            # 8) Print ONLY "idle" (quiet mode behavior)
            idle_count = int(df_temp["is_idle"].sum())

            if cfg.PRINT_IDLE_HEADER and idle_count > 0:
                print(f"\n[{device_name}] IDLE OUTPUT:")

            if idle_count == 0:
                if cfg.PRINT_NO_IDLE and cfg.PRINT_IDLE_HEADER:
                    print("  (no idle detected)")
                elif cfg.PRINT_NO_IDLE and (not cfg.PRINT_IDLE_HEADER):
                    print("(no idle detected)")
            else:
                if cfg.PRINT_IDLE_HEADER:
                    for _ in range(idle_count):
                        print("  idle")
                else:
                    for _ in range(idle_count):
                        print("idle")
                    #This is for debugging a dataset
                    # idle_times = df_temp[df_temp["is_idle"] == True]["time"]
                    # for i in idle_times:
                    #     print(f"{i}")

            results.append(
                DeviceResult(
                    device=device_name,
                    device_id=device_id,
                    tau=float(tau),
                    anomaly_sequences=int(np.sum(is_anom)),
                    idle_samples_mapped=int(idle_count),
                )
            )

        except Exception as e:
            # In QUIET mode: keep output clean (only "idle")
            if (not cfg.QUIET) or cfg.DEBUG_EXCEPTIONS:
                print(f"[WARN] Could not run inference for {device_name}: {e}")
            continue

    return results

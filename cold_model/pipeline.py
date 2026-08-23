from cold_model import config
from cold_model.data_io import read_csv, map_device_ids
from cold_model.feature_build import pick_device, filter_device, build_aligned_table, build_defrost_exclusion
from cold_model.preprocessing import (
    select_feature_columns, resample_and_interpolate, scale_features,
    build_sequences, pick_normal_sequences
)
from cold_model.tflite_infer import tflite_predict_sequences
from cold_model.scoring import mse_per_sequence, compute_tau, anomaly_mask
from cold_model.postprocess import map_anomalies_to_regions
import pandas as pd
import logging

def run(readings: list[dict]) -> list[tuple]:
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
    df_all = pd.DataFrame(data=dataset)
    # df_all = read_csv(config.CSV_S3)

    try:
        df_all = map_device_ids(df_all, config.JSON_S3)

        device_name = pick_device(df_all, config.DEVICE_FILTERS)

        df = filter_device(df_all, device_name)

        out = build_aligned_table(df)

        defrost_windows, exclusion_mask = build_defrost_exclusion(
            out, config.IGNORE_FAILS_DURING_DEFROST, config.COOLDOWN_MIN
        )

        feat_cols = select_feature_columns(out)

        df_res = resample_and_interpolate(out, feat_cols, config.RESAMPLE_FREQ)

        # scaling
        X_scaled, _scaler = scale_features(df_res, feat_cols)

        # sequencing
        X_all, idx_end = build_sequences(X_scaled, config.SEQ_LEN, config.STEP)

        # normalization
        X_normal = pick_normal_sequences(
            df_res, idx_end, X_all, device_name, config.NORMAL_TRAIN_WINDOWS
        )

        if not config.USE_TFLITE_FOR_TEST:
            raise ValueError("USE_TFLITE_FOR_TEST must be True for this pipeline.")

        # predict
        X_hat = tflite_predict_sequences(config.COLD_MODEL_PATH, X_all, batch_size=config.TFLITE_BATCH)

        # mse (distance/error among predicted and actual)
        seq_mse = mse_per_sequence(X_hat, X_all)

        # tau (treshold)
        tau = compute_tau(seq_mse, config.VAL_ERROR_QUANTILE)

        is_anom = anomaly_mask(seq_mse, tau)

        # regions
        regions = map_anomalies_to_regions(
            out=out,
            df_res=df_res,
            idx_end=idx_end,
            is_anom=is_anom,
            exclusion_mask=exclusion_mask,
            ignore_defrost=config.IGNORE_FAILS_DURING_DEFROST,
            defrost_windows=defrost_windows,
        )

        return regions
    except Exception as e:
        # In QUIET mode: keep output clean (only "idle")
        logging.error(e)

        return []
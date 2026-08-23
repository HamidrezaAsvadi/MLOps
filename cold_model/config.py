DEVICE_FILTERS = [
    "64fec30e-0d4a-4c49-b8ec-c998d6c4866b",
    "ef818899-5015-406f-a25a-07ea9b40ba42"
]

IGNORE_FAILS_DURING_DEFROST = True
COOLDOWN_MIN = 15

SEQ_LEN = 30
STEP = 1
VAL_SIZE = 0.2
RANDOM_SEED = 1
VAL_ERROR_QUANTILE = 0.9995

USE_TFLITE_FOR_TEST = True
COLD_MODEL_PATH = "assets/cold_model.v1.tflite"
HOT_MODEL_PATH = "assets/hot_model.v1.tflite"
TFLITE_BATCH = 256

CSV_S3  = "./assets/combined_dataset.csv"
JSON_S3 = "./assets/metadata.json"

RESAMPLE_FREQ = "30s"

NORMAL_TRAIN_WINDOWS = {
    "64fec30e-0d4a-4c49-b8ec-c998d6c4866b": [
        ("2025-04-01 00:10:00", "2025-04-01 16:00:00"),
        ("2025-04-03 00:10:00", "2025-04-06 12:00:00"),
        ("2025-04-07 00:00:00", "2025-04-09 00:00:00"),
        ("2025-04-11 10:00:00", "2025-04-12 00:00:00"),
        ("2025-04-13 19:00:00", "2025-04-14 00:00:00"),
    ],
    "__default__": None,
}

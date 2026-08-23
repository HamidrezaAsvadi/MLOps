import pandas as pd

REQUIRED_COLS = {"time", "device", "measure", "value_double", "value_bigint", "value_boolean"}

def read_csv(csv_path: str) -> pd.DataFrame:
    df_all = pd.read_csv(csv_path, low_memory=False)
    missing = REQUIRED_COLS - set(df_all.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df_all

def map_device_ids(df_all: pd.DataFrame, json_path: str) -> pd.DataFrame:
    # best-effort mapping; if fails, return df as-is
    try:
        meta = pd.read_json(json_path)

        if isinstance(meta, pd.DataFrame) and "id" in meta.columns:
            id_to_name = dict(zip(meta["id"].astype(str), meta.get("name", pd.Series([None]*len(meta))).astype(object)))
        else:
            id_to_name = {}
            try:
                for item in meta:
                    if isinstance(item, dict) and "id" in item:
                        id_to_name[str(item["id"])] = item.get("name")
            except Exception:
                id_to_name = {}

        df_all = df_all.copy()
        df_all["device"] = df_all["device"].astype(str).map(id_to_name).fillna(df_all["device"].astype(str))
        return df_all
    except Exception:
        return df_all

from typing import Dict, List


def build_name_to_id(metadata_obj, heating_device_names: List[str]) -> Dict[str, str]:
    if not isinstance(metadata_obj, list):
        raise ValueError("metadata.json must be a JSON list of device objects like [{'id':..., 'name':...}, ...]")

    name_to_id_all = {
        d["name"]: d["id"]
        for d in metadata_obj
        if isinstance(d, dict) and "name" in d and "id" in d
    }

    missing = [n for n in heating_device_names if n not in name_to_id_all]
    if missing:
        raise ValueError(f"These device names were not found in metadata.json: {missing}")

    return {name: name_to_id_all[name] for name in heating_device_names}

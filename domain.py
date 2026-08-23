import env

MAPPINGS = {
    "Evco-EVJ705": {
        "model": "HOT",
        "mappings": {
            "CabinetProbe": {
                "name": "Temperature",
                "type": lambda x: float(x),
            },
            "TemperatureSetPoint": {
                "name": "Setpoint",
                "type": lambda x: float(x),
            },
            "State": {
                "name": "State",
                "type": lambda x: int(x),
            },
            "Output1": {
                "name": "ReleState",
                "type": lambda x: int(x),
            },
        }
    },
    "Evco-EVJ705_2": {
        "model": "HOT",
        "mappings": {
            "CabinetProbe": {
                "name": "Temperature",
                "type": lambda x: float(x),
            },
            "TemperatureSetPoint": {
                "name": "Setpoint",
                "type": lambda x: float(x),
            },
            "State": {
                "name": "State",
                "type": lambda x: int(x),
            },
            "Output1": {
                "name": "ReleState",
                "type": lambda x: int(x),
            },
        }
    },
    "Evco-EV3244N9EWHXX1": {
        "model": "COLD",
        "mappings": {
            "ProbeR": {
                "name": "Temperature",
                "type": lambda x: float(x),
            },
            "SetPointR": {
                "name": "Setpoint",
                "type": lambda x: float(x),
            },
            "State": {
                "name": "State",
                "type": lambda x: int(x),
            },
            "Defrost": {
                "name": "Defrost",
                "type": lambda x: 1 if x.lower() == 'true' or x == '1' else 0,
            },
        }
    }
}

def map_reading(reading: dict) -> dict:
    mappings = MAPPINGS[reading["profileName"]]["mappings"]
    typing = mappings[reading["resourceName"]]
    if not typing:
        return reading

    reading["resourceName"] = typing["name"]
    reading["value"] = typing["type"](reading["value"])
    return reading

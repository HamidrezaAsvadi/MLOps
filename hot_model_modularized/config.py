from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class InferenceConfig:
    # -------- Output controls --------
    QUIET: bool = True               # True => suppress all debug/info/summary prints
    PRINT_IDLE_HEADER: bool = False  # True => prints "[device] IDLE OUTPUT:" header
    PRINT_NO_IDLE: bool = False      # True => prints "(no idle detected)" when idle_count==0
    DEBUG_EXCEPTIONS: bool = False   # True => print exception text even if QUIET

    # -------- Core settings --------
    HEATING_DEVICE_NAMES: List[str] = field(default_factory=lambda: ["Frontcooking Grill Left", "Frontcooking Grill Right"])
    MODE: str = "idle"

    RESAMPLE_RULE: str = "10S"
    SEQ_LEN: int = 30
    STEP: int = 1
    BATCH: int = 256

    # Thresholding
    TAU: Optional[float] = None
    TAU_SOURCE: str = "train_idle"   # "train_idle" or "global"
    TAU_QUANTILE: float = 0.95

    # Smoothing
    MIN_NORMAL_SEQ_RUN: int = 5

    # Feature scaling (must match training)
    STATE_FEATURE_SCALE: float = 0.3

    # -------- Paths --------
    # Keep them as strings so they can be either local paths or s3:// URIs
    METADATA_PATH: str = ""
    COMBINED_CSV_PATH: str = ""
    TFLITE_MODEL_PATH: str = ""

    # For robust default pathing relative to script location
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
    Creates a config with manually specified paths.
    project_root is ignored (kept only for CLI compatibility).
    """

    cfg = InferenceConfig()
    ROOT = cfg.SCRIPT_DIR.parents[0]
    cfg.METADATA_PATH = str(ROOT / "assets" / "metadata.json")
    cfg.COMBINED_CSV_PATH = str(ROOT / "assets" / "combined_dataset_ice_on.csv")
    cfg.TFLITE_MODEL_PATH = str(ROOT / "assets" / "hot_model.v1.tflite")

    return cfg

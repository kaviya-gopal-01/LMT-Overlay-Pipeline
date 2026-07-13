"""
config.py

Central configuration for the LMT overlay pipeline.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_VIDEO_DIR: Path = Path(os.environ.get("LMT_INPUT_DIR", "./input_videos")).resolve()
DATABASE_PATH: Path = Path(os.environ.get("LMT_DATABASE_PATH", "./tracking.sqlite")).resolve()
OUTPUT_VIDEO_DIR: Path = Path(os.environ.get("LMT_OUTPUT_DIR", "./output")).resolve()

# ---------------------------------------------------------------------------
# Video file naming / frame mapping
# ---------------------------------------------------------------------------
VIDEO_FILENAME_PATTERN = re.compile(r"_t(\d+)\.mp4$", re.IGNORECASE)
VIDEO_GLOB_PATTERN = "video_noOverlay_t*.mp4"

FRAME_CONVERSION: int = 2
RAW_VIDEO_FPS: float = 15.0

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DETECTION_TABLE = "DETECTION"

DETECTION_COLUMNS = (
    "FRAMENUMBER", "ANIMALID", "MASS_X", "MASS_Y", "MASS_Z",
    "FRONT_X", "FRONT_Y", "FRONT_Z", "BACK_X", "BACK_Y", "BACK_Z",
    "REARING", "LOOK_UP", "LOOK_DOWN", "DATA",
)

DB_QUERY_CHUNK_FRAMES: int = 5_000

# ---------------------------------------------------------------------------
# Mask decoding (see mask_decoder.py)
# ---------------------------------------------------------------------------
MASK_BYTE_SEPARATOR = ":"
MASK_FOREGROUND_VALUE = 1
MASK_STRICT_SIZE_VALIDATION: bool = True

# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------
ANIMAL_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    1: (0, 0, 255),      # Red
    2: (0, 200, 0),      # Green
    3: (255, 0, 255),      # Purple
    4: (0, 220, 255),    # Yellow
}
ANIMAL_COLOR_FALLBACK_BGR: Tuple[int, int, int] = (255, 255, 255) # White

MASK_ALPHA: float = 0.45    # 0 = invisible, 1 = fully opaque fill
DRAW_MASK_OUTLINE: bool = True
MASK_OUTLINE_THICKNESS: int = 1

DRAW_POSE_MARKERS: bool = True
POSE_MARKER_RADIUS: int = 3
POSE_MARKER_THICKNESS: int = -1

DRAW_POSTURE_LABEL: bool = True
POSTURE_LABEL_FONT_SCALE: float = 0.4
POSTURE_LABEL_THICKNESS: int = 1

# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------
OUTPUT_VIDEO_SUFFIX: str = "_corrected"
OUTPUT_FOURCC: str = "mp4v"
OUTPUT_FPS: float = RAW_VIDEO_FPS

# ---------------------------------------------------------------------------
# Logging / performance
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LMT_LOG_LEVEL", "INFO")
PROGRESS_BAR: bool = True


def validate_config() -> None:
    """Fail fast at startup with a clear message instead of a cryptic
    error deep inside video_processor."""
    errors = []

    if not INPUT_VIDEO_DIR.exists():
        errors.append(f"INPUT_VIDEO_DIR does not exist: {INPUT_VIDEO_DIR}")
    if not DATABASE_PATH.exists():
        errors.append(f"DATABASE_PATH does not exist: {DATABASE_PATH}")
    if FRAME_CONVERSION <= 0:
        errors.append(f"FRAME_CONVERSION must be positive, got {FRAME_CONVERSION}")
    if not (0.0 <= MASK_ALPHA <= 1.0):
        errors.append(f"MASK_ALPHA must be in [0, 1], got {MASK_ALPHA}")
    if len(OUTPUT_FOURCC) != 4:
        errors.append(f"OUTPUT_FOURCC must be a 4-character codec code, got {OUTPUT_FOURCC!r}")

    if errors:
        raise ValueError("Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
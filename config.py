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
# INPUT_VIDEO_DIR is kept for backward compatibility with
# video_processor.discover_videos(), which globs a single directory. The
# GUI-driven flow in main.py does NOT use it -- gui_selector.py returns an
# explicit list of video file paths (which may span multiple folders),
# and those are passed to video_processor.discover_videos_from_paths()
# instead. INPUT_VIDEO_DIR is therefore optional now; see validate_config()
# below, which no longer requires it to exist.
INPUT_VIDEO_DIR: Path = Path(os.environ.get("LMT_INPUT_DIR", "./input_videos")).resolve()
DATABASE_PATH: Path = Path(os.environ.get("LMT_DATABASE_PATH", "./tracking.sqlite")).resolve()

# Base directory under which each run creates its own timestamped
# subfolder (e.g. "./output/2026-07-20_17-34-12/"). main.py's GUI flow
# reassigns this module attribute to that timestamped subfolder before
# processing starts -- see main.py's _make_session_output_dir().
OUTPUT_VIDEO_DIR: Path = Path(os.environ.get("LMT_OUTPUT_DIR", "./output")).resolve()

# ---------------------------------------------------------------------------
# Video file naming / frame mapping
# ---------------------------------------------------------------------------
VIDEO_FILENAME_PATTERN = re.compile(r"_t(\d+)\.mp4$", re.IGNORECASE)
# Broad on purpose: glob has no digit-class matching, so this just narrows
# candidates to "anything ending in _t<something>.mp4". VIDEO_FILENAME_PATTERN
# (a real regex, checked per-file in probe_video) is the actual source of
# truth for "is this a valid raw video filename" -- it rejects anything
# where <something> isn't purely digits. This lets discovery accept any
# prefix ("video_noOverlay_t18024.mp4", "cam3_raw_noOverlay_t25000.mp4",
# "anything_t18024.mp4"), matching the fact that output_path already
# renames all of them to a uniform "video_t<N>.mp4" regardless of input prefix.
VIDEO_GLOB_PATTERN = "*_t*.mp4"

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
# Mask decoding
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
    3: (255, 0, 0),      # Blue
    4: (0, 220, 255),    # Yellow
}
ANIMAL_COLOR_FALLBACK_BGR: Tuple[int, int, int] = (255, 0, 255)

MASK_ALPHA: float = 0.45
DRAW_MASK_OUTLINE: bool = True
MASK_OUTLINE_THICKNESS: int = 1

DRAW_POSE_MARKERS: bool = True
POSE_MARKER_RADIUS: int = 3
POSE_MARKER_THICKNESS: int = -1

DRAW_POSTURE_LABEL: bool = True
POSTURE_LABEL_FONT_SCALE: float = 0.4
POSTURE_LABEL_THICKNESS: int = 1

DRAW_GLOBAL_FRAME_NUMBER: bool = True
FRAME_NUMBER_COLOR_BGR: Tuple[int, int, int] = (255, 255, 255)
FRAME_NUMBER_FONT_SCALE: float = 0.5
FRAME_NUMBER_THICKNESS: int = 1
FRAME_NUMBER_MARGIN_PX: int = 10

# ---------------------------------------------------------------------------
# Video output
# ---------------------------------------------------------------------------
# Output filenames are built as "{OUTPUT_VIDEO_PREFIX}_t{start_frame}{ext}",
# e.g. "video_t18024.mp4" -- independent of the raw input filename's
# prefix, and independent of which folder that input file lived in.
OUTPUT_VIDEO_PREFIX: str = "video"
OUTPUT_FOURCC: str = "mp4v"
OUTPUT_FPS: float = RAW_VIDEO_FPS

# ---------------------------------------------------------------------------
# GUI file selection
# ---------------------------------------------------------------------------
# Where gui_selector.py persists the last-used database path and last
# video directory across runs. Stored next to the project code rather
# than in the user's home directory, so it travels with the checkout and
# is easy to find/delete.
SETTINGS_FILE_PATH: Path = Path(__file__).resolve().parent / "settings.json"

# strftime format for each run's output subfolder name, e.g.
# "2026-07-20_17-34-12". Filesystem-safe on Windows/macOS/Linux (no
# colons), sorts chronologically as plain text.
OUTPUT_TIMESTAMP_FORMAT: str = "%Y-%m-%d_%H-%M-%S"

# ---------------------------------------------------------------------------
# Logging / performance
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LMT_LOG_LEVEL", "INFO")
PROGRESS_BAR: bool = True


def validate_config() -> None:
    """
    Sanity-check configuration at startup and fail fast with a clear
    message rather than surfacing a cryptic error deep in video_processor.

    INPUT_VIDEO_DIR is intentionally NOT validated here: the GUI-driven
    flow (main.py's default) supplies an explicit list of video file
    paths rather than a directory to glob, so INPUT_VIDEO_DIR may be
    unset or nonexistent and that's fine. It only matters if you call
    video_processor.discover_videos() directly with a folder yourself.
    """
    errors = []

    if not DATABASE_PATH.exists():
        errors.append(f"DATABASE_PATH does not exist: {DATABASE_PATH}")
    if FRAME_CONVERSION <= 0:
        errors.append(f"FRAME_CONVERSION must be positive, got {FRAME_CONVERSION}")
    if not (0.0 <= MASK_ALPHA <= 1.0):
        errors.append(f"MASK_ALPHA must be in [0, 1], got {MASK_ALPHA}")
    if len(OUTPUT_FOURCC) != 4:
        errors.append(f"OUTPUT_FOURCC must be a 4-character codec code, got {OUTPUT_FOURCC!r}")

    if errors:
        raise ValueError(
            "Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    OUTPUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
"""
database.py

SQLite access layer for the LMT tracking database. Treats the database as
read-only ground truth: connections are opened in SQLite's URI read-only
mode so the pipeline can never accidentally mutate the source of truth.

Provides chunked, indexed range queries over DETECTION so video_processor
can stream through a recording without loading the full detection
history into memory at once.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List

import config

logger = logging.getLogger(__name__)


class DetectionRow:
    """Lightweight, attribute-access wrapper around a single DETECTION row."""

    __slots__ = (
        "framenumber", "animal_id", "mass_x", "mass_y", "mass_z",
        "front_x", "front_y", "front_z", "back_x", "back_y", "back_z",
        "rearing", "look_up", "look_down", "data_xml",
    )

    def __init__(self, row: sqlite3.Row):
        self.framenumber: int = row["FRAMENUMBER"]
        self.animal_id: int = row["ANIMALID"]
        self.mass_x = row["MASS_X"]
        self.mass_y = row["MASS_Y"]
        self.mass_z = row["MASS_Z"]
        self.front_x = row["FRONT_X"]
        self.front_y = row["FRONT_Y"]
        self.front_z = row["FRONT_Z"]
        self.back_x = row["BACK_X"]
        self.back_y = row["BACK_Y"]
        self.back_z = row["BACK_Z"]
        self.rearing = bool(row["REARING"])
        self.look_up = bool(row["LOOK_UP"])
        self.look_down = bool(row["LOOK_DOWN"])
        self.data_xml: str = row["DATA"]

    def __repr__(self) -> str:  # pragma: no cover
        return f"DetectionRow(frame={self.framenumber}, animal={self.animal_id})"


def _readonly_uri(db_path: Path) -> str:
    posix_path = db_path.as_posix()
    return f"file:{posix_path}?mode=ro"


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a read-only connection to the LMT SQLite database. Read-only
    is enforced at the SQLite level, not just by convention."""
    path = Path(db_path) if db_path is not None else config.DATABASE_PATH
    uri = _readonly_uri(path)
    logger.debug("Opening SQLite connection (read-only): %s", uri)

    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def has_index_on_framenumber(conn: sqlite3.Connection) -> bool:
    """Check whether an index covering DETECTION.FRAMENUMBER already
    exists. Used only to emit a performance warning -- this module never
    creates indexes, since the connection is intentionally read-only."""
    cursor = conn.execute(f"PRAGMA index_list({config.DETECTION_TABLE})")
    for index_row in cursor.fetchall():
        index_name = index_row["name"]
        info_cursor = conn.execute(f"PRAGMA index_info({index_name})")
        columns = {col_row["name"] for col_row in info_cursor.fetchall()}
        if "FRAMENUMBER" in columns:
            return True
    return False


def warn_if_unindexed(conn: sqlite3.Connection) -> None:
    if not has_index_on_framenumber(conn):
        logger.warning(
            "DETECTION table has no index covering FRAMENUMBER. Range "
            "queries over a multi-hour recording will require a full "
            "table scan per chunk. Consider creating one out-of-band "
            "(outside this read-only pipeline) with:\n"
            "  CREATE INDEX idx_detection_framenumber ON DETECTION(FRAMENUMBER);"
        )


def fetch_detections_in_range(
    conn: sqlite3.Connection, start_frame: int, end_frame: int
) -> List[DetectionRow]:
    """Fetch DETECTION rows with FRAMENUMBER in [start_frame, end_frame),
    ordered by FRAMENUMBER then ANIMALID."""
    columns_sql = ", ".join(config.DETECTION_COLUMNS)
    query = (
        f"SELECT {columns_sql} FROM {config.DETECTION_TABLE} "
        "WHERE FRAMENUMBER >= ? AND FRAMENUMBER < ? "
        "ORDER BY FRAMENUMBER ASC, ANIMALID ASC"
    )
    cursor = conn.execute(query, (start_frame, end_frame))
    return [DetectionRow(row) for row in cursor.fetchall()]


def iter_detection_chunks(
    conn: sqlite3.Connection,
    start_frame: int,
    end_frame: int,
    chunk_size: int = config.DB_QUERY_CHUNK_FRAMES,
) -> Iterator[Dict[int, List[DetectionRow]]]:
    """Stream detections for [start_frame, end_frame) in fixed-size global
    frame-number chunks, pre-grouped by FRAMENUMBER for O(1) lookup while
    iterating video frames. Bounds peak memory to one chunk regardless of
    recording length."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    chunk_start = start_frame
    while chunk_start < end_frame:
        chunk_end = min(chunk_start + chunk_size, end_frame)
        rows = fetch_detections_in_range(conn, chunk_start, chunk_end)

        grouped: Dict[int, List[DetectionRow]] = {}
        for row in rows:
            grouped.setdefault(row.framenumber, []).append(row)

        yield grouped
        chunk_start = chunk_end


def get_global_frame_bounds(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return (min_framenumber, max_framenumber) across DETECTION. Used by
    main.py to sanity-check discovered videos overlap DB data."""
    cursor = conn.execute(
        f"SELECT MIN(FRAMENUMBER) AS min_f, MAX(FRAMENUMBER) AS max_f "
        f"FROM {config.DETECTION_TABLE}"
    )
    row = cursor.fetchone()
    if row is None or row["min_f"] is None:
        raise ValueError("DETECTION table is empty; nothing to render.")
    return int(row["min_f"]), int(row["max_f"])

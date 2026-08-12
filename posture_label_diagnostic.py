#!/usr/bin/env python3
"""
posture_label_diagnostic.py

Standalone diagnostic: compares DETECTION.REARING / DETECTION.LOOK_UP /
DETECTION.LOOK_DOWN against the corresponding isRearing / isLookingUp /
isLookingDown attributes embedded in each row's DATA XML blob, and
reports any discrepancies.

This exists to answer git issue #3: are the DETECTION posture *columns*
trustworthy, or should the overlay renderer read posture from the XML
DATA blob instead? This script only diagnoses -- it does not decide
which source the renderer should use, and it never modifies the
database. The connection is opened read-only via database.connect(),
exactly as the rest of the pipeline does, so running this script is
safe against a live/production tracking.sqlite.

Usage:
    python posture_label_diagnostic.py
        Opens a GUI file picker to select the SQLite database.

    python posture_label_diagnostic.py --db /path/to/tracking.sqlite
        Skips the picker; useful for scripted/headless runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import config
import database
import xml_parser

logger = logging.getLogger(__name__)


class Mismatch(NamedTuple):
    """A single posture-field disagreement between the DB column and the
    parsed XML attribute for one (frame, animal) detection."""

    framenumber: int
    animal_id: int
    field: str
    db_value: bool
    xml_value: bool


def _select_database_gui() -> Path:
    """
    Single-purpose GUI file picker for just the SQLite database.

    Deliberately not gui_selector.select_database_and_videos(): that
    function also prompts for raw video files, which this diagnostic
    has no use for. Tkinter is imported inside this function (same
    reasoning as gui_selector.py) so a missing/broken Tkinter
    installation only breaks the interactive path -- `--db` still works
    without Tkinter installed at all.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # we only want the native dialog, not an empty window
    root.attributes("-topmost", True)
    try:
        db_path_str = filedialog.askopenfilename(
            title="Select the LMT SQLite database to diagnose",
            filetypes=[("SQLite database", "*.sqlite *.db"), ("All files", "*.*")],
            parent=root,
        )
    finally:
        root.destroy()

    if not db_path_str:
        raise SystemExit("No database file was selected (dialog was cancelled).")
    return Path(db_path_str).resolve()


def compare_posture_labels(conn) -> Tuple[List[Mismatch], int, int]:
    """
    Stream every DETECTION row in chunks (same chunk-size strategy as
    video_processor.process_single_video, so a multi-hour recording's
    worth of rows is never loaded into memory at once), parse each row's
    DATA XML, and compare REARING/LOOK_UP/LOOK_DOWN against
    isRearing/isLookingUp/isLookingDown.

    A row whose DATA XML can't be parsed at all (missing, malformed, or
    missing the <DATA> element -- see xml_parser.XmlParseError) is
    logged and excluded from the comparison entirely, since there is no
    XML-side value to compare against; it is counted separately in the
    returned rows_malformed total rather than silently dropped.

    Returns (mismatches, rows_checked, rows_malformed).
    """
    min_frame, max_frame = database.get_global_frame_bounds(conn)

    mismatches: List[Mismatch] = []
    rows_checked = 0
    rows_malformed = 0

    chunk_size = config.DB_QUERY_CHUNK_FRAMES
    chunk_start = min_frame
    # max_frame is inclusive (see database.get_global_frame_bounds), so the
    # loop must run once more with an exclusive end past it to include it.
    while chunk_start <= max_frame:
        chunk_end = chunk_start + chunk_size
        rows = database.fetch_detections_in_range(conn, chunk_start, chunk_end)

        for row in rows:
            try:
                parsed = xml_parser.parse_detection_xml(row.data_xml)
            except xml_parser.XmlParseError as exc:
                rows_malformed += 1
                logger.warning(
                    "frame=%d animal=%d: could not parse DATA XML (%s); "
                    "excluded from comparison.",
                    row.framenumber, row.animal_id, exc,
                )
                continue

            rows_checked += 1
            for field, db_value, xml_value in (
                ("REARING/isRearing", row.rearing, parsed.is_rearing),
                ("LOOK_UP/isLookingUp", row.look_up, parsed.is_looking_up),
                ("LOOK_DOWN/isLookingDown", row.look_down, parsed.is_looking_down),
            ):
                if db_value != xml_value:
                    mismatches.append(Mismatch(
                        framenumber=row.framenumber,
                        animal_id=row.animal_id,
                        field=field,
                        db_value=db_value,
                        xml_value=xml_value,
                    ))

        chunk_start = chunk_end

    return mismatches, rows_checked, rows_malformed


def _report(mismatches: List[Mismatch], rows_checked: int, rows_malformed: int) -> None:
    print(f"Rows compared: {rows_checked}")
    print(f"Rows with unparseable/missing DATA XML (excluded above): {rows_malformed}")
    print(f"Mismatches found: {len(mismatches)}")

    if not mismatches:
        print("\nNo discrepancies between DETECTION columns and DATA XML posture attributes.")
        return

    by_field: Dict[str, int] = {}
    for m in mismatches:
        by_field[m.field] = by_field.get(m.field, 0) + 1

    print("\nMismatches by field:")
    for field, count in sorted(by_field.items()):
        print(f"  {field}: {count}")

    shown = min(50, len(mismatches))
    print(f"\nFirst {shown} mismatch(es) -- frame, animal, field, DB value, XML value:")
    for m in mismatches[:shown]:
        print(
            f"  frame={m.framenumber:>8}  animal={m.animal_id:>2}  "
            f"{m.field:<24}  DB={str(m.db_value):<5}  XML={str(m.xml_value):<5}"
        )
    if len(mismatches) > shown:
        print(f"  ... and {len(mismatches) - shown} more.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DETECTION.REARING/LOOK_UP/LOOK_DOWN columns against "
            "the DATA XML blob's isRearing/isLookingUp/isLookingDown "
            "values. Read-only: never modifies the database."
        ),
    )
    parser.add_argument(
        "--db", type=Path, default=None,
        help="Path to the SQLite database. If omitted, a GUI file picker is shown.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    db_path = args.db.resolve() if args.db is not None else _select_database_gui()
    if not db_path.exists():
        logger.error("Database file does not exist: %s", db_path)
        return 1

    try:
        with database.connect(db_path) as conn:
            mismatches, rows_checked, rows_malformed = compare_posture_labels(conn)
    except ValueError as exc:
        # e.g. database.get_global_frame_bounds() on an empty DETECTION table
        logger.error("%s", exc)
        return 1

    _report(mismatches, rows_checked, rows_malformed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
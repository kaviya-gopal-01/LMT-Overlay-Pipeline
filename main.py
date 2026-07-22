#!/usr/bin/env python3
"""
main.py

CLI entrypoint for the LMT overlay pipeline.

Usage:
    python main.py
    python main.py --output-base-dir /data/results --no-progress --log-level DEBUG

On startup, this launches two file-picker dialogs (via gui_selector.py):
first to select the LMT SQLite database, then to select one or more raw
video files. Selected files may live anywhere on disk, in any folder
structure -- there is no assumption of a fixed project layout. Each run
writes its output into a fresh, uniquely timestamped subfolder under
--output-base-dir (default: ./output), so repeated runs never overwrite
each other's results.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import database
import gui_selector
import video_processor

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render corrected LMT overlays (from SQLite ground truth) onto raw videos.",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=config.OUTPUT_VIDEO_DIR,
        help=(
            "Base directory under which each run creates its own "
            f"timestamped output subfolder (default: {config.OUTPUT_VIDEO_DIR}). "
            "E.g. <base>/2026-07-20_17-34-12/video_t18024.mp4"
        ),
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bars (useful for non-interactive/log-file runs).",
    )
    parser.add_argument(
        "--log-level", default=config.LOG_LEVEL, choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help=f"Logging verbosity (default: {config.LOG_LEVEL})",
    )
    return parser.parse_args(argv)


def _configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _make_unique_output_dir(base_dir: Path) -> Path:
    """
    Build this run's output directory as <base_dir>/<timestamp>. If that
    exact folder somehow already exists (two runs started within the
    same second -- unlikely but not impossible), append a numeric suffix
    until a free name is found. This is what guarantees "never silently
    overwrite previous outputs": we never write into a folder that
    already exists.
    """
    timestamp = datetime.now().strftime(config.OUTPUT_TIMESTAMP_FORMAT)
    candidate = base_dir / timestamp
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = base_dir / f"{timestamp}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _sanity_check_frame_overlap(conn, videos: list[video_processor.VideoInfo]) -> None:
    """
    Warn (but don't abort) if the selected videos' global frame ranges
    don't overlap the DETECTION table's frame range at all -- usually
    means the wrong database was picked for these videos.
    """
    if not videos:
        return

    try:
        db_min, db_max = database.get_global_frame_bounds(conn)
    except ValueError as exc:
        logger.error("Cannot sanity-check frame overlap: %s", exc)
        return

    video_min = min(v.start_frame for v in videos)
    video_max = max(v.end_frame for v in videos)

    if video_max <= db_min or video_min >= db_max:
        logger.warning(
            "Selected videos span global frames [%d, %d) but DETECTION "
            "spans [%d, %d] -- these do not overlap at all. Double-check "
            "you selected the right database for these videos.",
            video_min, video_max, db_min, db_max,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _configure_logging(args.log_level)

    if args.no_progress:
        config.PROGRESS_BAR = False

    try:
        database_path, video_paths = gui_selector.select_database_and_videos()
    except ImportError as exc:
        logger.error(
            "Tkinter is not available (%s). On Debian/Ubuntu, install it with: "
            "sudo apt-get install python3-tk. On other platforms, reinstall "
            "Python with Tk support enabled.", exc,
        )
        return 1
    except gui_selector.SelectionCancelled as exc:
        logger.warning("Selection cancelled: %s Nothing was processed.", exc)
        return 1

    config.DATABASE_PATH = database_path
    config.OUTPUT_VIDEO_DIR = _make_unique_output_dir(args.output_base_dir.resolve())

    try:
        config.validate_config()
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Using database: %s", config.DATABASE_PATH)
    logger.info("Output directory for this run: %s", config.OUTPUT_VIDEO_DIR)

    videos = video_processor.discover_videos_from_paths(video_paths)
    if not videos:
        logger.error("None of the selected video files were usable; nothing to do.")
        return 1
    logger.info(
        "Processing %d video(s), global frame range [%d, %d)",
        len(videos), videos[0].start_frame, videos[-1].end_frame,
    )

    start_time = time.monotonic()
    try:
        with database.connect(config.DATABASE_PATH) as conn:
            _sanity_check_frame_overlap(conn, videos)
            video_processor.process_all_videos(conn, videos)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; partially processed videos remain in %s", config.OUTPUT_VIDEO_DIR)
        return 130
    except Exception:
        logger.exception("Unexpected error during processing.")
        return 1

    elapsed = time.monotonic() - start_time
    logger.info("Done. Processed %d video(s) in %.1fs. Output: %s", len(videos), elapsed, config.OUTPUT_VIDEO_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
main.py

CLI entrypoint for the LMT overlay pipeline.

Usage:
    python main.py
    python main.py --input-dir /data/exp42/raw --db /data/exp42/tracking.sqlite --output-dir /data/exp42/corrected
    python main.py --no-progress --log-level DEBUG

All arguments are optional; any omitted argument falls back to config.py's
default (itself overridable via the LMT_INPUT_DIR / LMT_DATABASE_PATH /
LMT_OUTPUT_DIR environment variables). CLI arguments take precedence over
both.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import config
import database
import video_processor

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render corrected LMT overlays (from SQLite ground truth) onto raw videos.",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=config.INPUT_VIDEO_DIR,
        help=f"Directory containing raw video_noOverlay_t*.mp4 files (default: {config.INPUT_VIDEO_DIR})",
    )
    parser.add_argument(
        "--db", type=Path, default=config.DATABASE_PATH,
        help=f"Path to the LMT SQLite database (default: {config.DATABASE_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=config.OUTPUT_VIDEO_DIR,
        help=f"Directory to write corrected videos into (default: {config.OUTPUT_VIDEO_DIR})",
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


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    """Push CLI-provided paths/flags into config, which the rest of the
    pipeline reads from module-level constants."""
    config.INPUT_VIDEO_DIR = args.input_dir.resolve()
    config.DATABASE_PATH = args.db.resolve()
    config.OUTPUT_VIDEO_DIR = args.output_dir.resolve()
    if args.no_progress:
        config.PROGRESS_BAR = False


def _sanity_check_frame_overlap(conn, videos: list[video_processor.VideoInfo]) -> None:
    """
    Warn (but don't abort) if the discovered videos' global frame ranges
    don't overlap the DETECTION table's frame range at all -- usually
    means a mismatched database/video-directory pairing.
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
            "Discovered videos span global frames [%d, %d) but DETECTION "
            "spans [%d, %d] -- these do not overlap at all. Double-check "
            "that --input-dir and --db point at the same experiment.",
            video_min, video_max, db_min, db_max,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _configure_logging(args.log_level)
    _apply_cli_overrides(args)

    try:
        config.validate_config()
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Discovering raw videos in %s", config.INPUT_VIDEO_DIR)
    videos = video_processor.discover_videos(config.INPUT_VIDEO_DIR)
    if not videos:
        logger.error("No usable videos found; nothing to do.")
        return 1
    logger.info("Discovered %d video(s), global frame range [%d, %d)",
                len(videos), videos[0].start_frame, videos[-1].end_frame)

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
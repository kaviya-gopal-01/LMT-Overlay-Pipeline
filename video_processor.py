"""
video_processor.py

Sequential per-video driver for the overlay pipeline.

For each raw video:
  1. Open it once with cv2.VideoCapture (never reopened mid-stream).
  2. Determine its global SQLite frame range from its filename
     (video_noOverlay_t<start>.mp4) and its own frame count.
  3. Stream through it frame-by-frame in natural order, pulling
     detections from SQLite in bounded-size chunks aligned to that
     video's own frame range -- never a single query for the whole
     recording, never a per-frame query either.
  4. Render overlays onto each frame via overlay_renderer.
  5. Write the corrected frame to a new output video.

This design guarantees O(1) video file handles open at a time and O(chunk
size) detections held in memory at a time, regardless of how long the
overall recording is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import sqlite3

import config
import database
import overlay_renderer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements.txt
    tqdm = None

logger = logging.getLogger(__name__)


class VideoProcessingError(RuntimeError):
    """Raised when a raw video cannot be opened or read."""


@dataclass
class VideoInfo:
    path: Path
    start_frame: int          # global SQLite frame number this video starts at
    frame_count: int          # total frames per cv2's own count
    fps: float
    width: int
    height: int

    @property
    def end_frame(self) -> int:
        """Exclusive upper bound of this video's global frame range."""
        return self.start_frame + self.frame_count * config.FRAME_CONVERSION

    @property
    def output_path(self) -> Path:
        """
        Output filename is derived from already-parsed, trusted fields
        (start_frame, original extension) rather than string-manipulating
        the input filename. This generalizes cleanly regardless of the
        raw video's naming prefix: 'video_noOverlay_t18024.mp4',
        'cam2_raw_noOverlay_t18024.mp4', and 'anything_t18024.mp4' all
        produce the same output name: 'video_t18024.mp4'.
        """
        ext = self.path.suffix  # preserve original extension, e.g. ".mp4"
        return config.OUTPUT_VIDEO_DIR / f"{config.OUTPUT_VIDEO_PREFIX}_t{self.start_frame}{ext}"


def _extract_start_frame(video_path: Path) -> Optional[int]:
    match = config.VIDEO_FILENAME_PATTERN.search(video_path.name)
    if match is None:
        return None
    return int(match.group(1))


def probe_video(video_path: Path) -> VideoInfo:
    """
    Open a raw video just long enough to read its metadata (frame count,
    fps, dimensions) and its global start frame from its filename, then
    close it. The caller reopens the same path for actual streaming --
    this keeps discovery decoupled from processing and makes discovery
    failures (e.g. an unparseable filename) cheap to detect up front.

    Extension is checked explicitly, and separately from the start-frame
    pattern below, so an unsupported format (e.g. a stray .avi/.mov
    picked via the GUI's "All files" fallback) fails with a message
    about the format itself, rather than being folded into the generic
    "couldn't find _t<digits>" message that VIDEO_FILENAME_PATTERN
    produces for .mp4 files with a bad naming.
    """
    if video_path.suffix.lower() != ".mp4":
        raise VideoProcessingError(
            f"Unsupported video format {video_path.suffix!r} for {video_path.name}: "
            f"this pipeline only supports .mp4 input videos."
        )

    start_frame = _extract_start_frame(video_path)
    if start_frame is None:
        raise VideoProcessingError(
            f"Could not extract a global start frame from filename: {video_path.name} "
            f"(expected pattern like 'video_noOverlay_t18024.mp4')"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise VideoProcessingError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.RAW_VIDEO_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame_count <= 0:
        raise VideoProcessingError(f"Video reports zero or unknown frame count: {video_path}")
    if width <= 0 or height <= 0:
        raise VideoProcessingError(f"Video reports invalid dimensions ({width}x{height}): {video_path}")

    return VideoInfo(path=video_path, start_frame=start_frame, frame_count=frame_count, fps=fps, width=width, height=height,)


def _deduplicate_by_start_frame(videos: List[VideoInfo]) -> List[VideoInfo]:
    """
    Two videos sharing the same start_frame would produce the SAME output
    filename ("video_t<N>.ext") and silently overwrite each other, since
    output naming is keyed only by start_frame (not by original filename
    or original folder). Shared by both discover_videos() and
    discover_videos_from_paths() so this protection applies regardless of
    how the video list was assembled.
    """
    seen_start_frames: Dict[int, Path] = {}
    deduplicated: List[VideoInfo] = []
    for video in videos:
        prior = seen_start_frames.get(video.start_frame)
        if prior is not None:
            logger.error(
                "Duplicate start_frame=%d between %s and %s; skipping %s "
                "to avoid overwriting output from %s. Rename or deselect "
                "one of these raw files if both are actually needed.",
                video.start_frame, prior.name, video.path.name, video.path.name, prior.name,
            )
            continue
        seen_start_frames[video.start_frame] = video.path
        deduplicated.append(video)
    return deduplicated


def discover_videos(input_dir: Path = config.INPUT_VIDEO_DIR) -> List[VideoInfo]:
    """
    Find all raw videos matching config.VIDEO_GLOB_PATTERN under
    input_dir, probe each for metadata, and return them sorted by global
    start frame. Videos whose filename or file itself can't be parsed are
    logged and skipped rather than aborting discovery for the rest.

    This directory-glob discovery path is kept for scripted/headless use;
    the GUI flow in main.py uses discover_videos_from_paths() instead,
    since Tkinter's file picker returns an explicit list of files that
    may span multiple folders, not a single directory to glob.
    """
    candidates = sorted(input_dir.glob(config.VIDEO_GLOB_PATTERN))
    if not candidates:
        logger.warning(
            "No videos matched pattern %r under %s", config.VIDEO_GLOB_PATTERN, input_dir
        )
        return []

    videos: List[VideoInfo] = []
    for path in candidates:
        try:
            videos.append(probe_video(path))
        except VideoProcessingError as exc:
            logger.error("Skipping unusable video %s: %s", path, exc)

    videos.sort(key=lambda v: v.start_frame)
    return _deduplicate_by_start_frame(videos)


def discover_videos_from_paths(video_paths: List[Path]) -> List[VideoInfo]:
    """
    Probe an explicit, user-supplied list of video file paths -- e.g.
    returned by gui_selector.select_database_and_videos() -- rather than
    globbing a single directory. Videos may live in different folders;
    each path is probed independently via the same probe_video() used by
    discover_videos(), so filename parsing, codec checks, and error
    handling are identical between the two discovery paths.

    Same duplicate-start_frame protection as discover_videos() (see
    _deduplicate_by_start_frame's docstring for why that matters).
    """
    videos: List[VideoInfo] = []
    for path in video_paths:
        path = Path(path)
        try:
            videos.append(probe_video(path))
        except VideoProcessingError as exc:
            logger.error("Skipping unusable video %s: %s", path, exc)

    videos.sort(key=lambda v: v.start_frame)
    return _deduplicate_by_start_frame(videos)


def _iter_local_chunks(frame_count: int, chunk_size: int):
    """Yield (local_start, local_end) pairs covering [0, frame_count) in
    fixed-size, non-overlapping local-frame chunks."""
    local_start = 0
    while local_start < frame_count:
        local_end = min(local_start + chunk_size, frame_count)
        yield local_start, local_end
        local_start = local_end


def process_single_video(conn: sqlite3.Connection, video: VideoInfo) -> None:
    """
    Stream through a single raw video, rendering overlays from SQLite
    detections and writing the corrected result to video.output_path.
    """
    logger.info(
        "Processing %s (start_frame=%d, frames=%d, %dx%d @ %.2ffps) -> %s",
        video.path.name, video.start_frame, video.frame_count,
        video.width, video.height, video.fps, video.output_path.name,
    )

    cap = cv2.VideoCapture(str(video.path))
    if not cap.isOpened():
        raise VideoProcessingError(f"Could not (re)open video for processing: {video.path}")

    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_FOURCC)
    writer = cv2.VideoWriter(str(video.output_path), fourcc, config.OUTPUT_FPS, (video.width, video.height))
    if not writer.isOpened():
        cap.release()
        raise VideoProcessingError(f"Could not open VideoWriter for: {video.output_path}")

    local_chunk_size = max(1, config.DB_QUERY_CHUNK_FRAMES // config.FRAME_CONVERSION)

    frames_written = 0
    progress = None
    if config.PROGRESS_BAR and tqdm is not None:
        progress = tqdm(total=video.frame_count, desc=video.path.name, unit="frame")

    try:
        for local_start, local_end in _iter_local_chunks(video.frame_count, local_chunk_size):
            global_start = video.start_frame + local_start * config.FRAME_CONVERSION
            global_end = video.start_frame + local_end * config.FRAME_CONVERSION

            rows = database.fetch_detections_in_range(conn, global_start, global_end)
            detections_by_frame: Dict[int, list] = {}
            for row in rows:
                detections_by_frame.setdefault(row.framenumber, []).append(row)

            for local_frame in range(local_start, local_end):
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        "%s: expected %d frames but read failed at local frame %d "
                        "(global frame %d); stopping early for this video.",
                        video.path.name, video.frame_count, local_frame,
                        video.start_frame + local_frame * config.FRAME_CONVERSION,
                    )
                    return

                global_frame = video.start_frame + local_frame * config.FRAME_CONVERSION
                detections = detections_by_frame.get(global_frame, [])
                overlay_renderer.render_frame(frame, detections)
                overlay_renderer.draw_global_frame_number(frame, global_frame)

                writer.write(frame)
                frames_written += 1
                if progress is not None:
                    progress.update(1)
    finally:
        cap.release()
        writer.release()
        if progress is not None:
            progress.close()

    logger.info("Finished %s: wrote %d frames to %s", video.path.name, frames_written, video.output_path)


def process_all_videos(conn: sqlite3.Connection, videos: List[VideoInfo]) -> None:
    """
    Process every discovered video in order, logging and continuing past
    any single video's failure so one bad file doesn't abort an entire
    24-hour experiment's worth of already-queued work.
    """
    database.warn_if_unindexed(conn)

    for video in videos:
        try:
            process_single_video(conn, video)
        except VideoProcessingError as exc:
            logger.error("Failed to process %s: %s", video.path, exc)
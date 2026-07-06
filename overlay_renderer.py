"""
overlay_renderer.py

Renders corrected per-animal overlays (segmentation mask, pose markers,
posture labels) onto a single video frame, given that frame's DETECTION
rows from the database.

This module owns the per-detection XML parsing + mask decoding + drawing
pipeline, and is deliberately defensive at the per-detection level: one
corrupt or undecodable detection (bad XML, mask size mismatch, etc.) logs
a warning and is skipped -- it must never abort rendering for the rest of
the animals in that frame, since a single bad row 6 hours into a 24-hour
recording shouldn't take down the whole run.
"""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import cv2
import numpy as np

import config
import mask_decoder
import xml_parser
from database import DetectionRow

logger = logging.getLogger(__name__)


def get_animal_color(animal_id: int) -> Tuple[int, int, int]:
    """Return the BGR color for a given ANIMALID, falling back to a
    distinct 'unassigned' color for any ID not explicitly configured."""
    return config.ANIMAL_COLORS_BGR.get(animal_id, config.ANIMAL_COLOR_FALLBACK_BGR)


def _clip_roi_to_frame(
    frame_shape: Tuple[int, int],
    roi_x: int,
    roi_y: int,
    roi_w: int,
    roi_h: int,
) -> Tuple[int, int, int, int, int, int, int, int] | None:
    """
    Compute the clipped intersection of an ROI with the frame.

    Returns (frame_x0, frame_y0, frame_x1, frame_y1,
             mask_x0, mask_y0, mask_x1, mask_y1)
    or None if the ROI doesn't intersect the frame at all (e.g. an animal
    detected right at the edge of the cage with an ROI partially past
    frame bounds -- normal, not an error).
    """
    frame_h, frame_w = frame_shape

    x0, y0 = roi_x, roi_y
    x1, y1 = roi_x + roi_w, roi_y + roi_h

    clipped_x0, clipped_y0 = max(x0, 0), max(y0, 0)
    clipped_x1, clipped_y1 = min(x1, frame_w), min(y1, frame_h)

    if clipped_x1 <= clipped_x0 or clipped_y1 <= clipped_y0:
        return None

    mask_x0 = clipped_x0 - x0
    mask_y0 = clipped_y0 - y0
    mask_x1 = mask_x0 + (clipped_x1 - clipped_x0)
    mask_y1 = mask_y0 + (clipped_y1 - clipped_y0)

    return clipped_x0, clipped_y0, clipped_x1, clipped_y1, mask_x0, mask_y0, mask_x1, mask_y1


def _draw_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    roi_x: int,
    roi_y: int,
    color: Tuple[int, int, int],
) -> None:
    """Alpha-blend the colorized mask onto `frame` in place, clipped to
    frame bounds, then optionally draw its contour outline."""
    clip = _clip_roi_to_frame(frame.shape[:2], roi_x, roi_y, mask.shape[1], mask.shape[0])
    if clip is None:
        return
    fx0, fy0, fx1, fy1, mx0, my0, mx1, my1 = clip

    mask_region = mask[my0:my1, mx0:mx1]
    if not mask_region.any():
        return

    frame_region = frame[fy0:fy1, fx0:fx1]
    bool_region = mask_region.astype(bool)

    color_layer = np.empty_like(frame_region)
    color_layer[:] = color

    blended = cv2.addWeighted(frame_region, 1.0 - config.MASK_ALPHA, color_layer, config.MASK_ALPHA, 0)
    frame_region[bool_region] = blended[bool_region]

    if config.DRAW_MASK_OUTLINE:
        binary_full = mask_decoder.to_binary_uint8(mask)
        contours, _ = cv2.findContours(binary_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            offset_contours = [c + np.array([roi_x, roi_y]) for c in contours]
            cv2.drawContours(frame, offset_contours, -1, color, config.MASK_OUTLINE_THICKNESS)


def _draw_pose_markers(
    frame: np.ndarray,
    parsed: xml_parser.ParsedDetectionData,
    color: Tuple[int, int, int],
) -> None:
    points = (
        (parsed.front_x, parsed.front_y),
        (parsed.back_x, parsed.back_y),
        (parsed.mass_x, parsed.mass_y),
    )
    for px, py in points:
        if px is None or py is None:
            continue
        cv2.circle(
            frame,
            (int(round(px)), int(round(py))),
            config.POSE_MARKER_RADIUS,
            color,
            config.POSE_MARKER_THICKNESS,
        )


def _draw_posture_label(
    frame: np.ndarray,
    animal_id: int,
    parsed: xml_parser.ParsedDetectionData,
    color: Tuple[int, int, int],
    anchor_x: int,
    anchor_y: int,
) -> None:
    flags = []
    if parsed.is_rearing:
        flags.append("REAR")
    if parsed.is_looking_up:
        flags.append("LOOK_UP")
    if parsed.is_looking_down:
        flags.append("LOOK_DOWN")

    label = f"A{animal_id}" + (f" [{'|'.join(flags)}]" if flags else "")
    text_origin = (max(anchor_x, 0), max(anchor_y - 4, 10))
    cv2.putText(
        frame,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.POSTURE_LABEL_FONT_SCALE,
        color,
        config.POSTURE_LABEL_THICKNESS,
        cv2.LINE_AA,
    )


def render_detection(frame: np.ndarray, detection: DetectionRow) -> None:
    """
    Render a single animal's detection onto `frame` in place. Any
    per-detection failure (malformed XML, undecodable mask) is logged and
    swallowed so it cannot abort the rest of the frame's rendering.
    """
    color = get_animal_color(detection.animal_id)

    try:
        parsed = xml_parser.parse_detection_xml(detection.data_xml)
    except xml_parser.XmlParseError as exc:
        logger.warning(
            "Frame %s, animal %s: failed to parse DATA XML (%s); skipping this detection.",
            detection.framenumber, detection.animal_id, exc,
        )
        return

    if config.DRAW_POSE_MARKERS:
        _draw_pose_markers(frame, parsed, color)

    if parsed.has_mask:
        roi = parsed.roi
        try:
            mask = mask_decoder.decode_bool_mask(roi.bool_mask_data, roi.w, roi.h)
        except mask_decoder.MaskDecodeError as exc:
            logger.warning(
                "Frame %s, animal %s: failed to decode mask (%s); "
                "rendering pose markers only for this detection.",
                detection.framenumber, detection.animal_id, exc,
            )
        else:
            _draw_mask(frame, mask, roi.x, roi.y, color)

    if config.DRAW_POSTURE_LABEL:
        anchor_x = parsed.roi.x if parsed.has_mask else (
            int(parsed.mass_x) if parsed.mass_x is not None else 0
        )
        anchor_y = parsed.roi.y if parsed.has_mask else (
            int(parsed.mass_y) if parsed.mass_y is not None else 0
        )
        _draw_posture_label(frame, detection.animal_id, parsed, color, anchor_x, anchor_y)


def render_frame(frame: np.ndarray, detections: Iterable[DetectionRow]) -> np.ndarray:
    """
    Render all animal detections for a single frame onto it, in place.
    Returns the same frame object for convenient chaining at call sites.
    """
    for detection in detections:
        render_detection(frame, detection)
    return frame
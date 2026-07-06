"""
xml_parser.py

Parses the DETECTION.DATA XML column into structured Python objects.

Observed structure (confirmed against real sample rows):

    <root>
      <DATA front_x="..." front_y="..." front_z="..."
            back_x="..." back_y="..." back_z="..."
            mass_x="..." mass_y="..." mass_z="..."
            ear1_x="..." ear1_y="..." ear1_ratio="..."
            ear2_x="..." ear2_y="..." ear2_ratio="..."
            isRearing="true|false" isLookingUp="true|false"
            isLookingDown="true|false" t="209"/>
      <ROI>
        <classname>plugins.fab.livemousetracker.ROI2DAreaX</classname>
        <id>3654</id>
        <name>seg ok</name>
        ...
        <boundsX>113</boundsX>
        <boundsY>324</boundsY>
        <boundsW>31</boundsW>
        <boundsH>37</boundsH>
        <boolMaskData>78:5e:ad:92:...</boolMaskData>
      </ROI>
    </root>

<DATA> carries pose/posture as XML attributes. <ROI> carries the mask
bounding box and the raw mask payload as child element text. The <ROI>
element is entirely optional -- gap-filled / undetected frames have no
segmentation and therefore no <ROI> block (or an incomplete one); this
parser treats that as "no mask available" rather than an error, since it
is expected and common, not corrupt data.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class XmlParseError(ValueError):
    """Raised when a DETECTION.DATA XML blob cannot be parsed at all."""


@dataclass(frozen=True, slots=True)
class ROIBounds:
    """Bounding box + raw mask payload for a single detection's ROI."""

    x: int
    y: int
    w: int
    h: int
    bool_mask_data: str
    name: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ParsedDetectionData:
    """Structured view of a single DETECTION.DATA XML blob."""

    front_x: Optional[float]
    front_y: Optional[float]
    front_z: Optional[float]
    back_x: Optional[float]
    back_y: Optional[float]
    back_z: Optional[float]
    mass_x: Optional[float]
    mass_y: Optional[float]
    mass_z: Optional[float]
    ear1_x: Optional[float]
    ear1_y: Optional[float]
    ear1_ratio: Optional[float]
    ear2_x: Optional[float]
    ear2_y: Optional[float]
    ear2_ratio: Optional[float]
    is_rearing: bool
    is_looking_up: bool
    is_looking_down: bool
    t: Optional[int]
    roi: Optional[ROIBounds]

    @property
    def has_mask(self) -> bool:
        return self.roi is not None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _to_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() == "true"


def _parse_roi(roi_elem: ET.Element) -> Optional[ROIBounds]:
    def child_text(tag: str) -> Optional[str]:
        child = roi_elem.find(tag)
        if child is None or child.text is None:
            return None
        text = child.text.strip()
        return text or None

    bounds_x = _to_int(child_text("boundsX"))
    bounds_y = _to_int(child_text("boundsY"))
    bounds_w = _to_int(child_text("boundsW"))
    bounds_h = _to_int(child_text("boundsH"))
    mask_data = child_text("boolMaskData")
    name = child_text("name")

    if None in (bounds_x, bounds_y, bounds_w, bounds_h):
        logger.debug("ROI present but bounds incomplete; treating as no mask available.")
        return None

    if bounds_w <= 0 or bounds_h <= 0:
        logger.debug(
            "ROI bounds non-positive (w=%s, h=%s); treating as no mask available.",
            bounds_w, bounds_h,
        )
        return None

    if not mask_data:
        logger.debug("ROI present with valid bounds but no boolMaskData payload.")
        return None

    return ROIBounds(
        x=bounds_x,
        y=bounds_y,
        w=bounds_w,
        h=bounds_h,
        bool_mask_data=mask_data,
        name=name,
    )


def parse_detection_xml(xml_string: Optional[str]) -> ParsedDetectionData:
    """
    Parse a single DETECTION.DATA XML blob.

    Raises XmlParseError only when the XML itself is missing/malformed or
    the mandatory <DATA> element is absent -- these indicate genuinely
    corrupt rows, as opposed to a merely-absent <ROI> (undetected frame),
    which is represented via `ParsedDetectionData.roi is None`.
    """
    if not xml_string or not xml_string.strip():
        raise XmlParseError("Empty or None DATA XML string.")

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as exc:
        raise XmlParseError(f"Malformed DATA XML: {exc}") from exc

    data_elem = root.find("DATA")
    if data_elem is None:
        raise XmlParseError("DATA XML missing required <DATA> element.")

    attrs = data_elem.attrib

    roi_elem = root.find("ROI")
    roi_bounds = _parse_roi(roi_elem) if roi_elem is not None else None

    return ParsedDetectionData(
        front_x=_to_float(attrs.get("front_x")),
        front_y=_to_float(attrs.get("front_y")),
        front_z=_to_float(attrs.get("front_z")),
        back_x=_to_float(attrs.get("back_x")),
        back_y=_to_float(attrs.get("back_y")),
        back_z=_to_float(attrs.get("back_z")),
        mass_x=_to_float(attrs.get("mass_x")),
        mass_y=_to_float(attrs.get("mass_y")),
        mass_z=_to_float(attrs.get("mass_z")),
        ear1_x=_to_float(attrs.get("ear1_x")),
        ear1_y=_to_float(attrs.get("ear1_y")),
        ear1_ratio=_to_float(attrs.get("ear1_ratio")),
        ear2_x=_to_float(attrs.get("ear2_x")),
        ear2_y=_to_float(attrs.get("ear2_y")),
        ear2_ratio=_to_float(attrs.get("ear2_ratio")),
        is_rearing=_to_bool(attrs.get("isRearing")),
        is_looking_up=_to_bool(attrs.get("isLookingUp")),
        is_looking_down=_to_bool(attrs.get("isLookingDown")),
        t=_to_int(attrs.get("t")),
        roi=roi_bounds,
    )
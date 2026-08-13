"""
mask_decoder.py

Decodes the boolMaskData payload embedded in each detection's <ROI> XML
block into a 2D numpy segmentation mask.

ENCODING (determined empirically against a real sample row from this
project's own database):

    1. boolMaskData is a string of colon-separated hex byte values, e.g.
       "78:5e:ad:92:4b:16:...". Individual tokens may be 1 or 2 hex
       digits (Java's Integer.toHexString-style output does not
       zero-pad), so tokens must be parsed with int(token, 16) rather
       than assumed to be fixed-width byte pairs.

    2. Parsing every token into a byte yields a raw byte stream that
       begins with 0x78 0x5e -- the zlib header for "default
       compression" (produced by Java's java.util.zip.Deflater at its
       default level). This stream is a standard zlib-wrapped deflate
       stream and decompresses with Python's stdlib `zlib` module
       directly, no custom deflate handling needed.

    3. The decompressed payload is exactly `boundsW * boundsH` bytes,
       one byte per pixel, with values restricted to {0, 1} (verified:
       decompressed length matched width*height exactly, and the byte
       value set was exactly {0, 1}).

    4. Reshaping that byte buffer row-major as (boundsH, boundsW) (i.e.
       numpy's default `reshape(h, w)`) produces a single coherent,
       contiguous blob matching a mouse silhouette. The transposed
       (column-major) interpretation produces incoherent noise, which
       rules it out. Row-major (h, w) is therefore the correct
       orientation, matching standard image array conventions
       (row index = y, column index = x) with no additional flip.

This module intentionally does not guess at alternative encodings (RLE,
bit-packing, etc.) -- those were considered and ruled out by the byte
count check in step 3.
"""

from __future__ import annotations

import logging
import zlib
from typing import Iterable

import numpy as np

import config

logger = logging.getLogger(__name__)

class MaskDecodeError(ValueError):
    """Raised when boolMaskData cannot be decoded into a valid mask."""


def _hex_tokens_to_bytes(hex_string: str, separator: str) -> bytes:
    tokens: Iterable[str] = hex_string.split(separator)
    try:
        return bytes(int(tok, 16) for tok in tokens if tok != "")
    except ValueError as exc:
        raise MaskDecodeError(f"boolMaskData contains a non-hex token: {exc}") from exc


def decode_bool_mask(bool_mask_data: str, width: int, height: int, *, strict: bool | None = None,) -> np.ndarray:
    """
    Decode a boolMaskData string into a (height, width) uint8 array with
    values in {0, 1}, where 1 marks a foreground (mouse) pixel.

    strict:
        If True, a size mismatch or decode failure raises
        MaskDecodeError. If False, a size mismatch is logged and the
        payload is truncated/zero-padded to fit rather than raising --
        useful for exploratory work, but NOT recommended for production
        rendering since it can silently misalign a mask. If omitted
        (None, the default), resolves to config.MASK_STRICT_SIZE_VALIDATION
        at call time, so a runtime change to that config value is honored
        without needing to reimport this module.
    """
    if strict is None:
        strict = config.MASK_STRICT_SIZE_VALIDATION

    if not bool_mask_data or not bool_mask_data.strip():
        raise MaskDecodeError("Empty boolMaskData string.")
    if width <= 0 or height <= 0:
        raise MaskDecodeError(f"Invalid mask dimensions: width={width}, height={height}")

    compressed = _hex_tokens_to_bytes(bool_mask_data.strip(), config.MASK_BYTE_SEPARATOR)

    try:
        decompressed = zlib.decompress(compressed)
    except zlib.error as exc:
        raise MaskDecodeError(f"zlib decompression failed: {exc}") from exc

    expected_len = width * height
    actual_len = len(decompressed)

    if actual_len != expected_len:
        message = (
            f"Decompressed mask length {actual_len} != expected "
            f"width*height={expected_len} (width={width}, height={height})."
        )
        if strict:
            raise MaskDecodeError(message)
        logger.warning("%s Truncating/padding to fit (strict mode disabled).", message)
        if actual_len > expected_len:
            decompressed = decompressed[:expected_len]
        else:
            decompressed = decompressed + b"\x00" * (expected_len - actual_len)

    mask = np.frombuffer(decompressed, dtype=np.uint8).reshape(height, width)

    # Defensive check: values should only ever be 0 or 1. Anything else
    # indicates either a wrong encoding assumption or genuinely corrupt
    # data -- either way, downstream color/alpha math (which treats the
    # mask as a boolean selector) would silently misbehave otherwise.
    unique_values = np.unique(mask)
    if not np.all(np.isin(unique_values, (0, config.MASK_FOREGROUND_VALUE))):
        message = f"Decoded mask contains unexpected values: {unique_values.tolist()}"
        if strict:
            raise MaskDecodeError(message)
        logger.warning("%s Clamping to boolean via != 0.", message)
        mask = (mask != 0).astype(np.uint8)

    return mask


def to_binary_uint8(mask: np.ndarray) -> np.ndarray:
    """
    Convert a {0, config.MASK_FOREGROUND_VALUE}-valued mask into an
    OpenCV-friendly {0,255} uint8 mask, suitable for cv2.findContours /
    cv2.bitwise_* operations.

    Uses an explicit equality test against config.MASK_FOREGROUND_VALUE
    rather than treating the foreground value as a multiplier -- with
    the default MASK_FOREGROUND_VALUE=1 this produces identical output
    to a plain `mask * 255`, but it stays correct if that config value
    is ever anything other than 1 (a plain multiply would silently wrap
    around uint8's 0-255 range for any foreground value other than 1).
    """
    return np.where(mask == config.MASK_FOREGROUND_VALUE, np.uint8(255), np.uint8(0))

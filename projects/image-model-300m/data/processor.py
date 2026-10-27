"""Pinned raw-image decoding and fixed-size RGB letterbox processing."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = 64
BACKGROUND_RGB = (127, 127, 127)
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


def decode_display_rgb(raw: bytes) -> tuple[Image.Image, dict]:
    """Decode one nonanimated file, orient it, and return display-value RGB.

    ICC profiles are not transformed. Alpha images are composited on the same
    neutral gray used for letterboxing; grayscale images are expanded to RGB.
    """
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError("image bytes must be nonempty and at most 20 MiB")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("animated or multi-frame images are not supported")
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("only JPEG and PNG inputs are supported")
            if source.width <= 0 or source.height <= 0 or source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("image dimensions exceed the 40-megapixel limit")
            source_format = source.format
            orientation = int(source.getexif().get(274, 1))
            oriented = ImageOps.exif_transpose(source)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (*BACKGROUND_RGB, 255))
                display = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                display = oriented.convert("RGB")
            info = {
                "source_format": source_format,
                "source_width": int(source.width),
                "source_height": int(source.height),
                "exif_orientation": orientation,
                "display_width": int(display.width),
                "display_height": int(display.height),
                "icc_profile_applied": False,
            }
            return display.copy(), info
    except (OSError, SyntaxError) as error:
        raise ValueError("image bytes could not be decoded") from error


def letterbox_rgb(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    if size <= 0 or image.width <= 0 or image.height <= 0:
        raise ValueError("letterbox dimensions must be positive")
    rgb = image.convert("RGB")
    contained = ImageOps.contain(rgb, (size, size), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), BACKGROUND_RGB)
    canvas.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def preprocess_bytes(raw: bytes, size: int = IMAGE_SIZE) -> tuple[np.ndarray, dict]:
    display, info = decode_display_rgb(raw)
    return letterbox_rgb(display, size), info


def normalize_uint8(images: np.ndarray) -> np.ndarray:
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("processor output must be uint8 NHWC at 64x64 RGB")
    return images.astype(np.float32) / np.float32(127.5) - np.float32(1.0)

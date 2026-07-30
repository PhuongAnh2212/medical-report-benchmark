"""
utils/image.py

Shared image-loading utilities used by all model wrappers. Keeping this
logic in one place ensures every model sees images preprocessed the same
way (RGB conversion, corrupt-file handling) regardless of which VLM is
being benchmarked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from utils.logging import get_logger

logger = get_logger(__name__)


def load_image(image_path: str | Path) -> Optional[Image.Image]:
    """Load an image from disk as an RGB PIL Image.

    Chest X-rays are frequently stored as single-channel (grayscale) PNGs
    or DICOM-derived JPEGs; converting to RGB up front means every model
    wrapper can assume a 3-channel input, matching what their processors
    expect.

    Args:
        image_path: Path to the image file.

    Returns:
        A PIL Image in RGB mode, or None if the file could not be read.
    """
    path = Path(image_path)
    if not path.exists():
        logger.error("Image not found: %s", path)
        return None

    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - we want to log and continue benchmarking
        logger.error("Failed to load image %s: %s", path, exc)
        return None


def resize_if_needed(image: Image.Image, max_side: int = 1024) -> Image.Image:
    """Downscale an image so its longest side does not exceed `max_side`.

    Many VLM processors will internally resize anyway, but capping the
    input size here avoids unnecessary memory spikes and speeds up
    preprocessing for very large radiographs, without changing aspect ratio.

    Args:
        image: Input PIL Image.
        max_side: Maximum allowed length (in pixels) of the longer side.

    Returns:
        The original image if already small enough, otherwise a resized copy.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)

"""
models/internvl3.py

Wrapper for InternVL3 (e.g. OpenGVLab/InternVL3-8B) implementing the
BaseReportGenerator interface.

InternVL3 checkpoints are typically loaded via `trust_remote_code=True`
and use a dynamic-tiling image preprocessing scheme. This wrapper follows
the reference usage pattern published by OpenGVLab.

Requires:
    pip install transformers accelerate timm einops
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from models.base_model import BaseReportGenerator
from utils.logging import get_logger

logger = get_logger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int) -> T.Compose:
    """Standard ImageNet-style normalization transform used by InternVL."""
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: List, width: int, height: int, image_size: int
):
    """Pick the tiling grid (rows x cols) whose aspect ratio best matches the image."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image, min_num: int = 1, max_num: int = 6, image_size: int = 448
) -> List[Image.Image]:
    """Split an image into a set of tiles sized for InternVL's ViT encoder."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    cols = target_width // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def _load_image_as_pixel_values(
    image: Image.Image, input_size: int = 448, max_num: int = 6
) -> torch.Tensor:
    """Convert a PIL image into a stacked tensor of tiled, normalized pixel values."""
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    pixel_values = [transform(tile) for tile in tiles]
    return torch.stack(pixel_values)


class InternVL3ReportGenerator(BaseReportGenerator):
    """Report generator backed by InternVL3."""

    def __init__(self, config: Dict[str, Any], prompt_template: str) -> None:
        super().__init__(config, prompt_template)
        self.checkpoint = self.model_cfg.get("checkpoint", "OpenGVLab/InternVL3-8B")
        self.input_size = 448
        self.max_tiles = 6

    def load(self) -> None:
        """Load the InternVL3 model and tokenizer onto the configured device."""
        from transformers import AutoModel, AutoTokenizer

        device = self.model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = _DTYPE_MAP.get(self.model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

        logger.info("Loading InternVL3 checkpoint '%s' on %s (%s)", self.checkpoint, device, dtype)

        self._model = (
            AutoModel.from_pretrained(
                self.checkpoint,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )
        self._processor = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=True, use_fast=False
        )

    def generate(self, image: Image.Image) -> str:
        """Generate a radiology report for a single chest X-ray image.

        Args:
            image: RGB PIL Image.

        Returns:
            Generated report text.
        """
        self.ensure_loaded()
        device = next(self._model.parameters()).device
        dtype = next(self._model.parameters()).dtype

        pixel_values = _load_image_as_pixel_values(
            image, input_size=self.input_size, max_num=self.max_tiles
        ).to(device=device, dtype=dtype)

        prompt_text = self.build_prompt()
        question = f"<image>\n{prompt_text}"

        generation_config = dict(
            max_new_tokens=self.model_cfg.get("max_new_tokens", 512),
            do_sample=self.model_cfg.get("do_sample", False),
        )
        if generation_config["do_sample"]:
            generation_config["temperature"] = self.model_cfg.get("temperature", 0.2)
            generation_config["top_p"] = self.model_cfg.get("top_p", 0.9)

        with torch.no_grad():
            response = self._model.chat(
                self._processor,
                pixel_values,
                question,
                generation_config,
            )
        return response.strip()

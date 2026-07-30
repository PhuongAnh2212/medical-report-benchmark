"""
models/qwen25_vl.py

Wrapper for the Qwen VL model family implementing the BaseReportGenerator
interface. Supports both:

    Qwen/Qwen2-VL-*-Instruct      -> Qwen2VLForConditionalGeneration
    Qwen/Qwen2.5-VL-*-Instruct    -> Qwen2_5_VLForConditionalGeneration

The checkpoint id is read from configs/models.yaml (injected into
config["model"]["checkpoint"] by models.build_model). The architecture is
detected automatically from that checkpoint id -- see `_detect_architecture`
-- since the two families are distinct model classes in `transformers` and
loading a Qwen2-VL checkpoint with the 2.5 class (or vice versa) raises
"You are using a model of type ... to instantiate a model of type ...".

Requires:
    pip install transformers accelerate qwen-vl-utils

Note: both classes require a sufficiently recent `transformers` release.
If you hit an ImportError, upgrade transformers (see requirements.txt for
the minimum tested version).
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from PIL import Image

from models.base_model import BaseReportGenerator
from utils.logging import get_logger

logger = get_logger(__name__)

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# Default vision-token pixel budget passed to the Qwen processor. Both
# Qwen2-VL and Qwen2.5-VL use a dynamic-resolution vision encoder whose
# token count (and therefore O(n^2) attention memory) scales directly with
# input pixel count -- with no cap, a full-resolution chest X-ray (often
# several megapixels) can blow up to tens of GB of attention memory even on
# the 2B checkpoint. These values match the Qwen2-VL model card's
# recommendation for memory-constrained GPUs and bound vision tokens to
# roughly max_pixels / (28*28).
_DEFAULT_MIN_PIXELS = 256 * 28 * 28
_DEFAULT_MAX_PIXELS = 1280 * 28 * 28

# Substring markers used to detect the Qwen VL architecture family from a
# checkpoint id. Checked in this order; "Qwen2.5-VL" and "Qwen2-VL" never
# both match a single checkpoint id, so ordering is not load-bearing here,
# but 2.5 is checked first since it's this module's original/default family.
_ARCHITECTURE_MARKERS: tuple[tuple[str, str], ...] = (
    ("Qwen2.5-VL", "qwen2_5_vl"),
    ("Qwen2-VL", "qwen2_vl"),
)


def _detect_architecture(checkpoint: str) -> str:
    """Infer the Qwen VL architecture family from a checkpoint id.

    Args:
        checkpoint: HuggingFace checkpoint id, e.g.
            "Qwen/Qwen2.5-VL-7B-Instruct" or "Qwen/Qwen2-VL-2B-Instruct".

    Returns:
        "qwen2_5_vl" or "qwen2_vl".

    Raises:
        ValueError: If the checkpoint id matches neither known family.
    """
    for marker, architecture in _ARCHITECTURE_MARKERS:
        if marker in checkpoint:
            return architecture

    raise ValueError(
        f"Could not detect the Qwen VL architecture from checkpoint "
        f"'{checkpoint}'. Expected the checkpoint id to contain 'Qwen2.5-VL' "
        "or 'Qwen2-VL' (e.g. 'Qwen/Qwen2.5-VL-7B-Instruct' or "
        "'Qwen/Qwen2-VL-2B-Instruct')."
    )


class Qwen25VLReportGenerator(BaseReportGenerator):
    """Report generator backed by Qwen2-VL or Qwen2.5-VL (auto-detected)."""

    def __init__(self, config: Dict[str, Any], prompt_template: str) -> None:
        super().__init__(config, prompt_template)
        self.checkpoint = self.model_cfg.get("checkpoint", "Qwen/Qwen2.5-VL-7B-Instruct")

    def load(self) -> None:
        """Load the Qwen VL model and processor onto the configured device.

        Automatically picks `Qwen2VLForConditionalGeneration` or
        `Qwen2_5_VLForConditionalGeneration` based on `self.checkpoint`.
        """
        architecture = _detect_architecture(self.checkpoint)

        device = self.model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = _DTYPE_MAP.get(self.model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

        if architecture == "qwen2_vl":
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            logger.info("Detected architecture: Qwen2-VL")
            logger.info("Loading Qwen2VLForConditionalGeneration")
            model_class = Qwen2VLForConditionalGeneration
        else:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            logger.info("Detected architecture: Qwen2.5-VL")
            logger.info("Loading Qwen2_5_VLForConditionalGeneration")
            model_class = Qwen2_5_VLForConditionalGeneration

        logger.info("Loading checkpoint '%s' on %s (%s)", self.checkpoint, device, dtype)

        self._model = model_class.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            device_map=device,
        )

        # Bound vision tokens so a large input image can't blow up attention
        # memory (see CUDA OOM note above). `model.min_pixels`/`max_pixels`
        # in configs/default.yaml can override these per-GPU if needed.
        min_pixels = self.model_cfg.get("min_pixels") or _DEFAULT_MIN_PIXELS
        max_pixels = self.model_cfg.get("max_pixels") or _DEFAULT_MAX_PIXELS
        logger.info("Vision token pixel budget: min_pixels=%d, max_pixels=%d", min_pixels, max_pixels)

        self._processor = AutoProcessor.from_pretrained(
            self.checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self._model.eval()

    def generate(self, image: Image.Image) -> str:
        """Generate a radiology report for a single chest X-ray image.

        Args:
            image: RGB PIL Image.

        Returns:
            Generated report text, stripped of the input prompt/echo.
        """
        self.ensure_loaded()
        from qwen_vl_utils import process_vision_info

        prompt_text = self.build_prompt()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        chat_text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        gen_kwargs = dict(
            max_new_tokens=self.model_cfg.get("max_new_tokens", 512),
            do_sample=self.model_cfg.get("do_sample", False),
        )
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = self.model_cfg.get("temperature", 0.2)
            gen_kwargs["top_p"] = self.model_cfg.get("top_p", 0.9)

        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, **gen_kwargs)

        # Strip the input prompt tokens from the generated output.
        trimmed_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()

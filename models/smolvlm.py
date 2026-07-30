"""
models/smolvlm.py

Wrapper for SmolVLM2 (HuggingFaceTB/SmolVLM2-2.2B-Instruct) implementing
the BaseReportGenerator interface.

SmolVLM2 is natively supported by `transformers` (no trust_remote_code)
via the generic `AutoModelForImageTextToText` class introduced to unify
image/video/text chat-style VLMs. Images are passed as PIL.Image objects
directly inside the chat message content; `processor.apply_chat_template`
with `tokenize=True` handles both prompt templating and image
preprocessing in a single call.

Requires:
    pip install "transformers>=4.56.0" accelerate
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


class SmolVLMReportGenerator(BaseReportGenerator):
    """Report generator backed by SmolVLM2-Instruct."""

    def __init__(self, config: Dict[str, Any], prompt_template: str) -> None:
        super().__init__(config, prompt_template)
        self.checkpoint = self.model_cfg.get("checkpoint", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")

    def load(self) -> None:
        """Load the SmolVLM2 model and processor onto the configured device."""
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device_map = self.resolve_device_map()
        dtype = _DTYPE_MAP.get(self.model_cfg.get("dtype", "bfloat16"), torch.bfloat16)
        attn_implementation = self.model_cfg.get("attn_implementation")

        logger.info(
            "Loading SmolVLM2 checkpoint '%s' on device_map=%s (%s)", self.checkpoint, device_map, dtype
        )

        load_kwargs: Dict[str, Any] = dict(torch_dtype=dtype, device_map=device_map)
        if attn_implementation:
            # Not forced by default: flash_attention_2 requires a matching
            # compiled build that isn't guaranteed to be present (e.g. on
            # Kaggle GPUs); set model.attn_implementation in
            # configs/default.yaml to opt in where available.
            load_kwargs["_attn_implementation"] = attn_implementation

        self._model = AutoModelForImageTextToText.from_pretrained(self.checkpoint, **load_kwargs)
        self._processor = AutoProcessor.from_pretrained(self.checkpoint)
        self._model.eval()

    def generate(self, image: Image.Image) -> str:
        """Generate a radiology report for a single chest X-ray image.

        Args:
            image: RGB PIL Image.

        Returns:
            Generated report text, stripped of the input prompt/echo.
        """
        self.ensure_loaded()
        image = self.preprocess_image(image)

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

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=self._model.dtype)

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
        trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        output_text = self._processor.batch_decode(trimmed_ids, skip_special_tokens=True)
        return output_text[0].strip()

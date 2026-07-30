"""
models/qwen25_vl.py

Wrapper for Qwen2.5-VL (e.g. Qwen/Qwen2.5-VL-7B-Instruct) implementing the
BaseReportGenerator interface.

Requires:
    pip install transformers accelerate qwen-vl-utils

Note: Qwen2.5-VL ships as `Qwen2_5_VLForConditionalGeneration` in recent
`transformers` releases. If you hit an ImportError, upgrade transformers
(see requirements.txt for the minimum tested version).
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


class Qwen25VLReportGenerator(BaseReportGenerator):
    """Report generator backed by Qwen2.5-VL-Instruct."""

    def __init__(self, config: Dict[str, Any], prompt_template: str) -> None:
        super().__init__(config, prompt_template)
        self.checkpoint = self.model_cfg.get("checkpoint", "Qwen/Qwen2.5-VL-7B-Instruct")

    def load(self) -> None:
        """Load the Qwen2.5-VL model and processor onto the configured device."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        device = self.model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = _DTYPE_MAP.get(self.model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

        logger.info("Loading Qwen2.5-VL checkpoint '%s' on %s (%s)", self.checkpoint, device, dtype)

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=dtype,
            device_map=device,
        )
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

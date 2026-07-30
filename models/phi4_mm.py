"""
models/phi4_mm.py

Wrapper for Phi-4 Multimodal (microsoft/Phi-4-multimodal-instruct)
implementing the BaseReportGenerator interface.

As of its native `transformers` integration (Phi4MultimodalForCausalLM /
Phi4MultimodalProcessor, resolved automatically via AutoModelForCausalLM /
AutoProcessor), `trust_remote_code` is no longer required. The model bakes
in modality-specific LoRA adapters (vision / speech) that must be loaded
and *explicitly activated* via `load_adapter` + `set_adapter` before
generation -- skipping this silently runs the base (not vision-tuned)
weights and produces poor output instead of raising an error, so it's easy
to get wrong silently.

Requires:
    pip install "transformers>=4.56.0" accelerate peft
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

_VISION_ADAPTER_NAME = "vision"
_VISION_ADAPTER_SUBFOLDER = "vision-lora"


class Phi4MMReportGenerator(BaseReportGenerator):
    """Report generator backed by Phi-4 Multimodal, with the vision LoRA adapter active."""

    def __init__(self, config: Dict[str, Any], prompt_template: str) -> None:
        super().__init__(config, prompt_template)
        self.checkpoint = self.model_cfg.get("checkpoint", "microsoft/Phi-4-multimodal-instruct")

    def load(self) -> None:
        """Load Phi-4 Multimodal, the processor, and activate the vision LoRA adapter."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = self.model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        dtype = _DTYPE_MAP.get(self.model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

        logger.info("Loading Phi-4 Multimodal checkpoint '%s' on %s (%s)", self.checkpoint, device, dtype)

        self._processor = AutoProcessor.from_pretrained(self.checkpoint)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint, torch_dtype=dtype, device_map=device
        )

        logger.info("Loading and activating the '%s' LoRA adapter", _VISION_ADAPTER_NAME)
        self._model.load_adapter(
            self.checkpoint,
            adapter_name=_VISION_ADAPTER_NAME,
            device_map=device,
            adapter_kwargs={"subfolder": _VISION_ADAPTER_SUBFOLDER},
        )
        self._model.set_adapter(_VISION_ADAPTER_NAME)
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
        trimmed_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        output_text = self._processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0].strip()

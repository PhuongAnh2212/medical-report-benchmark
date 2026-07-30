"""
models/molmo.py

Wrapper for Molmo (allenai/Molmo-7B-D-0924) implementing the
BaseReportGenerator interface.

Molmo ships as `trust_remote_code` custom modeling code with a distinctive
API that doesn't match the other wrappers in this package:
  - the processor exposes `process(images=..., text=...)`, not the usual
    `processor(...)` / `apply_chat_template(...)`;
  - generation goes through `model.generate_from_batch(...)`, not
    `model.generate(...)`, since Molmo interleaves vision and text tokens
    in a way the standard `generate()` entrypoint doesn't handle.

Requires:
    pip install transformers accelerate einops torchvision
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


class MolmoReportGenerator(BaseReportGenerator):
    """Report generator backed by Molmo-7B-D."""

    def __init__(self, config: Dict[str, Any], prompt_template: str) -> None:
        super().__init__(config, prompt_template)
        self.checkpoint = self.model_cfg.get("checkpoint", "allenai/Molmo-7B-D-0924")

    def load(self) -> None:
        """Load Molmo's model and processor onto the configured device."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        device_map = self.resolve_device_map()
        dtype = _DTYPE_MAP.get(self.model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

        logger.info(
            "Loading Molmo checkpoint '%s' on device_map=%s (%s)", self.checkpoint, device_map, dtype
        )

        self._processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)

        # low_cpu_mem_usage=False: Molmo's trust_remote_code modeling code
        # hits the same meta-tensor ".item() cannot be called on meta
        # tensors" failure mode as InternVL under transformers' default
        # low_cpu_mem_usage=True fast-init path -- see models/internvl3.py
        # for the confirmed root cause. Same fix here, pre-emptively. This
        # also means from_pretrained's own `device_map` can't be used (it
        # would force the meta-init path back on) -- place the fully
        # materialized model via BaseReportGenerator.place_model instead,
        # which shards across GPUs (accelerate's dispatch_model) without
        # needing meta-init. Molmo is the largest checkpoint in this
        # registry (7B, ~14GB in bf16), so this is the model most likely to
        # actually need both of a Kaggle GPU T4 x2's GPUs.
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        ).eval()
        self._model = self.place_model(model, device_map)

    def generate(self, image: Image.Image) -> str:
        """Generate a radiology report for a single chest X-ray image.

        Args:
            image: RGB PIL Image.

        Returns:
            Generated report text.
        """
        self.ensure_loaded()
        from transformers import GenerationConfig

        # Molmo is the largest checkpoint in this registry (7B); an
        # uncapped multi-megapixel X-ray risks the same OOM class observed
        # with Qwen2-VL on smaller GPUs.
        image = self.preprocess_image(image)

        prompt_text = self.build_prompt()

        inputs = self._processor.process(images=[image], text=prompt_text)
        inputs = {k: v.to(self._model.device).unsqueeze(0) for k, v in inputs.items()}

        gen_config = GenerationConfig(
            max_new_tokens=self.model_cfg.get("max_new_tokens", 512),
            stop_strings="<|endoftext|>",
            do_sample=self.model_cfg.get("do_sample", False),
        )
        if gen_config.do_sample:
            gen_config.temperature = self.model_cfg.get("temperature", 0.2)
            gen_config.top_p = self.model_cfg.get("top_p", 0.9)

        with torch.no_grad():
            output = self._model.generate_from_batch(
                inputs, gen_config, tokenizer=self._processor.tokenizer
            )

        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        return self._processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

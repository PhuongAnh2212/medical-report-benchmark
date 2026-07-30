"""
models/llava_med.py

Placeholder wrapper for LLaVA-Med (e.g. microsoft/llava-med-v1.5-mistral-7b).

LLaVA-Med typically requires the LLaVA codebase (custom model classes not
available via a plain `AutoModel.from_pretrained` call) rather than a pure
`transformers` load. This is registered in configs/models.yaml with
`implemented: false`.

To implement:
    1. `pip install` the LLaVA-Med repo (see project README) or vendor its
       minimal model definitions under a new `third_party/llava_med/` dir.
    2. In `load()`, instantiate the LLaVA-Med model/tokenizer/image processor.
    3. In `generate()`, run the model's standard single-image chat inference.
    4. Set `implemented: true` in configs/models.yaml.
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from models.base_model import BaseReportGenerator
from utils.logging import get_logger

logger = get_logger(__name__)


class LLaVAMedReportGenerator(BaseReportGenerator):
    """Placeholder report generator for LLaVA-Med. Not yet implemented."""

    def __init__(self, config: Dict[str, Any], prompt_template: str) -> None:
        super().__init__(config, prompt_template)
        self.checkpoint = self.model_cfg.get("checkpoint", "microsoft/llava-med-v1.5-mistral-7b")

    def load(self) -> None:
        raise NotImplementedError(
            "LLaVAMedReportGenerator.load() is a placeholder. "
            f"Implement loading for checkpoint '{self.checkpoint}' using the "
            "LLaVA-Med codebase's model builder."
        )

    def generate(self, image: Image.Image) -> str:
        raise NotImplementedError(
            "LLaVAMedReportGenerator.generate() is a placeholder. "
            "Implement single-image chat inference once load() is implemented."
        )

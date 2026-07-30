"""
models/chexagent.py

Placeholder wrapper for CheXagent (e.g. StanfordAIMI/CheXagent-8b), a
domain-specific chest X-ray foundation model.

Registered in configs/models.yaml with `implemented: false`.

To implement:
    1. Load the model/tokenizer/processor per CheXagent's model card
       (typically `AutoModelForCausalLM` + a vision tower, trust_remote_code=True).
    2. In `generate()`, follow CheXagent's documented report-generation
       prompt format (it may expect a specific instruction template distinct
       from prompts/report_generation.txt -- check the model card and adjust
       `self.build_prompt()` usage accordingly).
    3. Set `implemented: true` in configs/models.yaml.
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from models.base_model import BaseReportGenerator
from utils.logging import get_logger

logger = get_logger(__name__)


class CheXagentReportGenerator(BaseReportGenerator):
    """Placeholder report generator for CheXagent. Not yet implemented."""

    def __init__(self, config: Dict[str, Any], prompt_template: str) -> None:
        super().__init__(config, prompt_template)
        self.checkpoint = self.model_cfg.get("checkpoint", "StanfordAIMI/CheXagent-8b")

    def load(self) -> None:
        raise NotImplementedError(
            "CheXagentReportGenerator.load() is a placeholder. "
            f"Implement loading for checkpoint '{self.checkpoint}' per the "
            "CheXagent model card."
        )

    def generate(self, image: Image.Image) -> str:
        raise NotImplementedError(
            "CheXagentReportGenerator.generate() is a placeholder. "
            "Implement inference once load() is implemented."
        )

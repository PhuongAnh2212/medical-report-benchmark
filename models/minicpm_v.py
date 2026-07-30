"""
models/minicpm_v.py

Placeholder wrapper for MiniCPM-V (e.g. openbmb/MiniCPM-V-2_6).

This model is registered in configs/models.yaml with `implemented: false`,
so `models.build_model()` will raise NotImplementedError until this class
is filled in. To implement:

    1. Load `AutoModel`/`AutoTokenizer` from the checkpoint with
       `trust_remote_code=True` in `load()`.
    2. In `generate()`, follow MiniCPM-V's documented `.chat()` API,
       passing the PIL image and `self.build_prompt()` as the question.
    3. Set `implemented: true` in configs/models.yaml.

No other file in the framework needs to change.
"""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from models.base_model import BaseReportGenerator
from utils.logging import get_logger

logger = get_logger(__name__)


class MiniCPMVReportGenerator(BaseReportGenerator):
    """Placeholder report generator for MiniCPM-V. Not yet implemented."""

    def __init__(self, config: Dict[str, Any], prompt_template: str) -> None:
        super().__init__(config, prompt_template)
        self.checkpoint = self.model_cfg.get("checkpoint", "openbmb/MiniCPM-V-2_6")

    def load(self) -> None:
        raise NotImplementedError(
            "MiniCPMVReportGenerator.load() is a placeholder. "
            "Implement model/tokenizer loading for checkpoint "
            f"'{self.checkpoint}' following MiniCPM-V's official usage guide."
        )

    def generate(self, image: Image.Image) -> str:
        raise NotImplementedError(
            "MiniCPMVReportGenerator.generate() is a placeholder. "
            "Implement the .chat()-style call once load() is implemented."
        )

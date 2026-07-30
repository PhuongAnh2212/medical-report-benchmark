"""
models/base_model.py

Defines the common interface every Vision-Language Model wrapper must
implement. `inference/generate_reports.py` only ever talks to this
interface, so adding a new model to the benchmark never requires touching
the inference or evaluation pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image


class BaseReportGenerator(ABC):
    """Abstract base class for all radiology report generation models.

    Subclasses must implement `load()` and `generate()`. Everything else
    (prompt loading, config storage) is handled here so subclasses stay
    focused on model-specific logic only.
    """

    def __init__(self, config: Dict[str, Any], prompt_template: str) -> None:
        """
        Args:
            config: The full benchmark config dict (see configs/default.yaml).
                    Subclasses typically read config["model"] for device,
                    dtype, generation params, etc.
            prompt_template: The raw prompt text (already loaded from the
                    file referenced in config["prompt"]["template_file"]).
        """
        self.config = config
        self.model_cfg = config.get("model", {})
        self.prompt_template = prompt_template
        self._model = None
        self._processor = None
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor/tokenizer into memory.

        Implementations should be idempotent (safe to call once and reused
        across many `generate()` calls) and should respect
        `self.model_cfg["device"]` and `self.model_cfg["dtype"]`.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, image: Image.Image) -> str:
        """Generate a radiology report string for a single chest X-ray image.

        Args:
            image: A PIL Image (already loaded and RGB-converted by the
                   caller via utils.image.load_image).

        Returns:
            The generated report as a plain string. Implementations must
            return ONLY the report text -- no chain-of-thought, no
            role-play preamble, no markdown formatting.
        """
        raise NotImplementedError

    def ensure_loaded(self) -> None:
        """Lazily call `load()` exactly once."""
        if not self._loaded:
            self.load()
            self._loaded = True

    def build_prompt(self, extra_context: Optional[str] = None) -> str:
        """Assemble the final text prompt from the template.

        Args:
            extra_context: Optional additional text appended to the base
                template (used by cot.txt / self_refine.txt strategies).

        Returns:
            The final prompt string to send to the model.
        """
        if extra_context:
            return f"{self.prompt_template}\n\n{extra_context}"
        return self.prompt_template

    @property
    def name(self) -> str:
        """Short identifier used in filenames and the leaderboard."""
        return self.model_cfg.get("name", self.__class__.__name__)

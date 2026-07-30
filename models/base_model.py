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

from utils.logging import get_logger

logger = get_logger(__name__)


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

    def unload(self) -> None:
        """Release the model/processor and free GPU memory, if any was used.

        Concrete default so existing subclasses don't have to implement
        this; override for model-specific cleanup (e.g. detaching LoRA
        adapters) and call `super().unload()` at the end. Not called by
        inference/generate_reports.py, which runs one model per process --
        provided for callers that load multiple models in one process (e.g.
        a notebook comparing models back-to-back) and need to free memory
        between them.
        """
        self._model = None
        self._processor = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def resolve_device_map(self) -> Any:
        """Resolve device placement, defaulting to multi-GPU sharding when available.

        Returns `"auto"` (accelerate's automatic multi-GPU sharding) whenever
        `model.device_map` isn't explicitly configured, `model.device`
        resolves to `"cuda"`, and more than one CUDA device is visible (e.g.
        Kaggle's GPU T4 x2 accelerator) -- otherwise returns the single
        configured device string (`"cuda"`, `"cpu"`, `"mps"`, ...). Set
        `model.device_map` in configs/default.yaml (e.g. to `"cuda:0"`) to
        force single-GPU placement even when multiple GPUs are visible.

        Returns:
            `"auto"`, a specific device string, or whatever
            `model.device_map` was explicitly set to.
        """
        configured = self.model_cfg.get("device_map")
        if configured is not None:
            return configured

        import torch

        device = self.model_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and torch.cuda.device_count() > 1:
            return "auto"
        return device

    def place_model(self, model: Any, device_map: Any) -> Any:
        """Place an already-materialized model onto `device_map`.

        For models loaded with `low_cpu_mem_usage=False` (see the
        meta-tensor workaround in models/internvl3.py / models/molmo.py),
        `from_pretrained`'s own `device_map` dispatch can't be used directly:
        transformers forces the accelerate meta-init fast path internally
        whenever `device_map` is passed to `from_pretrained`, which
        reintroduces the exact crash `low_cpu_mem_usage=False` was set to
        avoid. Instead, this loads the model normally (real tensors, fully
        materialized on CPU) and *afterward* shards the already-real model
        across GPUs via accelerate's `dispatch_model`, which needs no
        meta-init. Falls back to a single device if automatic sharding fails
        (e.g. an unusual custom architecture `infer_auto_device_map` can't
        confidently split) rather than crashing the run.

        Args:
            model: A model already loaded via `from_pretrained(low_cpu_mem_usage=False)`.
            device_map: `"auto"` for multi-GPU sharding, or a single device string.

        Returns:
            The model, placed on `device_map`.
        """
        if device_map != "auto":
            return model.to(device_map)

        try:
            from accelerate import dispatch_model, infer_auto_device_map

            computed_map = infer_auto_device_map(
                model, no_split_module_classes=getattr(model, "_no_split_modules", None)
            )
            logger.info("Sharding model across devices: %s", computed_map)
            return dispatch_model(model, device_map=computed_map)
        except Exception as exc:  # noqa: BLE001 - degrade to single-GPU rather than crash
            import torch

            fallback = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.warning(
                "Automatic multi-GPU sharding failed (%s); falling back to %s.", exc, fallback
            )
            return model.to(fallback)

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Downscale `image` to a safe pixel budget before model-specific preprocessing.

        An uncapped, full-resolution input image can blow up a VLM's vision
        token count -- and therefore attention memory -- enough to cause a
        CUDA OOM even on small checkpoints (observed with multi-megapixel
        chest X-rays). This is a blunt, model-agnostic PIL-level cap for
        wrappers whose HF processor doesn't expose a more precise
        architecture-specific knob (e.g. Qwen2-VL/2.5-VL's min_pixels/
        max_pixels, applied directly in that wrapper instead). Opt-in --
        call this from `generate()` only if needed.

        Args:
            image: RGB PIL Image as loaded by utils.image.load_image.

        Returns:
            The image, resized if it exceeded `model.max_image_side`
            (configs/default.yaml; default 1024px on the longest side).
        """
        from utils.image import resize_if_needed

        max_side = self.model_cfg.get("max_image_side") or 1024
        return resize_if_needed(image, max_side=max_side)

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

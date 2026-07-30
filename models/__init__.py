"""
models package

Provides `build_model(model_name, config, prompt_template)`, which looks up
the requested model in configs/models.yaml, dynamically imports the
corresponding module/class, and instantiates it. This indirection means
`inference/generate_reports.py` never needs an `if/elif` chain of model
names -- adding a model to configs/models.yaml is enough for it to be
runnable, as long as the class exists and implements BaseReportGenerator.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

from models.base_model import BaseReportGenerator
from utils.io import load_model_registry
from utils.logging import get_logger

logger = get_logger(__name__)


def build_model(
    model_name: str,
    config: Dict[str, Any],
    prompt_template: str,
    models_config_path: str = "configs/models.yaml",
) -> BaseReportGenerator:
    """Instantiate a report-generator model by name.

    Args:
        model_name: Key into configs/models.yaml (e.g. "qwen25_vl").
        config: Full benchmark config dict.
        prompt_template: Loaded prompt text.
        models_config_path: Path to the model registry YAML.

    Returns:
        An instantiated (but not yet `.load()`-ed) BaseReportGenerator subclass.

    Raises:
        ValueError: If model_name is not in the registry.
        NotImplementedError: If the registry marks the model as a placeholder
            (implemented: false).
        ImportError / AttributeError: If the module/class cannot be found --
            indicates a misconfigured registry entry.
    """
    registry = load_model_registry(models_config_path)
    if model_name not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {list(registry.keys())}"
        )

    entry = registry[model_name]

    if not entry.get("implemented", False):
        raise NotImplementedError(
            f"Model '{model_name}' ({entry.get('display_name', model_name)}) is a "
            f"placeholder and not yet implemented. Notes: {entry.get('notes', 'n/a')}"
        )

    module = importlib.import_module(entry["module"])
    model_class = getattr(module, entry["class_name"])

    logger.info(
        "Building model '%s' (%s) from checkpoint '%s'",
        model_name,
        entry.get("display_name", model_name),
        entry.get("checkpoint", "n/a"),
    )

    # Inject the checkpoint id into the model config section so the wrapper
    # class can read it without needing to re-parse models.yaml itself.
    config = dict(config)
    config["model"] = dict(config.get("model", {}))
    config["model"]["checkpoint"] = entry["checkpoint"]
    config["model"]["name"] = model_name

    return model_class(config=config, prompt_template=prompt_template)

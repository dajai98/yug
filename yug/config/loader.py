"""Load and validate YUG configuration.

Load order: built-in defaults (``defaults.yaml``) are read first, then
optionally overridden by a user-supplied YAML file, then the merged
result is validated. This matches Directive Section 16 ("configuration
must control ... logging, data paths") for the fields Phase 0 actually
has, without inventing sections for subsystems that don't exist yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from yug.config.schema import (
    BrainConfig,
    LoggingConfig,
    MemoryConfig,
    PathsConfig,
    ToolsConfig,
    YugConfig,
    validate,
)
from yug.utils.errors import ConfigurationError

_DEFAULTS_PATH = Path(__file__).parent / "defaults.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration file {path} must contain a YAML mapping at the top "
            f"level, got {type(data).__name__}"
        )
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config(raw: dict[str, Any]) -> YugConfig:
    logging_raw = raw.get("logging", {}) or {}
    paths_raw = raw.get("paths", {}) or {}
    brain_raw = raw.get("brain", {}) or {}
    memory_raw = raw.get("memory", {}) or {}
    tools_raw = raw.get("tools", {}) or {}

    known_top_level = {"logging", "paths", "brain", "memory", "tools"}
    unknown = set(raw.keys()) - known_top_level
    if unknown:
        raise ConfigurationError(
            f"Unknown top-level configuration section(s): {sorted(unknown)}. "
            f"Known sections: {sorted(known_top_level)}."
        )

    if "enabled" in tools_raw and isinstance(tools_raw["enabled"], list):
        tools_raw = dict(tools_raw)
        tools_raw["enabled"] = tuple(tools_raw["enabled"])

    try:
        return YugConfig(
            logging=LoggingConfig(**logging_raw),
            paths=PathsConfig(**paths_raw),
            brain=BrainConfig(**brain_raw),
            memory=MemoryConfig(**memory_raw),
            tools=ToolsConfig(**tools_raw),
        )
    except TypeError as exc:
        raise ConfigurationError(f"Invalid configuration field(s): {exc}") from exc


def load_config(user_config_path: str | Path | None = None) -> YugConfig:
    """Load, merge, and validate configuration.

    Args:
        user_config_path: Optional path to a user override YAML file.
            If given and it does not exist, raises :class:`ConfigurationError`
            (an explicitly requested override file must exist). If omitted,
            only built-in defaults are used.

    Returns:
        A validated :class:`YugConfig`.

    Raises:
        ConfigurationError: file missing/unreadable/malformed, or an
            unknown configuration section is present.
        ConfigValidationError: a known field has an invalid value.
    """
    merged = _read_yaml(_DEFAULTS_PATH)

    if user_config_path is not None:
        user_raw = _read_yaml(Path(user_config_path))
        merged = _deep_merge(merged, user_raw)

    config = _build_config(merged)
    validate(config)
    return config

"""Configuration schema and validation.

The Chief Implementation Directive lists Pydantic as an option ("where
appropriate") for schema/validation. Configuration so far (logging
level, filesystem paths, and now brain backend selection) is a
handful of scalar fields with simple, static validation rules. Stdlib
``dataclasses`` plus an explicit ``validate()`` function satisfy this
without adding a dependency; Pydantic would be justified once
configuration grows nested, dynamic, or externally-facing validation
needs (e.g. tool/security config in later phases), at which point it
should be reconsidered against that concrete need.

Phase 1 addition: ``BrainConfig``, so a brain adapter's model path and
backend are configuration-driven rather than hardcoded (Directive,
Phase 1 "LLAMA.CPP ADAPTER": "configuration-driven model path", "no
hardcoded user-specific paths").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from yug.utils.errors import ConfigValidationError

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_BRAIN_BACKENDS = frozenset({"llama_cpp", "mock", "yug_native"})


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"


@dataclass
class PathsConfig:
    """Filesystem paths YUG reads from / writes to.

    Only ``logs_dir`` is consumed by Phase 0 code. ``data_dir`` is
    included because logs are conventionally a subdirectory of the data
    root, giving a single configurable base path; models/memory paths
    (referenced by the eventual full config, Directive Section 29) are
    not added until the subsystems that use them (brain, memory) exist.
    """

    data_dir: str = "./data"
    logs_dir: str = "./data/logs"


@dataclass
class BrainConfig:
    """Brain backend selection and model location.

    Kept intentionally small: just enough for a brain adapter to be
    constructed from configuration. ``model_id`` is a free-form label
    (used e.g. in logs and ``GenerationResult.model_id`` when not
    overridden); it does not have to match a manifest entry.

    ``model_path`` defaults to ``None`` (unconfigured) rather than a
    guessed path -- Phase 0/1 ship with no bundled model, so pretending
    a default path exists would violate "no hardcoded user-specific
    paths" and "do not fabricate."

    ``checkpoint_dir`` / ``tokenizer_path`` are consumed only by the
    ``yug_native`` backend (NB-10's ``YugNativeAdapter``), mirroring
    how ``model_path``/``context_length``/``n_threads`` are consumed
    only by ``llama_cpp`` -- ``BrainConfig`` stays one flat dataclass
    with per-backend fields the other backends simply ignore, rather
    than growing a nested per-backend schema for this one addition.
    Both default to ``None`` for the same "no fabricated default path"
    reason as ``model_path``.
    """

    backend: str = "llama_cpp"
    model_id: str = "unconfigured"
    model_path: str | None = None
    context_length: int = 4096
    n_threads: int | None = None
    checkpoint_dir: str | None = None
    tokenizer_path: str | None = None


@dataclass
class MemoryConfig:
    """Working-memory storage location.

    Phase 2 scope only (see ``yug.memory`` docstring): a single JSON
    file path for :class:`yug.memory.working_memory.WorkingMemory`'s
    write-through persistence. Kept as small as ``BrainConfig`` was at
    Phase 1 -- no section for a future database backend is added until
    a concrete need for one exists.
    """

    storage_path: str = "./data/memory.json"


_DEFAULT_ENABLED_TOOLS = ("calculator", "python_exec", "file_read", "file_search")


@dataclass
class ToolsConfig:
    """Phase 3 tool subsystem configuration.

    ``enabled`` gates which registered tools :class:`yug.tools.permissions.PermissionChecker`
    will allow the executor to run (Directive Section 15/12: "maintain
    the existing autonomy model" -- this IS the whole autonomy model
    for Phase 3, an explicit allow-list, not an autonomy-level system).
    ``file_root`` bounds file_read/file_search; the size/count/timeout
    fields bound resource usage per Directive Section 15's example.
    """

    enabled: tuple[str, ...] = _DEFAULT_ENABLED_TOOLS
    file_root: str = "./data/tool_files"
    file_read_max_bytes: int = 1_048_576  # 1 MiB
    file_search_max_results: int = 100
    file_search_max_file_bytes: int = 1_048_576
    python_exec_timeout_seconds: float = 5.0


@dataclass
class YugConfig:
    """Top-level configuration."""

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)


def validate(config: YugConfig) -> None:
    """Validate a :class:`YugConfig` instance in place.

    Raises:
        ConfigValidationError: if any field fails validation.
    """
    if config.logging.level.upper() not in _VALID_LOG_LEVELS:
        raise ConfigValidationError(
            f"logging.level must be one of {sorted(_VALID_LOG_LEVELS)}, "
            f"got {config.logging.level!r}"
        )

    for field_name in ("data_dir", "logs_dir"):
        value = getattr(config.paths, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ConfigValidationError(
                f"paths.{field_name} must be a non-empty string, got {value!r}"
            )

    if config.brain.backend not in _VALID_BRAIN_BACKENDS:
        raise ConfigValidationError(
            f"brain.backend must be one of {sorted(_VALID_BRAIN_BACKENDS)}, "
            f"got {config.brain.backend!r}"
        )

    if not isinstance(config.brain.model_id, str) or not config.brain.model_id.strip():
        raise ConfigValidationError(
            f"brain.model_id must be a non-empty string, got {config.brain.model_id!r}"
        )

    if config.brain.model_path is not None and not str(config.brain.model_path).strip():
        raise ConfigValidationError(
            "brain.model_path must be a non-empty string when provided, or omitted/null"
        )

    if config.brain.checkpoint_dir is not None and not str(config.brain.checkpoint_dir).strip():
        raise ConfigValidationError(
            "brain.checkpoint_dir must be a non-empty string when provided, or omitted/null"
        )

    if config.brain.tokenizer_path is not None and not str(config.brain.tokenizer_path).strip():
        raise ConfigValidationError(
            "brain.tokenizer_path must be a non-empty string when provided, or omitted/null"
        )

    if not isinstance(config.brain.context_length, int) or config.brain.context_length <= 0:
        raise ConfigValidationError(
            f"brain.context_length must be a positive integer, got "
            f"{config.brain.context_length!r}"
        )

    if config.brain.n_threads is not None and (
        not isinstance(config.brain.n_threads, int) or config.brain.n_threads <= 0
    ):
        raise ConfigValidationError(
            f"brain.n_threads must be a positive integer or null, got "
            f"{config.brain.n_threads!r}"
        )

    if not isinstance(config.memory.storage_path, str) or not config.memory.storage_path.strip():
        raise ConfigValidationError(
            f"memory.storage_path must be a non-empty string, got {config.memory.storage_path!r}"
        )

    if not isinstance(config.tools.enabled, (tuple, list)) or not all(
        isinstance(name, str) and name for name in config.tools.enabled
    ):
        raise ConfigValidationError(
            f"tools.enabled must be a sequence of non-empty strings, got {config.tools.enabled!r}"
        )

    if not isinstance(config.tools.file_root, str) or not config.tools.file_root.strip():
        raise ConfigValidationError(
            f"tools.file_root must be a non-empty string, got {config.tools.file_root!r}"
        )

    for field_name in ("file_read_max_bytes", "file_search_max_results", "file_search_max_file_bytes"):
        value = getattr(config.tools, field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigValidationError(
                f"tools.{field_name} must be a positive integer, got {value!r}"
            )

    timeout = config.tools.python_exec_timeout_seconds
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ConfigValidationError(
            f"tools.python_exec_timeout_seconds must be a positive number, got {timeout!r}"
        )

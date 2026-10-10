"""Minimal model manifest mechanism.

Directive scope: "Create a minimal manifest mechanism... Do not build a
sophisticated 'model zoo' yet." This is intentionally just a dataclass
plus a JSON loader -- no download management, no checksum verification
enforcement, no fallback chains. Those are explicit M2+ concerns
(Architecture Decision Report, Section 4.7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from yug.brain.errors import ModelLoadError


@dataclass(frozen=True)
class ModelManifest:
    """Describes one model a brain adapter could load.

    ``parameter_count`` is informational only (see
    ``ModelCapabilities`` docstring) -- never used for runtime
    branching.
    """

    model_id: str
    provider: str  # backend/adapter name, e.g. "llama_cpp"
    model_path: str
    context_length: int
    parameter_count: int | None = None
    quantization: str | None = None
    checksum: str | None = None


def load_manifests(path: str | Path) -> list[ModelManifest]:
    """Load a list of manifests from a JSON file.

    The file must contain a JSON array of objects with at least
    ``model_id``, ``provider``, ``model_path``, and ``context_length``.
    An empty array is valid (no manifests registered yet).

    Raises:
        ModelLoadError: if the file is missing, is not valid JSON, is
            not a JSON array, or an entry is missing a required field.
    """
    manifest_path = Path(path)
    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Manifest file not found: {manifest_path}") from exc

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"Invalid JSON in manifest file {manifest_path}: {exc}") from exc

    if not isinstance(raw, list):
        raise ModelLoadError(
            f"Manifest file {manifest_path} must contain a JSON array, "
            f"got {type(raw).__name__}"
        )

    manifests: list[ModelManifest] = []
    required = {"model_id", "provider", "model_path", "context_length"}
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ModelLoadError(f"Manifest entry {i} in {manifest_path} is not an object")
        missing = required - entry.keys()
        if missing:
            raise ModelLoadError(
                f"Manifest entry {i} in {manifest_path} is missing required "
                f"field(s): {sorted(missing)}"
            )
        manifests.append(
            ModelManifest(
                model_id=entry["model_id"],
                provider=entry["provider"],
                model_path=entry["model_path"],
                context_length=entry["context_length"],
                parameter_count=entry.get("parameter_count"),
                quantization=entry.get("quantization"),
                checksum=entry.get("checksum"),
            )
        )
    return manifests


def get_manifest(manifests: list[ModelManifest], model_id: str) -> ModelManifest:
    """Find a manifest by ``model_id``.

    Raises:
        ModelLoadError: if no manifest with that ``model_id`` exists.
    """
    for manifest in manifests:
        if manifest.model_id == model_id:
            return manifest
    raise ModelLoadError(f"No manifest registered for model_id={model_id!r}")

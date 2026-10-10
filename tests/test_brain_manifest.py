from __future__ import annotations

import json
from pathlib import Path

import pytest

import yug.brain as _brain_pkg
from yug.brain.errors import ModelLoadError
from yug.brain.manifest import ModelManifest, get_manifest, load_manifests

# Resolved from the imported package's own location rather than a
# filesystem path relative to this test file. The top-level test tree
# (tests/) and the source tree (SOURCE/yug/) are not siblings in this
# repo's layout, so a Path(__file__)-relative guess is layout-fragile;
# going through the package's __file__ is correct under any layout,
# including an installed distribution.
_SHIPPED_MANIFEST_PATH = Path(_brain_pkg.__file__).parent / "model_zoo" / "manifests.json"


def test_shipped_manifest_file_loads_as_empty_list():
    """No model is bundled with Phase 1 -- the shipped manifest file
    must be a valid, empty registry, not a fabricated model entry."""
    manifests = load_manifests(_SHIPPED_MANIFEST_PATH)
    assert manifests == []


def test_load_manifests_parses_valid_entries(tmp_path):
    path = tmp_path / "manifests.json"
    path.write_text(
        json.dumps(
            [
                {
                    "model_id": "test-model",
                    "provider": "llama_cpp",
                    "model_path": "/models/test.gguf",
                    "context_length": 4096,
                    "parameter_count": 500_000_000,
                    "quantization": "Q4_K_M",
                }
            ]
        )
    )

    manifests = load_manifests(path)

    assert len(manifests) == 1
    assert manifests[0] == ModelManifest(
        model_id="test-model",
        provider="llama_cpp",
        model_path="/models/test.gguf",
        context_length=4096,
        parameter_count=500_000_000,
        quantization="Q4_K_M",
        checksum=None,
    )


def test_load_manifests_missing_file_raises():
    with pytest.raises(ModelLoadError):
        load_manifests("/nonexistent/manifests.json")


def test_load_manifests_malformed_json_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")

    with pytest.raises(ModelLoadError):
        load_manifests(path)


def test_load_manifests_non_array_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"model_id": "test-model"}')

    with pytest.raises(ModelLoadError):
        load_manifests(path)


def test_load_manifests_missing_required_field_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"model_id": "test-model"}]))

    with pytest.raises(ModelLoadError):
        load_manifests(path)


def test_get_manifest_found():
    manifests = [
        ModelManifest(model_id="a", provider="llama_cpp", model_path="/a.gguf", context_length=1024),
        ModelManifest(model_id="b", provider="llama_cpp", model_path="/b.gguf", context_length=2048),
    ]
    assert get_manifest(manifests, "b").model_path == "/b.gguf"


def test_get_manifest_not_found_raises():
    with pytest.raises(ModelLoadError):
        get_manifest([], "missing-model")

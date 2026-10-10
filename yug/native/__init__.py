"""YUG native brain code: model configuration, architecture, and
inference runtime -- everything the production system needs to
execute a YUG-owned model.

Per ``docs/architecture/NATIVE_BRAIN_ARCHITECTURE.md`` (Architectural
Decision #16), this package explicitly excludes training loops,
datasets, optimizers, and training-only evaluation code, which live in
the top-level ``training/`` package instead. ``training/`` may import
from here; this package must never import from ``training/``.

As of NB-03, this package contains only model *configuration*
(``yug.native.config``) -- no architecture/forward-pass/inference code
exists yet (that is NB-04 onward).

NB-04 adds ``yug.native.model`` (``YugTransformer`` and its
components), which requires PyTorch. It is deliberately NOT re-exported
here: `import yug.native` / `from yug.native import ModelConfig` must
keep working with zero PyTorch dependency, for anyone who only needs
model *configuration* (e.g. inspecting/serializing a config, estimating
parameter count) without installing PyTorch. Import
``yug.native.model`` explicitly to opt into the PyTorch dependency.
"""

from yug.native.config import ModelConfig, ParameterEstimate, estimate_parameter_count, validate
from yug.native.errors import ModelConfigSerializationError, ModelConfigValidationError, NativeError

__all__ = [
    "ModelConfig",
    "ParameterEstimate",
    "estimate_parameter_count",
    "validate",
    "NativeError",
    "ModelConfigValidationError",
    "ModelConfigSerializationError",
]

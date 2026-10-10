"""M1 composition root: the ONE place concrete brain adapters are
selected from configuration.

This module exists to close a single, narrow gap: nothing previously
assembled ``Config -> brain adapter -> YugCore`` outside of test code
(see ``tests/test_brain_config_integration.py``, whose own comment
says this dispatch is "real YUGCore wiring (Phase 6)"). Every piece
being wired together here (``BrainConfig``, ``MockBrainAdapter``,
``LlamaCppAdapter``, ``WorkingMemory``, ``YugCore``) already exists and
is already independently tested -- this module adds no new behavior to
any of them.

Architectural boundary this module exists to preserve (Directive:
"YUG -> BrainInterface -> Brain Adapter -> Backend/Model", never
"YUG -> llama.cpp"): this is the ONLY production module allowed to
import a concrete adapter. ``yug.runtime.core.YugCore`` imports only
``yug.brain.interface.BrainInterface`` -- it has no idea which adapter
class this module chose, and must never be given one. See
``tests/test_composition.py::test_yugcore_does_not_import_concrete_adapters``
for the check that keeps this true.

Backend selection is a plain dict keyed by ``BrainConfig.backend`` --
deliberately not a bigger dependency-injection framework, matching
this milestone's own scope limit. Adding a future backend (e.g. an
eventual own-model adapter) means adding one entry here and one value
to ``yug.config.schema._VALID_BRAIN_BACKENDS`` -- nothing else in this
module, or in ``YugCore``/``WorkingMemory``/tools/verification, changes.

Composition wiring: ``yug_native`` closes exactly this gap for
``YugNativeAdapter`` (NB-10) -- it was written and independently tested
(``tests/test_brain_yug_native_adapter.py``) but was never registered
here, so it was previously unreachable through ``YugConfig``. One
deliberate deviation from ``mock``/``llama_cpp``'s pattern is required
by ``YugNativeAdapter.__init__``'s own (frozen, unmodified) signature:
it takes a live tokenizer *instance*, not a path, so
``_build_yug_native_adapter`` below must load the tokenizer eagerly,
here, rather than deferring all filesystem access into the adapter's
own lazy-load path the way the other two factories do. This is a
consequence of an interface this module does not own and was not
asked to change -- not a new design choice -- and is confined to this
one factory function.
"""

from __future__ import annotations

from collections.abc import Callable

from yug.brain.adapters.llama_cpp import LlamaCppAdapter
from yug.brain.adapters.mock import MockBrainAdapter
from yug.brain.adapters.yug_native import YugNativeAdapter
from yug.brain.interface import BrainInterface
from yug.config.schema import BrainConfig, YugConfig
from yug.memory.working_memory import WorkingMemory
from yug.runtime.core import YugCore
from yug.tokenizer.bpe import ByteLevelBpeTokenizer
from yug.tokenizer.errors import SerializationError
from yug.utils.errors import ConfigValidationError


def _build_yug_native_adapter(cfg: BrainConfig) -> YugNativeAdapter:
    """Construct :class:`YugNativeAdapter` from ``BrainConfig``.

    Unlike the ``mock``/``llama_cpp`` factories, this one touches the
    filesystem before returning -- ``YugNativeAdapter.__init__``
    (frozen, NB-10, not modified here) requires an already-constructed
    tokenizer *instance*, not a path, so there is no lazy-load path
    available to defer this into. Importing ``yug.tokenizer.bpe`` here
    has no PyTorch cost (the tokenizer module has none), so this does
    not weaken the "native extras stay optional" property the adapter
    itself otherwise preserves for the model/generation code.

    Raises:
        ConfigValidationError: ``checkpoint_dir`` or ``tokenizer_path``
            is unset -- both are required to construct this backend,
            unlike ``llama_cpp.model_path``, which the adapter itself
            can leave unset and fail on lazily at load time (this
            backend's constructor cannot: it needs a real tokenizer
            object up front, not a path it could defer resolving).
    """
    if not cfg.checkpoint_dir:
        raise ConfigValidationError(
            "brain.checkpoint_dir is required when brain.backend='yug_native'"
        )
    if not cfg.tokenizer_path:
        raise ConfigValidationError(
            "brain.tokenizer_path is required when brain.backend='yug_native'"
        )

    try:
        tokenizer = ByteLevelBpeTokenizer.load(cfg.tokenizer_path)
    except SerializationError as exc:
        raise ConfigValidationError(
            f"brain.tokenizer_path {cfg.tokenizer_path!r} could not be loaded: {exc}"
        ) from exc

    return YugNativeAdapter(
        checkpoint_dir=cfg.checkpoint_dir,
        tokenizer=tokenizer,
        model_id=cfg.model_id,
    )


# Each factory takes the validated BrainConfig and returns a fresh
# BrainInterface implementation. mock/llama_cpp defer all real loading
# to first use, as before; yug_native cannot fully defer construction
# (see _build_yug_native_adapter's own docstring) but still defers the
# expensive part -- loading model weights -- to first use exactly like
# the other two.
_BACKEND_FACTORIES: dict[str, Callable[[BrainConfig], BrainInterface]] = {
    "mock": lambda cfg: MockBrainAdapter(model_id=cfg.model_id),
    "llama_cpp": lambda cfg: LlamaCppAdapter(
        model_path=cfg.model_path,
        model_id=cfg.model_id,
        context_length=cfg.context_length,
        n_threads=cfg.n_threads,
    ),
    "yug_native": _build_yug_native_adapter,
}


def build_brain(brain_config: BrainConfig) -> BrainInterface:
    """Construct the concrete adapter named by ``brain_config.backend``.

    This is a second, independent check of ``backend`` -- ``load_config()``
    already validates it against the same set of known backends before
    this function is normally reached, but ``build_brain`` must not
    silently trust an unvalidated ``BrainConfig`` a caller constructed
    by hand (e.g. directly, in a test, without going through
    ``load_config``/``validate``). An unknown backend is never silently
    mapped to a default; it fails explicitly here exactly as it would
    at config-validation time.

    Raises:
        ConfigValidationError: ``brain_config.backend`` does not name a
            known adapter.
    """
    factory = _BACKEND_FACTORIES.get(brain_config.backend)
    if factory is None:
        raise ConfigValidationError(
            f"brain.backend must be one of {sorted(_BACKEND_FACTORIES)}, "
            f"got {brain_config.backend!r}"
        )
    return factory(brain_config)


def build_core(config: YugConfig) -> YugCore:
    """Construct a fully-wired ``YugCore`` from a validated ``YugConfig``.

    Scope is deliberately minimal (Config -> Brain -> YugCore only):
    no ``ToolExecutor`` and no ``VerificationEngine`` are constructed or
    attached here. Both remain optional constructor arguments on
    ``YugCore`` and are out of scope for this milestone -- wiring them
    in is a separate, later decision, not a byproduct of this one.
    """
    brain = build_brain(config.brain)
    memory = WorkingMemory(storage_path=config.memory.storage_path)
    return YugCore(brain, memory)

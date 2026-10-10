"""YugCore: the composition layer joining BrainInterface, WorkingMemory,
and (optionally) ToolExecutor.

Pipeline, matching this milestone's own architecture diagram exactly:

    request -> load conversation history from memory -> append user turn
    -> construct brain input (plain history -> Message objects)
    -> call BrainInterface.generate() -> obtain response
    -> persist the updated history (user + assistant turns)
    -> optionally execute an EXPLICITLY named tool (never brain-selected)
    -> optionally run an EXPLICITLY named verification (never automatic)
    -> return a structured Response

Three design decisions worth stating explicitly, since none is
dictated by the milestone brief and all three affect testable
behavior:

1. Tool execution happens AFTER the brain call and is NOT gated on the
   brain having said anything about tools -- ``tool_name`` is a
   parameter the CALLER supplies to ``respond()``, never something
   derived from ``GenerationResult.tool_calls``. There is no code path
   here from brain output to tool execution; that is the exact
   boundary that keeps this a runtime and not an agent.
2. Conversation history persisted to WorkingMemory contains ONLY
   {"role", "content"} turns (JSON-serializable, matching
   WorkingMemory's own value constraint -- a frozen ``Message``
   dataclass is not itself JSON-serializable, which is why this module
   converts explicitly at the boundary rather than storing ``Message``
   objects directly). A tool result is never folded into history as a
   fabricated conversation turn; it is returned separately on
   ``Response.tool_result``. Inventing a "tool" history-role
   convention that nothing else in the repository defines would be
   exactly the kind of speculative protocol this milestone is told not
   to invent.
3. Verification is wired in with EXACTLY the same shape and the exact
   same non-authority boundary as tool execution: ``verify_name`` /
   ``verify_parameters`` are parameters the CALLER supplies, never
   derived from brain output or from ``tool_result``. YugCore does not
   decide what gets verified or against what -- it only dispatches an
   explicit, caller-specified check and reports the outcome. This
   keeps ``yug.verification`` exactly as independent inside the
   runtime as it is documented to be on its own
   (:mod:`yug.verification.engine`'s own docstring: "No automatic
   invocation, no brain, no Agent, no tool execution"). Unlike tool
   execution, a malformed verifier call (unknown verifier name, or a
   malformed schema/expression) is NOT swallowed into a failed result
   here -- ``VerificationEngine.verify`` itself deliberately raises for
   malformed input rather than returning a sentinel (see
   :mod:`yug.verification.errors`), and this module preserves that
   distinction unchanged rather than papering over it. The only case
   turned into a structured, non-raising failure is the one genuinely
   new at this layer: no ``VerificationEngine`` configured at all,
   mirroring exactly how a missing ``ToolExecutor`` is handled below.
"""

from __future__ import annotations

import asyncio
from typing import Any

from yug.brain.errors import BrainError
from yug.brain.interface import BrainInterface
from yug.brain.types import Message
from yug.memory.working_memory import WorkingMemory
from yug.runtime.errors import CoreRuntimeError
from yug.runtime.types import Response
from yug.tools.executor import ToolExecutor
from yug.tools.types import ToolResult
from yug.verification.engine import VerificationEngine
from yug.verification.types import VerificationResult

_DEFAULT_HISTORY_KEY = "conversation_history"


class YugCore:
    """Explicitly constructed -- no hidden global runtime instance,
    matching every other subsystem's "no hidden state" convention.

    Args:
        brain: any :class:`BrainInterface` implementation. YugCore
            depends only on this abstraction -- it never imports or
            references a concrete adapter.
        memory: a :class:`WorkingMemory` instance. Conversation history
            is stored under ``history_key`` in this instance; whether
            that persists across a real restart depends entirely on
            whether ``memory`` itself was constructed with a
            ``storage_path`` (YugCore does not care and does not
            duplicate that decision).
        executor: optional. If ``None``, an explicit ``tool_name`` in
            :meth:`respond` produces a failed ``ToolResult`` (not an
            exception) rather than a silent no-op or a bypass of the
            permission/validation pipeline.
        verification_engine: optional. If ``None``, an explicit
            ``verify_name`` in :meth:`respond` produces a failed
            ``VerificationResult`` (not an exception) -- the same
            "missing collaborator is a structured failure, not a
            silent no-op" convention as ``executor`` above.
        history_key: the WorkingMemory key conversation history is
            stored under. Configurable so multiple independent
            conversations can share one WorkingMemory instance.
        temperature, max_tokens: forwarded to
            ``BrainInterface.generate()`` unchanged on every call.
    """

    def __init__(
        self,
        brain: BrainInterface,
        memory: WorkingMemory,
        executor: ToolExecutor | None = None,
        verification_engine: VerificationEngine | None = None,
        *,
        history_key: str = _DEFAULT_HISTORY_KEY,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self._brain = brain
        self._memory = memory
        self._executor = executor
        self._verification_engine = verification_engine
        self._history_key = history_key
        self._temperature = temperature
        self._max_tokens = max_tokens

    def respond(
        self,
        request: str,
        *,
        tool_name: str | None = None,
        tool_parameters: dict[str, Any] | None = None,
        verify_name: str | None = None,
        verify_parameters: dict[str, Any] | None = None,
    ) -> Response:
        """Process one user request through the brain, persist the
        resulting conversation turn, and optionally execute an
        explicitly-named tool and/or run an explicitly-named
        verification.

        Args:
            verify_name: ``None`` (default, no verification is run),
                ``"schema"``, or ``"math"`` -- forwarded verbatim to
                :meth:`yug.verification.engine.VerificationEngine.verify`.
                Never derived from ``request``, brain output, or
                ``tool_result`` -- the caller decides what, if
                anything, gets verified and supplies its exact
                arguments via ``verify_parameters``, exactly as it
                already does for ``tool_name``/``tool_parameters``.
            verify_parameters: keyword arguments for the selected
                verifier (e.g. ``{"data": ..., "schema": ...}`` for
                ``"schema"``, or ``{"expression": ..., "expected": ...}``
                for ``"math"``). Ignored if ``verify_name`` is ``None``.

        Raises:
            CoreRuntimeError: ``request`` is not a non-empty string.
                (Brain failures do NOT raise -- they are reported via
                ``Response.success=False`` / ``Response.error``, so a
                caller does not need a try/except around every call
                just to handle an ordinary generation failure.)
            VerifierNotFoundError, SchemaVerificationError,
                MathVerificationError: propagated unchanged from
                ``VerificationEngine.verify`` for a malformed
                ``verify_name``/``verify_parameters`` call -- NOT
                caught here. See the module docstring (point 3) for
                why this one case deliberately does not follow the
                "structured failure, never an exception" convention
                that the rest of ``respond()`` uses.
        """
        if not isinstance(request, str) or not request.strip():
            raise CoreRuntimeError(f"request must be a non-empty string, got {request!r}")

        history = self._load_history()
        history.append({"role": "user", "content": request})
        messages = [Message(role=turn["role"], content=turn["content"]) for turn in history]

        try:
            generation = asyncio.run(
                self._brain.generate(
                    messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
            )
        except BrainError as exc:
            # Deliberately do NOT persist here: the user turn was
            # appended to a local list only, never saved. A failed
            # round-trip must not leave a dangling user message with
            # no assistant reply in the persisted history.
            return Response(
                success=False,
                content="",
                request_text=request,
                error=f"{type(exc).__name__}: {exc}",
                conversation_key=self._history_key,
            )
        except Exception as exc:  # noqa: BLE001 -- deliberate safety net,
            # mirroring yug.tools.executor.ToolExecutor's own convention:
            # an unexpected bug in a brain adapter surfaces as a
            # structured failure, not an exception escaping respond()'s
            # documented API.
            return Response(
                success=False,
                content="",
                request_text=request,
                error=f"{type(exc).__name__}: {exc}",
                conversation_key=self._history_key,
            )

        history.append({"role": "assistant", "content": generation.content})
        self._save_history(history)

        tool_result: ToolResult | None = None
        if tool_name is not None:
            if self._executor is None:
                tool_result = ToolResult(
                    success=False,
                    error="No ToolExecutor configured for this YugCore instance",
                )
            else:
                tool_result = self._executor.execute(tool_name, tool_parameters or {})

        verification_result: VerificationResult | None = None
        if verify_name is not None:
            if self._verification_engine is None:
                verification_result = VerificationResult(
                    passed=False,
                    verifier=verify_name,
                    reason="No VerificationEngine configured for this YugCore instance",
                )
            else:
                # Deliberately NOT wrapped in try/except: malformed
                # verifier input (VerifierNotFoundError,
                # SchemaVerificationError, MathVerificationError) must
                # propagate unchanged, exactly as
                # VerificationEngine.verify() itself documents. See
                # the module docstring, point 3.
                verification_result = self._verification_engine.verify(
                    verify_name, **(verify_parameters or {})
                )

        return Response(
            success=True,
            content=generation.content,
            request_text=request,
            tool_result=tool_result,
            verification_result=verification_result,
            conversation_key=self._history_key,
        )

    def _load_history(self) -> list[dict[str, str]]:
        if self._memory.contains(self._history_key):
            return list(self._memory.get(self._history_key))
        return []

    def _save_history(self, history: list[dict[str, str]]) -> None:
        self._memory.set(self._history_key, history)

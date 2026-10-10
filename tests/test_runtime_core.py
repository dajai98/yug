from __future__ import annotations

import pytest

from yug.brain.adapters.mock import MockBrainAdapter
from yug.brain.errors import BrainError
from yug.brain.interface import BrainInterface
from yug.brain.types import GenerationResult, HealthState, HealthStatus, ModelCapabilities
from yug.memory.working_memory import WorkingMemory
from yug.runtime.core import YugCore
from yug.runtime.errors import CoreRuntimeError


class _EchoBrain(BrainInterface):
    """A second, independent BrainInterface implementation -- used to
    prove YugCore depends on the abstraction, not on MockBrainAdapter
    specifically."""

    async def generate(self, messages, tools=None, temperature=0.7, max_tokens=1024, stop_sequences=None):
        return GenerationResult(content=f"ECHO:{messages[-1].content}", finish_reason="stop", model_id="echo")

    async def embed(self, texts):
        return []

    def get_capabilities(self):
        return ModelCapabilities(
            model_id="echo", context_length=100, supports_tools=False,
            supports_vision=False, embedding_dim=None,
        )

    def health_check(self):
        return HealthStatus(state=HealthState.HEALTHY, detail="ok", model_loaded=True)


class _FailingBrain(BrainInterface):
    async def generate(self, messages, tools=None, temperature=0.7, max_tokens=1024, stop_sequences=None):
        raise BrainError("simulated backend failure")

    async def embed(self, texts):
        return []

    def get_capabilities(self):
        return ModelCapabilities(
            model_id="failing", context_length=100, supports_tools=False,
            supports_vision=False, embedding_dim=None,
        )

    def health_check(self):
        return HealthStatus(state=HealthState.UNAVAILABLE, detail="down", model_loaded=False)


def test_successful_request_returns_brain_content():
    core = YugCore(MockBrainAdapter(), WorkingMemory())

    response = core.respond("hello there")

    assert response.success is True
    assert "hello there" in response.content


def test_brain_receives_accumulated_conversation_context():
    memory = WorkingMemory()
    core = YugCore(MockBrainAdapter(), memory)
    core.respond("first")

    second = core.respond("second")

    # MockBrainAdapter reports how many messages it received.
    assert "3 message(s)" in second.content  # user1, assistant1, user2


def test_runtime_works_with_a_different_brain_interface_implementation():
    """Concrete adapter must not be hardcoded into YugCore."""
    core = YugCore(_EchoBrain(), WorkingMemory())

    response = core.respond("swap test")

    assert response.content == "ECHO:swap test"


def test_brain_failure_is_surfaced_not_raised():
    core = YugCore(_FailingBrain(), WorkingMemory())

    response = core.respond("this will fail")

    assert response.success is False
    assert response.error is not None
    assert response.content == ""


def test_brain_failure_does_not_persist_the_dangling_user_turn():
    memory = WorkingMemory()
    core = YugCore(_FailingBrain(), memory)

    core.respond("this will fail")

    assert memory.contains("conversation_history") is False


@pytest.mark.parametrize("bad_request", ["", "   ", None, 123, []])
def test_malformed_request_raises_core_runtime_error(bad_request):
    core = YugCore(MockBrainAdapter(), WorkingMemory())

    with pytest.raises(CoreRuntimeError):
        core.respond(bad_request)


def test_response_reports_the_conversation_key():
    core = YugCore(MockBrainAdapter(), WorkingMemory(), history_key="custom_key")

    response = core.respond("hi")

    assert response.conversation_key == "custom_key"

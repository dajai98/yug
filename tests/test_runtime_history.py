from __future__ import annotations

from yug.brain.adapters.mock import MockBrainAdapter
from yug.memory.working_memory import WorkingMemory
from yug.runtime.core import YugCore


def test_user_and_assistant_turns_persist_to_memory():
    memory = WorkingMemory()
    core = YugCore(MockBrainAdapter(), memory)

    core.respond("first message")

    history = memory.get("conversation_history")
    assert history[0] == {"role": "user", "content": "first message"}
    assert history[1]["role"] == "assistant"


def test_history_ordering_is_deterministic_across_turns():
    memory = WorkingMemory()
    core = YugCore(MockBrainAdapter(), memory)

    core.respond("a")
    core.respond("b")

    roles = [turn["role"] for turn in memory.get("conversation_history")]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_conversation_survives_a_fresh_runtime_instance(tmp_path):
    """The actual restart scenario, matching Phase 2's own test
    convention: discard both YugCore and WorkingMemory, construct
    fresh instances against the same storage_path, and confirm the
    conversation is still there."""
    storage_path = str(tmp_path / "conversation.json")

    memory_one = WorkingMemory(storage_path=storage_path)
    core_one = YugCore(MockBrainAdapter(), memory_one)
    core_one.respond("remember this")
    del core_one, memory_one

    memory_two = WorkingMemory(storage_path=storage_path)
    core_two = YugCore(MockBrainAdapter(), memory_two)

    history = memory_two.get("conversation_history")
    assert history[0]["content"] == "remember this"

    response = core_two.respond("second, after restart")
    assert "3 message(s)" in response.content


def test_two_runtimes_with_separate_memory_are_isolated():
    core_a = YugCore(MockBrainAdapter(), WorkingMemory())
    core_b = YugCore(MockBrainAdapter(), WorkingMemory())

    core_a.respond("only in a")
    response_b = core_b.respond("fresh for b")

    assert "1 message(s)" in response_b.content  # b never saw a's history


def test_history_is_plain_json_serializable_dicts_not_message_objects():
    """WorkingMemory rejects non-JSON-serializable values; this
    confirms the runtime stores plain dicts, not Message dataclass
    instances, at the persistence boundary."""
    import json

    memory = WorkingMemory()
    core = YugCore(MockBrainAdapter(), memory)
    core.respond("hi")

    history = memory.get("conversation_history")
    json.dumps(history)  # must not raise
    assert all(isinstance(turn, dict) for turn in history)

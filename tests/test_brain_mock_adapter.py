from __future__ import annotations

import asyncio

from yug.brain.adapters.mock import MockBrainAdapter
from yug.brain.interface import BrainInterface
from yug.brain.types import HealthState, Message


def test_mock_adapter_satisfies_brain_interface():
    assert isinstance(MockBrainAdapter(), BrainInterface)


def test_generate_marks_output_as_synthetic():
    adapter = MockBrainAdapter()
    result = asyncio.run(
        adapter.generate([Message(role="user", content="hello there")])
    )
    assert "[mock-brain]" in result.content
    assert "hello there" in result.content
    assert result.finish_reason == "stop"
    assert result.model_id == "mock-brain-v1"


def test_generate_uses_last_user_message():
    adapter = MockBrainAdapter()
    messages = [
        Message(role="system", content="be nice"),
        Message(role="user", content="first"),
        Message(role="assistant", content="ok"),
        Message(role="user", content="second"),
    ]
    result = asyncio.run(adapter.generate(messages))
    assert "'second'" in result.content


def test_embed_is_deterministic():
    adapter = MockBrainAdapter()
    e1 = asyncio.run(adapter.embed(["same text"]))[0]
    e2 = asyncio.run(adapter.embed(["same text"]))[0]
    assert e1.values == e2.values


def test_embed_dimension_matches_capabilities():
    adapter = MockBrainAdapter()
    caps = adapter.get_capabilities()
    embeddings = asyncio.run(adapter.embed(["text"]))
    assert embeddings[0].dim == caps.embedding_dim


def test_health_check_always_healthy_no_dependency():
    adapter = MockBrainAdapter()
    status = adapter.health_check()
    assert status.state == HealthState.HEALTHY
    assert status.model_loaded is True


def test_get_capabilities_documents_synthetic_nature():
    adapter = MockBrainAdapter()
    caps = adapter.get_capabilities()
    assert any("synthetic" in limitation for limitation in caps.known_limitations)

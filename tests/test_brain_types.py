from __future__ import annotations

from yug.brain.types import (
    Embedding,
    GenerationResult,
    HealthState,
    HealthStatus,
    Message,
    ModelCapabilities,
    ToolDefinition,
)


def test_message_is_a_plain_role_content_pair():
    m = Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"


def test_tool_definition_defaults_to_empty_parameters():
    t = ToolDefinition(name="calculator", description="adds numbers")
    assert t.parameters == {}


def test_generation_result_optional_fields_default_to_none():
    r = GenerationResult(content="hi", finish_reason="stop", model_id="m1")
    assert r.tool_calls is None
    assert r.usage is None


def test_embedding_dim_matches_values_length():
    e = Embedding(values=(0.1, 0.2, 0.3), model_id="m1")
    assert e.dim == 3


def test_model_capabilities_parameter_count_is_optional():
    caps = ModelCapabilities(
        model_id="m1",
        context_length=4096,
        supports_tools=False,
        supports_vision=False,
        embedding_dim=None,
    )
    assert caps.parameter_count is None
    assert caps.recommended_temperature_range == (0.0, 1.0)


def test_health_status_requires_explicit_state():
    status = HealthStatus(state=HealthState.DEGRADED, detail="not loaded yet", model_loaded=False)
    assert status.state == HealthState.DEGRADED
    assert status.model_loaded is False

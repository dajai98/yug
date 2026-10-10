from __future__ import annotations

import pytest

from yug.brain.adapters.mock import MockBrainAdapter
from yug.memory.working_memory import WorkingMemory
from yug.runtime.core import YugCore
from yug.verification.engine import VerificationEngine
from yug.verification.errors import MathVerificationError, VerifierNotFoundError


def _core(engine):
    return YugCore(MockBrainAdapter(), WorkingMemory(), verification_engine=engine)


def test_explicit_math_verification_reaches_the_engine_and_passes():
    core = _core(VerificationEngine())

    response = core.respond(
        "what is 2+2",
        verify_name="math",
        verify_parameters={"expression": "2 + 2", "expected": 4},
    )

    assert response.success is True  # brain call succeeds independently
    assert response.verification_result.passed is True
    assert response.verification_result.verifier == "math"


def test_explicit_schema_verification_reaches_the_engine_and_fails():
    core = _core(VerificationEngine())

    response = core.respond(
        "check shape",
        verify_name="schema",
        verify_parameters={
            "data": {"a": "not an integer"},
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            },
        },
    )

    assert response.success is True
    assert response.verification_result.passed is False
    assert response.verification_result.verifier == "schema"


def test_no_verification_engine_configured_produces_structured_failure_not_exception():
    core = YugCore(MockBrainAdapter(), WorkingMemory())  # verification_engine=None

    response = core.respond("check please", verify_name="math", verify_parameters={"expression": "1+1", "expected": 2})

    assert response.verification_result.passed is False
    assert "No VerificationEngine" in response.verification_result.reason
    assert response.success is True  # the brain part still worked


def test_no_verify_name_means_no_verification_result():
    core = _core(VerificationEngine())

    response = core.respond("just chatting, no verification")

    assert response.verification_result is None


def test_unknown_verifier_name_propagates_instead_of_becoming_a_failed_result():
    core = _core(VerificationEngine())

    with pytest.raises(VerifierNotFoundError):
        core.respond("check please", verify_name="nonexistent_verifier", verify_parameters={})


def test_malformed_verifier_input_propagates_instead_of_becoming_a_failed_result():
    core = _core(VerificationEngine())

    with pytest.raises(MathVerificationError):
        core.respond(
            "check please",
            verify_name="math",
            verify_parameters={"expression": "1 + 1", "expected": "not a number"},
        )


def test_tool_and_verification_can_both_be_requested_in_the_same_call():
    core = _core(VerificationEngine())

    response = core.respond(
        "do both",
        verify_name="math",
        verify_parameters={"expression": "3 * 3", "expected": 9},
    )

    assert response.tool_result is None  # no tool_name was supplied
    assert response.verification_result.passed is True

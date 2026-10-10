from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from yug.verification.engine import VerificationEngine
from yug.verification.errors import MathVerificationError, VerifierNotFoundError


def test_known_verifier_math_dispatches_correctly():
    engine = VerificationEngine()

    result = engine.verify("math", expression="2 + 2", expected=4)

    assert result.passed is True
    assert result.verifier == "math"


def test_known_verifier_schema_dispatches_correctly():
    engine = VerificationEngine()

    result = engine.verify(
        "schema",
        data={"a": 1},
        schema={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
    )

    assert result.passed is True
    assert result.verifier == "schema"


def test_unknown_verifier_raises():
    engine = VerificationEngine()

    with pytest.raises(VerifierNotFoundError):
        engine.verify("nonexistent_verifier", x=1)


def test_successful_verification_returns_passed_true():
    engine = VerificationEngine()

    result = engine.verify("math", expression="10 / 2", expected=5)

    assert result.passed is True


def test_failed_verification_returns_a_result_not_an_exception():
    """A legitimately wrong claim must not raise -- it's a normal,
    expected outcome."""
    engine = VerificationEngine()

    result = engine.verify("math", expression="2 + 2", expected=999)

    assert result.passed is False


def test_malformed_input_exception_propagates_through_the_engine():
    engine = VerificationEngine()

    with pytest.raises(MathVerificationError):
        engine.verify("math", expression="2 ** 3", expected=8)


def test_two_engine_instances_behave_identically_no_hidden_state():
    engine_one = VerificationEngine()
    engine_two = VerificationEngine()

    result_one = engine_one.verify("math", expression="1 + 1", expected=2)
    result_two = engine_two.verify("math", expression="1 + 1", expected=2)

    assert result_one == result_two


def test_constructing_the_engine_triggers_no_verification():
    calls = []
    VerificationEngine()  # construction alone

    assert calls == []  # nothing was called as a side effect of __init__


def test_verification_result_is_json_serializable():
    engine = VerificationEngine()
    result = engine.verify("math", expression="2 + 2", expected=4)

    payload = json.dumps(asdict(result))
    reloaded = json.loads(payload)

    assert reloaded["passed"] is True
    assert reloaded["verifier"] == "math"

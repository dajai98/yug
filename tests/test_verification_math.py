from __future__ import annotations

import pytest

from yug.verification.errors import MathVerificationError
from yug.verification.math import verify_math


def test_correct_arithmetic_passes():
    result = verify_math("2 + 2", 4)

    assert result.passed is True


def test_incorrect_arithmetic_fails_with_reason():
    result = verify_math("2 + 2", 5)

    assert result.passed is False
    assert "4" in result.reason


@pytest.mark.parametrize("expression", ["2 +", "((2 + 3)", ""])
def test_malformed_expression_raises(expression):
    with pytest.raises(MathVerificationError):
        verify_math(expression, 1)


@pytest.mark.parametrize(
    "expression",
    [
        "2 ** 3",
        "2 % 3",
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "[].__class__",
        "(lambda: 1)()",
        "[1, 2, 3]",
    ],
)
def test_unsupported_or_malicious_expression_raises_never_evaluates(expression):
    """These must never reach os.system/open/import -- the AST
    allow-list rejects the node types outright, exactly like
    yug.tools.builtin.calculator's proven approach."""
    with pytest.raises(MathVerificationError):
        verify_math(expression, 1)


def test_division_by_zero_raises():
    with pytest.raises(MathVerificationError):
        verify_math("1 / 0", 1)


def test_non_numeric_expected_raises():
    with pytest.raises(MathVerificationError):
        verify_math("2 + 2", "four")  # type: ignore[arg-type]


def test_result_is_deterministic():
    assert verify_math("3 * 3", 9) == verify_math("3 * 3", 9)


def test_float_comparison_uses_tolerance():
    result = verify_math("1 / 3", 0.3333333333333333)

    assert result.passed is True


def test_result_includes_actual_and_expected_in_details():
    result = verify_math("10 / 4", 2.5)

    assert result.details["expression"] == "10 / 4"
    assert result.details["actual"] == 2.5
    assert result.details["expected"] == 2.5


def test_result_verifier_field_is_math():
    result = verify_math("1 + 1", 2)

    assert result.verifier == "math"

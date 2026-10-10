from __future__ import annotations

import pytest

from yug.tools.builtin.calculator import calculate
from yug.tools.errors import CalculatorError


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 2", 4),
        ("10 * 5", 50),
        ("100 / 4", 25),
        ("(2 + 3) * 4", 20),
        ("-5 + 3", -2),
        ("2 - -3", 5),
    ],
)
def test_arithmetic(expression, expected):
    assert calculate(expression) == expected


def test_division_by_zero_raises():
    with pytest.raises(CalculatorError):
        calculate("1 / 0")


@pytest.mark.parametrize("expression", ["2 +", "((2 + 3)", "2 ** 3", "2 % 3"])
def test_invalid_or_unsupported_expression_raises(expression):
    with pytest.raises(CalculatorError):
        calculate(expression)


def test_oversized_expression_raises():
    with pytest.raises(CalculatorError):
        calculate("1+" * 500 + "1")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "[].__class__",
        "(lambda: 1)()",
        "'a' + 'b'",
        "[1, 2, 3]",
    ],
)
def test_malicious_or_non_arithmetic_expressions_rejected(expression):
    """These must never reach os.system/open/import -- the AST
    allow-list rejects Call/Attribute/Name/string-literal/list nodes
    outright, not via a denylist of known-bad substrings."""
    with pytest.raises(CalculatorError):
        calculate(expression)

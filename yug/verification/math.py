"""MathVerifier: independently evaluate a simple arithmetic expression
and compare it against a claimed/expected value.

The evaluator below is an ADAPTED COPY of the safe AST-walk approach
in ``yug.tools.builtin.calculator`` (same allow-list technique: parse
with ``ast.parse(..., mode="eval")``, walk by hand, accept only
``Constant``/``BinOp(+-*/)``/``UnaryOp(+-)``, reject everything else --
never ``eval()``/``exec()``). It is duplicated here rather than
imported so ``yug.verification`` has no dependency on ``yug.tools``
(see this package's ``__init__`` docstring) -- the two modules will
drift only if the underlying safety technique itself changes, which is
an acceptable, explicit tradeoff for subsystem independence.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from yug.verification.errors import MathVerificationError
from yug.verification.types import VerificationResult

_MAX_EXPRESSION_LENGTH = 200

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise MathVerificationError(f"unsupported literal: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise MathVerificationError("division by zero")
        return _BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_eval_node(node.operand))

    raise MathVerificationError(f"unsupported expression element: {type(node).__name__}")


def _safe_eval(expression: str) -> float:
    if not isinstance(expression, str) or not expression.strip():
        raise MathVerificationError(f"expression must be a non-empty string, got {expression!r}")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise MathVerificationError(
            f"expression exceeds maximum length of {_MAX_EXPRESSION_LENGTH} characters"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise MathVerificationError(f"malformed expression: {exc}") from exc

    return _eval_node(tree.body)


def verify_math(expression: str, expected: Any) -> VerificationResult:
    """Independently evaluate ``expression`` and check it equals
    ``expected``.

    Raises:
        MathVerificationError: ``expression`` is empty, too long, not
            valid syntax, contains anything outside the safe +-*/
            allow-list, divides by zero, or ``expected`` is not a
            number. NOT raised when the expression evaluates fine but
            to a value different from ``expected`` -- that produces
            ``VerificationResult(passed=False)``.
    """
    if not isinstance(expected, (int, float)) or isinstance(expected, bool):
        raise MathVerificationError(f"expected must be a number, got {type(expected).__name__}")

    actual = _safe_eval(expression)
    passed = math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)

    reason = None
    if not passed:
        reason = f"expression {expression!r} evaluates to {actual!r}, expected {expected!r}"

    return VerificationResult(
        passed=passed,
        verifier="math",
        reason=reason,
        details={"expression": expression, "actual": actual, "expected": expected},
    )

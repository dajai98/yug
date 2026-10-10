"""Safe calculator: +, -, *, /, (), unary +/-, on int/float literals
only.

Never uses ``eval()`` or ``exec()``. The expression is parsed with
``ast.parse(..., mode="eval")`` and then walked by hand, accepting
only a small allow-listed set of AST node types (``Constant``,
``BinOp`` with +-*/, ``UnaryOp`` with unary +/-). Anything else --
``Name`` (variables/builtins like ``__import__``), ``Call``
(``open(...)``, ``os.system(...)``), ``Attribute``, ``Import``, string
literals, comprehensions, and every other AST node type -- is rejected
by the allow-list itself, not by a denylist of "known-bad" patterns.
This is a genuine safety property of the approach, not merely a
tested-so-far claim: there is no code path from a
:class:`ast.Constant`/``BinOp``/``UnaryOp``-only AST to arbitrary code
execution.
"""

from __future__ import annotations

import ast
import operator

from yug.tools.errors import CalculatorError

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
            raise CalculatorError(f"unsupported literal: {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise CalculatorError("division by zero")
        return _BINARY_OPERATORS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_eval_node(node.operand))

    raise CalculatorError(f"unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> float:
    """Evaluate ``expression`` and return the numeric result.

    Raises:
        CalculatorError: the expression is too long, is not valid
            Python syntax, contains anything other than numbers,
            parentheses, and +-*/, or divides by zero.
    """
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise CalculatorError(
            f"expression exceeds maximum length of {_MAX_EXPRESSION_LENGTH} characters"
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"invalid expression: {exc}") from exc

    return _eval_node(tree.body)

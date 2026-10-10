"""Phase 3's four core tools, each as a
:class:`~yug.tools.types.ToolDefinition` ready to register.

``file_read``/``file_search``'s ``root`` and ``max_bytes``/``max_results``
are bound at definition time (via a closure), not passed by the
caller, so the allowed-root restriction cannot be widened by a
malicious or buggy tool-call argument -- the caller can only ever
supply ``path``/``pattern`` relative to the root that was configured
when :func:`build_default_tools` was called.
"""

from __future__ import annotations

from yug.tools.builtin.calculator import calculate
from yug.tools.builtin.file_read import read_file
from yug.tools.builtin.file_search import search_files
from yug.tools.builtin.python_exec import execute_python
from yug.tools.types import ParameterSpec, ToolDefinition


def build_default_tools(
    file_root: str,
    file_read_max_bytes: int,
    file_search_max_results: int,
    file_search_max_file_bytes: int,
    python_exec_timeout_seconds: float,
) -> tuple[ToolDefinition, ...]:
    """Construct the four Phase 3 :class:`ToolDefinition`\\ s, with
    file-access tools bound to ``file_root`` and the configured
    limits."""

    calculator_tool = ToolDefinition(
        name="calculator",
        description="Evaluate a safe arithmetic expression (+, -, *, /, parentheses).",
        parameters=(ParameterSpec(name="expression", type="string", required=True),),
        handler=lambda expression: calculate(expression),
    )

    file_read_tool = ToolDefinition(
        name="file_read",
        description="Read a UTF-8 text file, restricted to a configured root directory.",
        parameters=(ParameterSpec(name="path", type="string", required=True),),
        handler=lambda path: read_file(path, root=file_root, max_bytes=file_read_max_bytes),
    )

    file_search_tool = ToolDefinition(
        name="file_search",
        description="Search for files by glob pattern, restricted to a configured root directory.",
        parameters=(ParameterSpec(name="pattern", type="string", required=True),),
        handler=lambda pattern: search_files(
            pattern,
            root=file_root,
            max_results=file_search_max_results,
            max_file_bytes=file_search_max_file_bytes,
        ),
    )

    python_exec_tool = ToolDefinition(
        name="python_exec",
        description="Execute Python code in a subprocess with a timeout (see sandbox module for isolation limits).",
        parameters=(ParameterSpec(name="code", type="string", required=True),),
        handler=lambda code: execute_python(code, timeout_seconds=python_exec_timeout_seconds),
    )

    return (calculator_tool, file_read_tool, file_search_tool, python_exec_tool)

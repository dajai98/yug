"""Tool subsystem exceptions.

Follows the established convention (``yug.brain.errors``,
``yug.memory.errors``, ``yug.tokenizer.errors``, ``yug.native.errors``):
a subsystem-rooted class under :class:`yug.utils.errors.YugError`, with
subclasses only for failure modes the implementation actually
distinguishes and exercises.
"""

from __future__ import annotations

from yug.utils.errors import YugError


class ToolError(YugError):
    """Base class for all yug.tools failures."""


class ToolNotFoundError(ToolError):
    """Raised when looking up or executing a tool name that is not
    registered."""


class ToolAlreadyRegisteredError(ToolError):
    """Raised by :meth:`~yug.tools.registry.ToolRegistry.register` when
    a tool with the same name is already registered. Duplicate
    registration is rejected outright rather than silently overwriting
    -- an accidental double-registration should be visible, not
    silently resolved by "last one wins"."""


class ToolValidationError(ToolError):
    """Raised when parameters given to a tool call do not satisfy its
    :class:`~yug.tools.types.ToolDefinition` schema -- missing required
    parameter, wrong type, or an unexpected parameter name."""


class PermissionDeniedError(ToolError):
    """Raised when a tool call is not permitted by the current
    permission configuration. Never reaches the tool implementation --
    the executor checks permission before calling it."""


class CalculatorError(ToolError):
    """Raised for an invalid, unsupported, or unsafe calculator
    expression (syntax error, disallowed construct, division by
    zero, oversized input)."""


class FileAccessError(ToolError):
    """Raised for a file_read/file_search request that is invalid or
    would escape the tool's allowed root: missing file, directory
    given where a file was expected, oversized file, or a path that
    resolves outside the configured root (``../`` traversal, an
    absolute path, or a symlink escape)."""


class PythonExecutionError(ToolError):
    """Raised when the python_exec sandbox itself cannot run the
    request -- e.g. it timed out. NOT raised for an exception inside
    the executed code itself; that is captured as normal
    stdout/stderr/exit_code output (see
    :mod:`yug.tools.builtin.python_exec`'s docstring for the
    reasoning)."""

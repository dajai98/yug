"""Tool subsystem: discovery, validation, permission-checking, and
execution of a small set of deterministic/sandboxed tools.

Per architectural invariant #4 ("tools are permission-controlled"),
every tool call passes through :class:`yug.tools.executor.ToolExecutor`,
which enforces registry lookup, parameter validation, and a permission
check, in that order, before a tool implementation ever runs. Tools
must not be invoked directly, bypassing the executor.

Phase 3 scope only: calculator, python_exec, file_read, file_search --
no agent runtime, no autonomous tool selection.
"""

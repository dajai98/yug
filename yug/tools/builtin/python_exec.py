"""python_exec tool: runs code through
:func:`yug.tools.sandbox.run_python_code` -- see that module's
docstring for the honest, explicit statement of what is and is not
isolated.

Design choice worth stating explicitly: an exception *raised by the
executed code itself* (e.g. ``1 / 0``) is NOT a python_exec tool
failure. It is normal output -- the code ran, to completion or to its
own failure point, inside the sandbox exactly as requested, and its
traceback is captured in ``stderr`` with a non-zero ``exit_code``,
same as running any Python script from a shell. Only the sandbox
itself failing to run the request at all (a timeout) is a
:class:`~yug.tools.errors.PythonExecutionError`.
"""

from __future__ import annotations

from yug.tools.errors import PythonExecutionError
from yug.tools.sandbox import run_python_code


def execute_python(code: str, timeout_seconds: float = 5.0) -> dict:
    """Execute ``code`` in a subprocess and return its outcome.

    Returns:
        A dict with keys ``stdout``, ``stderr``, ``exit_code``.

    Raises:
        PythonExecutionError: execution did not complete within
            ``timeout_seconds``.
    """
    result = run_python_code(code, timeout_seconds=timeout_seconds)
    if result.timed_out:
        raise PythonExecutionError(f"execution timed out after {timeout_seconds}s")

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
    }

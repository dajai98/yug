from __future__ import annotations

import pytest

from yug.tools.builtin.python_exec import execute_python
from yug.tools.errors import PythonExecutionError


def test_successful_execution_returns_stdout():
    result = execute_python("print('hello')")

    assert result["stdout"].strip() == "hello"
    assert result["exit_code"] == 0


def test_stderr_is_captured():
    result = execute_python("import sys; sys.stderr.write('oops')")

    assert "oops" in result["stderr"]


def test_exception_in_code_is_captured_not_raised():
    """An exception inside the executed code is normal output (a
    non-zero exit code and a traceback in stderr), not a
    PythonExecutionError -- see the module's docstring for why."""
    result = execute_python("1 / 0")

    assert result["exit_code"] != 0
    assert "ZeroDivisionError" in result["stderr"]


def test_timeout_raises_python_execution_error():
    with pytest.raises(PythonExecutionError):
        execute_python("import time; time.sleep(10)", timeout_seconds=0.5)


def test_environment_is_minimal_not_fully_inherited(monkeypatch):
    """Isolation-boundary behavior: a variable set in this test
    process's own environment must NOT leak into the executed code --
    env is explicitly minimal, not inherited wholesale."""
    monkeypatch.setenv("YUG_TEST_CANARY_VAR", "should-not-leak")

    result = execute_python(
        "import os; print(os.environ.get('YUG_TEST_CANARY_VAR', 'NOTSET'))"
    )

    assert "NOTSET" in result["stdout"]

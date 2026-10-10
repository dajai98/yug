"""Tests for the CLI entry point.

Covers the Phase 0 exit criterion: ``python -m yug --version`` works.
"""

from __future__ import annotations

import subprocess
import sys

from yug import __milestone__, __version__
from yug.__main__ import main


def test_version_flag_returns_zero_and_prints_version(capsys):
    exit_code = main(["--version"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out
    assert __milestone__ in captured.out


def test_no_args_prints_help_and_returns_zero(capsys):
    exit_code = main([])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()


def test_module_invocation_via_subprocess():
    """Exercises the actual exit-criterion command as a subprocess,
    not just the in-process `main()` function, to prove `python -m yug
    --version` genuinely works end to end."""
    result = subprocess.run(
        [sys.executable, "-m", "yug", "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        # Explicit, not inherited: on Windows, leaving stdin=None makes
        # Popen try to duplicate the *parent's* stdin handle for the
        # child. Under pytest's capture harness that parent handle can
        # be invalid, causing OSError: [WinError 6] before the child
        # even starts. The child never reads stdin, so DEVNULL is both
        # correct and avoids the whole handle-inheritance path.
        stdin=subprocess.DEVNULL,
    )

    assert result.returncode == 0
    assert __version__ in result.stdout

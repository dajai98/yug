"""Subprocess-based execution boundary for python_exec.

HONEST SECURITY STATEMENT, per Directive Section 9/18 -- read before
relying on this for anything sensitive:

Docker is NOT present or configured anywhere in this repository or
environment (checked before writing this module: no ``Dockerfile``, no
``docker`` invocation anywhere, and the sandbox this code itself runs
in during development has no ``docker`` binary and no daemon socket
available). Per the directive's explicit instruction not to invent a
fake sandbox or claim false security, this module implements a real,
but limited, boundary:

    VERIFIED:
        - a separate OS process (not the YUG process's own interpreter
          / memory space)
        - a wall-clock execution timeout (subprocess.run(timeout=...))
        - `python -I` isolated mode (ignores PYTHONPATH, user
          site-packages, and PYTHONSTARTUP -- reduces environment-based
          code injection)
        - a minimal, explicit environment (not the full inherited
          parent environment)
        - a restricted working directory
        - CPU-time and address-space resource limits via
          ``resource.setrlimit``, WHERE AVAILABLE (POSIX only -- see
          below)
        - no inherited stdin: the child's stdin is explicitly
          ``subprocess.DEVNULL``, not the parent's own stdin handle.
          This was originally added as a Windows compatibility fix
          (inheriting an invalid/non-duplicable parent stdin handle
          under certain launch contexts, e.g. pytest's output capture,
          caused ``OSError: [WinError 6] The handle is invalid`` inside
          CPython's ``Popen._make_inheritable``), but it is also a
          genuine, deliberate safety property independent of that bug:
          executed code can never receive unexpected interactive input
          from whatever happens to be attached to YUG's own stdin.

    NOT VERIFIED / NOT PROVIDED:
        - filesystem isolation: the executed code can still read/write
          anywhere the OS user running YUG can, outside the working
          directory restriction (there is no chroot, no container
          filesystem, no bind-mount restriction)
        - network isolation: the executed code can make network
          connections; nothing here blocks that
        - resource limits are POSIX-only (``resource`` module is not
          importable on Windows -- this is detected and the limits are
          simply skipped there, not silently faked)
        - no seccomp/syscall filtering of any kind

This is a development/test-appropriate process boundary with a
timeout, not a security sandbox suitable for untrusted code. Anyone
integrating this into a context where the executed code could be
adversarial (rather than the local user's own code) must add real
isolation (a container, a VM, or a restricted-syscall sandbox) before
relying on it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import resource  # POSIX only

    _RESOURCE_MODULE_AVAILABLE = True
except ImportError:  # pragma: no cover -- exercised only on Windows
    _RESOURCE_MODULE_AVAILABLE = False

# Resource ceilings applied on POSIX only. Deliberately generous (this
# is a timeout/isolation boundary, not a resource-starvation defense)
# -- just enough to stop a runaway allocation loop from consuming the
# whole host.
_MAX_CPU_SECONDS = 30
_MAX_ADDRESS_SPACE_BYTES = 512 * 1024 * 1024  # 512 MB


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def _apply_resource_limits() -> None:  # pragma: no cover -- POSIX-only, exercised only there
    resource.setrlimit(resource.RLIMIT_CPU, (_MAX_CPU_SECONDS, _MAX_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (_MAX_ADDRESS_SPACE_BYTES, _MAX_ADDRESS_SPACE_BYTES))


def run_python_code(code: str, timeout_seconds: float) -> SandboxResult:
    """Run ``code`` as a standalone Python subprocess and capture its
    output. Never raises for a failure *inside* the executed code
    (a non-zero exit code and/or stderr traceback are normal,
    successfully-captured output) -- see
    :mod:`yug.tools.builtin.python_exec`'s docstring for why.

    Only ``subprocess`` itself failing to start, or timing out, is
    reported via ``timed_out`` / a non-``0`` process-start failure.
    """
    with tempfile.TemporaryDirectory(prefix="yug_python_exec_") as work_dir:
        minimal_env = {"PATH": os.environ.get("PATH", "")}
        preexec_fn = _apply_resource_limits if (_RESOURCE_MODULE_AVAILABLE and os.name == "posix") else None

        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", code],
                cwd=work_dir,
                env=minimal_env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                preexec_fn=preexec_fn,  # noqa: S603 -- deliberate, documented boundary
            )
            return SandboxResult(
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_code=completed.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                stdout=(exc.stdout or ""),
                stderr=(exc.stderr or ""),
                exit_code=-1,
                timed_out=True,
            )

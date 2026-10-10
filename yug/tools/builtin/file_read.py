"""Read-only file access, restricted to an explicitly configured root.

Security note on the classic pathlib pitfall this guards against:
``Path("/allowed/root") / "/etc/passwd"`` evaluates to
``Path("/etc/passwd")`` -- joining with the ``/`` operator DISCARDS the
base when the right-hand side is itself absolute. This module rejects
an absolute ``path`` argument explicitly, before any joining happens,
rather than relying on the post-join containment check alone to catch
it (the containment check is still applied too, as defense in depth
against symlink escapes).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from yug.tools.errors import FileAccessError


def read_file(path: str, root: str, max_bytes: int) -> str:
    """Read a UTF-8 text file at ``path``, relative to ``root``.

    Raises:
        FileAccessError: ``path`` is absolute, contains a ``..``
            segment, resolves outside ``root`` (including via a
            symlink), does not exist, is a directory, exceeds
            ``max_bytes``, or is not valid UTF-8.
    """
    root_resolved = Path(root).resolve()

    if PurePosixPath(path).is_absolute() or Path(path).is_absolute():
        raise FileAccessError(f"path must be relative to the allowed root, got absolute path: {path!r}")
    if ".." in Path(path).parts:
        raise FileAccessError(f"path must not contain '..' segments: {path!r}")

    candidate = (root_resolved / path).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise FileAccessError(f"path resolves outside the allowed root: {path!r}")

    if not candidate.exists():
        raise FileAccessError(f"file not found: {path!r}")
    if candidate.is_dir():
        raise FileAccessError(f"path is a directory, not a file: {path!r}")

    size = candidate.stat().st_size
    if size > max_bytes:
        raise FileAccessError(f"file size {size} exceeds max_bytes={max_bytes}: {path!r}")

    try:
        return candidate.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise FileAccessError(f"file is not valid UTF-8 text: {path!r}") from exc

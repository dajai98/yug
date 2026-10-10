"""Read-only recursive file search, restricted to an explicitly
configured root. Deterministic ordering (sorted paths); bounded by a
maximum result count and a per-file size limit; unreadable files are
skipped rather than aborting the whole search."""

from __future__ import annotations

from pathlib import Path

from yug.tools.errors import FileAccessError


def search_files(pattern: str, root: str, max_results: int, max_file_bytes: int) -> list[str]:
    """Search for files matching ``pattern`` (a ``pathlib``-style glob,
    e.g. ``"*.py"`` or ``"**/*.txt"``) under ``root``.

    Returns:
        A sorted list of paths, relative to ``root``, of matching
        *files* (directories are never included) that are no larger
        than ``max_file_bytes``, capped at ``max_results`` entries.
        Files that raise an OS error when stat'd (permission denied,
        broken symlink, etc.) are silently skipped, not treated as a
        search failure.

    Raises:
        FileAccessError: ``pattern`` contains a ``..`` segment (an
            explicit traversal attempt), or ``root`` does not exist /
            is not a directory.
    """
    if ".." in Path(pattern).parts:
        raise FileAccessError(f"pattern must not contain '..' segments: {pattern!r}")

    root_resolved = Path(root).resolve()
    if not root_resolved.is_dir():
        raise FileAccessError(f"search root does not exist or is not a directory: {root!r}")

    matches: list[str] = []
    for candidate in sorted(root_resolved.rglob(pattern)):
        if len(matches) >= max_results:
            break
        if not candidate.is_file():
            continue
        try:
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root_resolved):
                continue  # symlink escaping the root -- skip, not an error
            if resolved.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue  # unreadable/broken entry -- skip, not a search failure
        matches.append(str(resolved.relative_to(root_resolved)))

    return matches

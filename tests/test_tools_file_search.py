from __future__ import annotations

import pytest

from yug.tools.builtin.file_search import search_files
from yug.tools.errors import FileAccessError


@pytest.fixture
def root(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "c.py").write_text("c")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested")
    (tmp_path / "big.txt").write_text("x" * 1000)
    return tmp_path


def test_finds_matching_files(root):
    matches = search_files("*.txt", root=str(root), max_results=100, max_file_bytes=1_000_000)

    assert "a.txt" in matches
    assert "b.txt" in matches
    assert "c.py" not in matches


def test_no_matches_returns_empty_list(root):
    matches = search_files("*.nonexistent", root=str(root), max_results=100, max_file_bytes=1_000_000)

    assert matches == []


def test_recursive_pattern_finds_nested_files(root):
    matches = search_files("**/*.txt", root=str(root), max_results=100, max_file_bytes=1_000_000)

    assert any(m.endswith("nested.txt") for m in matches)


def test_result_ordering_is_deterministic(root):
    first = search_files("*.txt", root=str(root), max_results=100, max_file_bytes=1_000_000)
    second = search_files("*.txt", root=str(root), max_results=100, max_file_bytes=1_000_000)

    assert first == second == sorted(first)


def test_max_results_is_enforced(root):
    matches = search_files("*.txt", root=str(root), max_results=1, max_file_bytes=1_000_000)

    assert len(matches) == 1


def test_max_file_bytes_excludes_oversized_files(root):
    matches = search_files("big.txt", root=str(root), max_results=100, max_file_bytes=10)

    assert matches == []


def test_dotdot_in_pattern_raises(root):
    with pytest.raises(FileAccessError):
        search_files("../*.txt", root=str(root), max_results=100, max_file_bytes=1_000_000)


def test_nonexistent_root_raises(tmp_path):
    with pytest.raises(FileAccessError):
        search_files("*.txt", root=str(tmp_path / "does_not_exist"), max_results=100, max_file_bytes=1_000_000)

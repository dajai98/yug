from __future__ import annotations

import pytest

from yug.tools.builtin.file_read import read_file
from yug.tools.errors import FileAccessError


@pytest.fixture
def root(tmp_path):
    (tmp_path / "hello.txt").write_text("hello world")
    (tmp_path / "empty.txt").write_text("")
    (tmp_path / "big.txt").write_text("x" * 1000)
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested.txt").write_text("nested")
    return tmp_path


def test_valid_file_read(root):
    assert read_file("hello.txt", root=str(root), max_bytes=1_000_000) == "hello world"


def test_nested_file_read(root):
    assert read_file("subdir/nested.txt", root=str(root), max_bytes=1_000_000) == "nested"


def test_missing_file_raises(root):
    with pytest.raises(FileAccessError):
        read_file("nope.txt", root=str(root), max_bytes=1_000_000)


def test_directory_instead_of_file_raises(root):
    with pytest.raises(FileAccessError):
        read_file("subdir", root=str(root), max_bytes=1_000_000)


def test_empty_file_reads_as_empty_string(root):
    assert read_file("empty.txt", root=str(root), max_bytes=1_000_000) == ""


def test_oversized_file_raises(root):
    with pytest.raises(FileAccessError):
        read_file("big.txt", root=str(root), max_bytes=10)


def test_dotdot_traversal_raises(root):
    with pytest.raises(FileAccessError):
        read_file("../outside.txt", root=str(root), max_bytes=1_000_000)


def test_absolute_path_outside_root_raises(root, tmp_path_factory):
    outside_root = tmp_path_factory.mktemp("outside")
    secret = outside_root / "secret.txt"
    secret.write_text("SECRET")

    with pytest.raises(FileAccessError):
        read_file(str(secret), root=str(root), max_bytes=1_000_000)


def test_symlink_escaping_root_raises(root, tmp_path_factory):
    outside_root = tmp_path_factory.mktemp("outside")
    secret = outside_root / "secret.txt"
    secret.write_text("SECRET")
    link = root / "escape_link"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("symlink creation not permitted in this environment")

    with pytest.raises(FileAccessError):
        read_file("escape_link", root=str(root), max_bytes=1_000_000)

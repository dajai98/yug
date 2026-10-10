from __future__ import annotations

import pytest

from yug.utils.types import VersionInfo


def test_version_info_str_format():
    info = VersionInfo(version="0.1.0", milestone="M1-Phase0")
    assert str(info) == "yug 0.1.0 (M1-Phase0)"


def test_version_info_is_frozen():
    info = VersionInfo(version="0.1.0", milestone="M1-Phase0")
    with pytest.raises(AttributeError):
        info.version = "9.9.9"

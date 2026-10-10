from __future__ import annotations

import json
import logging

from yug.utils.logging import configure_logging, get_logger


def test_get_logger_returns_namespaced_logger():
    logger = get_logger("foo")
    assert logger.name == "yug.foo"


def test_log_output_is_valid_json_with_expected_fields(capsys):
    configure_logging(level="INFO", stream=True)
    logger = get_logger("test_logging")

    logger.info("hello world", extra={"task_id": "abc123"})

    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["level"] == "INFO"
    assert record["logger"] == "yug.test_logging"
    assert record["message"] == "hello world"
    assert record["extra"]["task_id"] == "abc123"
    assert "timestamp" in record


def test_configure_logging_replaces_handlers_not_stacks_them():
    configure_logging(level="INFO", stream=True)
    configure_logging(level="INFO", stream=True)

    root = logging.getLogger("yug")
    assert len(root.handlers) == 1


def test_log_file_is_created_when_log_dir_given(tmp_path):
    log_dir = tmp_path / "logs"
    configure_logging(level="INFO", log_dir=log_dir, stream=False)
    logger = get_logger("test_file_logging")

    logger.warning("disk write check")

    log_file = log_dir / "yug.log"
    assert log_file.exists()
    content = log_file.read_text().strip()
    record = json.loads(content.splitlines()[-1])
    assert record["message"] == "disk write check"

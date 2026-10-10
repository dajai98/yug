from __future__ import annotations

from yug.utils.errors import ConfigurationError, ConfigValidationError, YugError


def test_configuration_error_is_a_yug_error():
    assert issubclass(ConfigurationError, YugError)


def test_config_validation_error_is_a_configuration_error():
    assert issubclass(ConfigValidationError, ConfigurationError)
    assert issubclass(ConfigValidationError, YugError)


def test_yug_error_is_a_real_exception_and_catchable():
    try:
        raise ConfigValidationError("bad value")
    except YugError as exc:
        assert "bad value" in str(exc)
    else:
        raise AssertionError("expected YugError to be raised and caught")

"""YUG exception hierarchy.

All exceptions raised by YUG subsystems must inherit from :class:`YugError`
so that calling code (and eventually the agent runtime's error-handling
transitions) can catch YUG-specific failures distinctly from unrelated
Python exceptions.

Only the exception types actually used by Phase 0 code (configuration
loading/validation) are defined here. Other subsystems' branches
(BrainError, TokenizerError, NativeError, MemorySubsystemError,
ToolError, AgentError, VerificationError, SecurityError, ...) live in
their own subsystem's ``errors.py`` module and are added when that
subsystem is implemented -- adding them here speculatively, before
there is code to exercise them, would violate the same rule.
"""

from __future__ import annotations


class YugError(Exception):
    """Base class for all YUG-specific exceptions."""


class ConfigurationError(YugError):
    """Raised when configuration cannot be loaded or is structurally invalid.

    Examples: the config file is missing when explicitly requested, the
    file is not valid YAML, or a required key is absent.
    """


class ConfigValidationError(ConfigurationError):
    """Raised when configuration values fail validation rules.

    Distinct from :class:`ConfigurationError` so callers can tell apart
    "could not read/parse configuration" from "configuration was read but
    the values are invalid" (e.g. wrong type, out-of-range value).
    """

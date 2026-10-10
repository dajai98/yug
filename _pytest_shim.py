"""Minimal stand-in for the parts of the pytest API this test suite uses.
Not real pytest. Built only because pytest cannot be installed in this
network-disabled sandbox. See run_tests_shim.py for full disclosure."""
import contextlib
import importlib


class Skipped(Exception):
    def __init__(self, reason=""):
        super().__init__(reason)
        self.reason = reason


def skip(reason=""):
    raise Skipped(reason)


def importorskip(modname):
    try:
        return importlib.import_module(modname)
    except ImportError:
        raise Skipped(f"could not import {modname!r}")


class _ExcInfo:
    def __init__(self):
        self.value = None


@contextlib.contextmanager
def raises(exc_types, match=None):
    holder = _ExcInfo()
    try:
        yield holder
    except exc_types as e:
        holder.value = e
        if match is not None:
            import re
            assert re.search(match, str(e)), f"{match!r} not found in {e!r}"
        return
    else:
        raise AssertionError(f"Expected {exc_types} to be raised, nothing was raised")


class _Mark:
    def parametrize(self, argnames, argvalues, ids=None):
        def deco(fn):
            existing = getattr(fn, "_shim_parametrize", [])
            existing = existing + [(argnames, argvalues)]
            fn._shim_parametrize = existing
            return fn
        return deco

    def skipif(self, condition, reason=""):
        def deco(fn):
            if condition:
                fn._shim_skip_reason = reason
            return fn
        return deco

    def skip(self, reason=""):
        def deco(fn):
            fn._shim_skip_reason = reason
            return fn
        return deco

    def __getattr__(self, item):
        # e.g. pytest.mark.xfail -- not used in this suite,
        # but degrade gracefully rather than hard-crash collection.
        def deco(*a, **kw):
            def wrap(fn):
                return fn
            return wrap
        return deco


mark = _Mark()


def fixture(*args, **kwargs):
    # Support both @pytest.fixture and @pytest.fixture(params=...)/(...)
    if len(args) == 1 and callable(args[0]) and not kwargs:
        fn = args[0]
        fn._is_shim_fixture = True
        return fn

    def deco(fn):
        fn._is_shim_fixture = True
        return fn
    return deco

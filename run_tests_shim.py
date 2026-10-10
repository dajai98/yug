"""
Minimal pytest-compatible test runner. NOT real pytest.

Why this exists: `pip install pytest` fails in this sandbox with
"No matching distribution found for pytest" (no network access -- verified
directly, not assumed). Real pytest and real PyTorch are both unavailable.
This script actually imports and executes the real, unmodified test files
in tests/, resolving a small set of fixtures (tmp_path, tmp_path_factory,
capsys, monkeypatch, and any locally defined @pytest.fixture), expanding
@pytest.mark.parametrize, and supporting pytest.raises/skip/importorskip.

Disclosed limitations:
  - Only module-level `def test_*(...)` functions are collected (matches
    this suite -- no test classes are used here, confirmed by grep).
  - Fixture resolution is a simple one-level dependency lookup, not the
    full pytest DI graph. Sufficient for this suite's fixtures (checked
    by inspection before writing this).
  - A test file that fails to import is reported as a COLLECTION ERROR
    for every test in that file, not silently dropped.
"""
import importlib
import importlib.util
import inspect
import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _pytest_shim as pytest_shim  # noqa: E402
sys.modules["pytest"] = pytest_shim

TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)


class FakeCapsys:
    def __init__(self):
        self._out = []
        self._err = []

    def readouterr(self):
        import collections
        Result = collections.namedtuple("Result", ["out", "err"])
        out = "".join(self._out)
        err = "".join(self._err)
        self._out.clear()
        self._err.clear()
        return Result(out=out, err=err)


class CapturingStream:
    def __init__(self, sink_list, real):
        self.sink_list = sink_list
        self.real = real

    def write(self, s):
        self.sink_list.append(s)
        return len(s)

    def flush(self):
        pass


class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setenv(self, name, value):
        old = os.environ.get(name, None)
        had = name in os.environ
        os.environ[name] = value
        self._undo.append(lambda: (os.environ.__setitem__(name, old) if had else os.environ.pop(name, None)))

    def delenv(self, name, raising=True):
        had = name in os.environ
        old = os.environ.get(name)
        if had:
            del os.environ[name]
        elif raising:
            raise KeyError(name)
        self._undo.append(lambda: os.environ.__setitem__(name, old) if had else None)

    def chdir(self, path):
        old = os.getcwd()
        os.chdir(str(path))
        self._undo.append(lambda: os.chdir(old))

    def setattr(self, target, name, value):
        old = getattr(target, name)
        setattr(target, name, value)
        self._undo.append(lambda: setattr(target, name, old))

    def undo(self):
        for fn in reversed(self._undo):
            try:
                fn()
            except Exception:
                pass
        self._undo.clear()


class TmpPathFactory:
    def __init__(self, base):
        self.base = base
        self._n = 0

    def mktemp(self, name):
        self._n += 1
        import pathlib
        p = pathlib.Path(self.base) / f"{name}{self._n}"
        p.mkdir(parents=True, exist_ok=True)
        return p


def make_fixture_values(base_tmp_root):
    import pathlib
    tmp_path = pathlib.Path(tempfile.mkdtemp(dir=base_tmp_root))
    tmp_path_factory = TmpPathFactory(tempfile.mkdtemp(dir=base_tmp_root))
    capsys = FakeCapsys()
    mp = MonkeyPatch()
    return {
        "tmp_path": tmp_path,
        "tmp_path_factory": tmp_path_factory,
        "capsys": capsys,
        "monkeypatch": mp,
    }, mp, capsys


def resolve_local_fixture(module, name, cache, ctx_values):
    if name in cache:
        return cache[name]
    fn = getattr(module, name, None)
    if fn is None or not getattr(fn, "_is_shim_fixture", False):
        raise LookupError(name)
    sig = inspect.signature(fn)
    kwargs = {}
    for p in sig.parameters:
        kwargs[p] = get_fixture_value(module, p, cache, ctx_values)
    val = fn(**kwargs)
    if inspect.isgenerator(val):
        first = next(val)
        cache[name] = first
        cache.setdefault("__gens__", []).append(val)
        return first
    cache[name] = val
    return val


def get_fixture_value(module, name, cache, ctx_values):
    if name in ctx_values:
        return ctx_values[name]
    return resolve_local_fixture(module, name, cache, ctx_values)


def run_module(path, base_tmp_root, results):
    modname = os.path.splitext(os.path.basename(path))[0]
    try:
        spec = importlib.util.spec_from_file_location(modname, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[modname] = module
        spec.loader.exec_module(module)
    except Exception as e:
        results["collect_errors"].append((modname, "".join(traceback.format_exception(type(e), e, e.__traceback__))))
        return

    test_fns = [
        (n, getattr(module, n)) for n in dir(module)
        if n.startswith("test_") and inspect.isfunction(getattr(module, n))
        and getattr(module, n).__module__ == modname
    ]

    for name, fn in test_fns:
        param_sets = getattr(fn, "_shim_parametrize", None)
        if param_sets:
            # only supports a single parametrize decorator (sufficient here; checked)
            argnames, argvalues = param_sets[0]
            names = [a.strip() for a in argnames.split(",")] if isinstance(argnames, str) else list(argnames)
            for raw in argvalues:
                if len(names) == 1:
                    values = (raw,)
                elif not isinstance(raw, (tuple, list)):
                    values = (raw,)
                else:
                    values = raw
                pid = ",".join(repr(v) for v in values)
                run_single(module, modname, name, fn, base_tmp_root, results,
                           extra_params=dict(zip(names, values)), pid=pid)
        else:
            run_single(module, modname, name, fn, base_tmp_root, results)


def run_single(module, modname, name, fn, base_tmp_root, results, extra_params=None, pid=None):
    full_name = f"{modname}.py::{name}" + (f"[{pid}]" if pid else "")
    skip_reason = getattr(fn, "_shim_skip_reason", None)
    if skip_reason is not None:
        results["skipped"].append((full_name, skip_reason))
        return
    ctx_values, mp, capsys = make_fixture_values(base_tmp_root)
    cache = {}
    sig = inspect.signature(fn)
    kwargs = {}
    try:
        for p in sig.parameters:
            if extra_params and p in extra_params:
                kwargs[p] = extra_params[p]
                continue
            kwargs[p] = get_fixture_value(module, p, cache, ctx_values)
    except LookupError as e:
        results["errors"].append((full_name, f"fixture not found: {e}"))
        return

    real_out, real_err = sys.stdout, sys.stderr
    if "capsys" in kwargs:
        sys.stdout = CapturingStream(capsys._out, real_out)
        sys.stderr = CapturingStream(capsys._err, real_err)
    try:
        fn(**kwargs)
        results["passed"].append(full_name)
    except pytest_shim.Skipped as s:
        results["skipped"].append((full_name, s.reason))
    except Exception as e:
        results["failed"].append((full_name, "".join(traceback.format_exception(type(e), e, e.__traceback__))))
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        mp.undo()
        for gen in cache.get("__gens__", []):
            try:
                next(gen)
            except StopIteration:
                pass
        try:
            shutil.rmtree(ctx_values["tmp_path"], ignore_errors=True)
            shutil.rmtree(ctx_values["tmp_path_factory"].base, ignore_errors=True)
        except Exception:
            pass


def main():
    base_tmp_root = tempfile.mkdtemp(prefix="yug_shim_")
    results = {"passed": [], "failed": [], "skipped": [], "errors": [], "collect_errors": []}
    files = sorted(f for f in os.listdir(TESTS_DIR) if f.startswith("test_") and f.endswith(".py"))
    for f in files:
        run_module(os.path.join(TESTS_DIR, f), base_tmp_root, results)

    print("=" * 70)
    for modname, tb in results["collect_errors"]:
        print(f"COLLECTION ERROR: {modname}")
        print(tb)
        print("-" * 70)
    for full_name, tb in results["failed"]:
        print(f"FAILED: {full_name}")
        print(tb)
        print("-" * 70)
    for full_name, reason in results["skipped"]:
        print(f"SKIPPED: {full_name} ({reason})")
    for full_name, msg in results["errors"]:
        print(f"ERROR: {full_name}: {msg}")

    print("=" * 70)
    print(f"passed={len(results['passed'])} failed={len(results['failed'])} "
          f"skipped={len(results['skipped'])} errors={len(results['errors'])} "
          f"collection_errors={len(results['collect_errors'])}")
    shutil.rmtree(base_tmp_root, ignore_errors=True)
    return 0 if not results["failed"] and not results["errors"] and not results["collect_errors"] else 1


if __name__ == "__main__":
    sys.exit(main())

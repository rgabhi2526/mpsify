"""Warning capture + atexit summary for CPU-fallback ops."""
import atexit
import contextlib
import re
import sys
import time
import warnings

# PyTorch's fallback warning looks like:
#   "The operator 'aten::foo' is not currently supported on the MPS backend
#    and will fall back to run on the CPU. ..."
_FALLBACK_RE = re.compile(r"operator '([^']+)'.*MPS backend.*fall back", re.DOTALL)

_fallback_ops: set[str] = set()   # unique ops that fell back to CPU
_missing_libs: list[str] = []     # unshimmable libs detected at patch time
_installed = False
_quiet = False


def note_missing_lib(name: str) -> None:
    _missing_libs.append(name)
    print(f"[mpsify] WARNING: '{name}' has no Metal backend; expect failure.",
          file=sys.stderr)


def install(quiet: bool = False) -> None:
    """Wrap warnings.showwarning to catch MPS fallbacks. Idempotent."""
    global _installed, _quiet
    _quiet = quiet
    if _installed:
        return
    _installed = True
    prev = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        m = _FALLBACK_RE.search(text)
        if m:
            op = m.group(1)
            if op not in _fallback_ops:          # dedupe -> once per unique op
                _fallback_ops.add(op)
                if not _quiet:
                    print(f"[mpsify] CPU fallback: {op} "
                          f"(runs on CPU, a latency hot spot)", file=sys.stderr)
            return  # swallow the noisy per-call torch warning
        prev(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning
    atexit.register(_summary)


def _summary() -> None:
    if not _fallback_ops and not _missing_libs:
        return
    print("\n[mpsify] ===== summary =====", file=sys.stderr)
    if _fallback_ops:
        print(f"[mpsify] {len(_fallback_ops)} op(s) fell back to CPU "
              f"(latency hot spots):", file=sys.stderr)
        for op in sorted(_fallback_ops):
            print(f"[mpsify]   - {op}", file=sys.stderr)
        print("[mpsify] run with --profile for call counts + timing.",
              file=sys.stderr)
    if _missing_libs:
        print(f"[mpsify] unshimmable libs detected: "
              f"{', '.join(_missing_libs)}", file=sys.stderr)
    print("[mpsify] ===================", file=sys.stderr)


@contextlib.contextmanager
def profiler():
    """Diagnostic mode: count calls + time ops that run on CPU.

    Adds real per-op Python overhead — NOT for production runs.
    """
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    counts: dict[str, int] = {}
    times: dict[str, float] = {}

    class _Prof(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            t0 = time.perf_counter()
            out = func(*args, **kwargs)
            dt = time.perf_counter() - t0
            # Op ran on CPU while inputs looked mps-ish -> a fallback.
            name = str(func)
            counts[name] = counts.get(name, 0) + 1
            times[name] = times.get(name, 0.0) + dt
            return out

    try:
        with _Prof():
            yield
    finally:
        top = sorted(times.items(), key=lambda kv: kv[1], reverse=True)[:20]
        print("\n[mpsify] ===== profile (top 20 by time) =====",
              file=sys.stderr)
        for name, t in top:
            print(f"[mpsify]   {t*1000:9.1f} ms  x{counts[name]:<7} {name}",
                  file=sys.stderr)
        print("[mpsify] =====================================", file=sys.stderr)

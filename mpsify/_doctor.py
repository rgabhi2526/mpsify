"""`mpsify doctor <script>` — scan a script for CUDA usage without running it."""
import re
import sys

from . import _UNSHIMMABLE

# (label, regex) — what mpsify will transparently remap at runtime.
_REMAP = [
    (".cuda() calls", re.compile(r"\.cuda\s*\(")),
    ("'cuda' device strings", re.compile(r"""['"]cuda(:\d+)?['"]""")),
    ("torch.cuda.* calls", re.compile(r"torch\.cuda\.")),
    ("map_location", re.compile(r"map_location")),
    ("AMP (autocast/GradScaler)", re.compile(r"amp\.(autocast|GradScaler)|autocast\s*\(")),
    ("DataParallel", re.compile(r"DataParallel")),
    ("nccl backend", re.compile(r"['\"]nccl['\"]")),
    ("pin_memory", re.compile(r"pin_memory")),
]


def doctor(path: str) -> int:
    try:
        src = open(path, encoding="utf-8").read()
    except OSError as e:
        print(f"[mpsify] cannot read {path}: {e}", file=sys.stderr)
        return 2

    print(f"[mpsify] doctor: {path}\n")
    hits = [(label, len(rx.findall(src))) for label, rx in _REMAP]
    hits = [(l, n) for l, n in hits if n]
    if hits:
        print("  Will remap to MPS at runtime:")
        for label, n in hits:
            print(f"    ✓ {label}  ({n})")
    else:
        print("  No CUDA-specific calls found — nothing to remap.")

    # unshimmable lib imports
    bad = [lib for lib in _UNSHIMMABLE
           if re.search(rf"\b(import|from)\s+{re.escape(lib)}\b", src)]
    if bad:
        print("\n  ⚠ Cannot fix (no Metal backend):")
        for lib in bad:
            print(f"    ✗ {lib}")
        print("\n  Verdict: will run, but the libraries above are likely to fail.")
        return 1
    print("\n  Verdict: should run under `python -m mpsify`.")
    return 0

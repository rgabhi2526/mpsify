"""mpsify — point it at a CUDA-hardcoded PyTorch repo, run it on Apple MPS.

Usage:
    python -m mpsify train.py --args      # wrapper, zero source edits
    import mpsify                          # fallback: patch on import

Knobs (env vars):
    PYTORCH_ENABLE_MPS_FALLBACK=1   set automatically by the wrapper
    MPSIFY_AMP=1                    re-enable AMP/autocast (default off = fp32)
    MPSIFY_QUIET=1                  suppress live warnings, keep exit summary
"""
import importlib.util
import os
import sys
import warnings

from . import _report

_patched = False

_UNSHIMMABLE = ["bitsandbytes", "apex", "deepspeed", "flash_attn", "triton"]


def _remap_device(dev):
    """Return dev with any cuda reference swapped to mps; else unchanged."""
    if isinstance(dev, str) and dev.startswith("cuda"):
        return "mps"
    # torch.device object
    if type(dev).__name__ == "device" and getattr(dev, "type", None) == "cuda":
        return "mps"
    return dev


def patch() -> None:
    """Monkeypatch the live torch module so cuda code runs on mps. Idempotent."""
    global _patched
    if _patched:
        return

    quiet = os.environ.get("MPSIFY_QUIET") == "1"
    _report.install(quiet=quiet)

    # Fallback env var must be set before torch initializes to take effect.
    torch_already = "torch" in sys.modules
    if torch_already and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        warnings.warn("[mpsify] torch imported before mpsify; "
                      "CPU fallback may not engage. Prefer `python -m mpsify`.")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch

    if not torch.backends.mps.is_available():
        warnings.warn("[mpsify] MPS not available on this machine; "
                      "cuda calls will still be remapped but there is no GPU.")

    # --- torch.cuda.* stubs -------------------------------------------------
    mps = torch.mps
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.set_device = lambda *a, **k: None
    torch.cuda.synchronize = lambda *a, **k: mps.synchronize()
    torch.cuda.empty_cache = lambda: mps.empty_cache()
    torch.cuda.get_device_name = lambda *a, **k: "mps"
    torch.cuda.manual_seed = lambda seed: mps.manual_seed(seed)
    torch.cuda.manual_seed_all = lambda seed: mps.manual_seed(seed)

    # --- .cuda() -> .to('mps') ---------------------------------------------
    def _tensor_cuda(self, *a, **k):
        return self.to("mps")

    def _module_cuda(self, *a, **k):
        return self.to("mps")

    torch.Tensor.cuda = _tensor_cuda
    torch.nn.Module.cuda = _module_cuda

    # --- torch.device('cuda') -> mps ---------------------------------------
    _orig_device = torch.device

    def _device(*a, **k):
        if a and isinstance(a[0], str) and a[0].startswith("cuda"):
            a = ("mps",) + a[1:]
        return _orig_device(*a, **k)

    torch.device = _device

    # --- Tensor.to / Module.to remap ---------------------------------------
    def _wrap_to(orig):
        def to(self, *a, **k):
            if a:
                a = (_remap_device(a[0]),) + a[1:]
            if "device" in k:
                k["device"] = _remap_device(k["device"])
            return orig(self, *a, **k)
        return to

    torch.Tensor.to = _wrap_to(torch.Tensor.to)
    torch.nn.Module.to = _wrap_to(torch.nn.Module.to)

    # --- Tensor.pin_memory() -> no-op (no pinned memory on MPS) -------------
    torch.Tensor.pin_memory = lambda self, *a, **k: self

    # --- torch.load(map_location='cuda') -----------------------------------
    _orig_load = torch.load

    def _load(*a, **k):
        if "map_location" in k:
            k["map_location"] = _remap_device(k["map_location"])
        return _orig_load(*a, **k)

    torch.load = _load

    # --- DataLoader: no pinned memory on MPS -------------------------------
    _orig_dl_init = torch.utils.data.DataLoader.__init__

    def _dl_init(self, *a, **k):
        k["pin_memory"] = False
        return _orig_dl_init(self, *a, **k)

    torch.utils.data.DataLoader.__init__ = _dl_init

    # --- AMP: default off (fp32) for correctness, knob to re-enable ---------
    if os.environ.get("MPSIFY_AMP") != "1":
        import contextlib

        @contextlib.contextmanager
        def _noop_autocast(*a, **k):
            yield

        class _NoopScaler:
            def __init__(self, *a, **k): pass
            def scale(self, x): return x
            def step(self, opt, *a, **k): return opt.step(*a, **k)
            def update(self, *a, **k): pass
            def unscale_(self, *a, **k): pass
            def get_scale(self): return 1.0
            def state_dict(self): return {}
            def load_state_dict(self, *a, **k): pass

        torch.cuda.amp.autocast = _noop_autocast
        torch.cuda.amp.GradScaler = _NoopScaler
        if hasattr(torch, "amp"):
            torch.amp.GradScaler = _NoopScaler

    # --- DataParallel -> identity (single device) --------------------------
    _orig_dp = torch.nn.DataParallel

    def _data_parallel(module, *a, **k):
        return module

    torch.nn.DataParallel = _data_parallel

    # --- distributed: nccl -> gloo -----------------------------------------
    if hasattr(torch, "distributed"):
        _orig_ipg = torch.distributed.init_process_group

        def _ipg(*a, **k):
            if k.get("backend") == "nccl":
                k["backend"] = "gloo"
            if a and a[0] == "nccl":
                a = ("gloo",) + a[1:]
            return _orig_ipg(*a, **k)

        torch.distributed.init_process_group = _ipg

    # --- detect unshimmable libs -------------------------------------------
    for lib in _UNSHIMMABLE:
        if lib in sys.modules or importlib.util.find_spec(lib) is not None:
            _report.note_missing_lib(lib)

    _patched = True


def load(path, *, fp32: bool = True, **kwargs):
    """Load a checkpoint saved on CUDA straight onto MPS.

    Weights are device-agnostic numbers, so a CUDA-trained checkpoint runs on
    MPS as-is — the only snag is torch restoring tensors to their saved device.
    This forces map_location to mps (cpu if no MPS) and, by default, upcasts
    fp16 tensors to fp32 (MPS half-precision has known-broken ops).
    """
    import torch
    patch()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    kwargs.setdefault("map_location", dev)
    obj = torch.load(path, **kwargs)
    if fp32:
        obj = _upcast_fp16(obj)
    return obj


def _upcast_fp16(obj):
    """Recursively cast float16 tensors -> float32 in tensors/dicts/lists."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.float() if obj.dtype == torch.float16 else obj
    if isinstance(obj, dict):
        return {k: _upcast_fp16(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_upcast_fp16(v) for v in obj)
    return obj


patch()

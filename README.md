# mpsify

[![PyPI](https://img.shields.io/pypi/v/mpsify)](https://pypi.org/project/mpsify/)
[![Python](https://img.shields.io/pypi/pyversions/mpsify)](https://pypi.org/project/mpsify/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Run CUDA-hardcoded PyTorch repos on Apple Silicon (MPS) — with zero source edits.**

You inherit someone's training script. It's full of `.cuda()`, `device='cuda'`,
`torch.cuda.amp`, `map_location='cuda'`. PyTorch-MPS could run the actual math
fine on your Mac — but the code dies before it gets the chance. `mpsify` patches
`torch` at import time so all of that transparently retargets to MPS.

```bash
pip install mpsify
```

## Usage

Point it at any script — no edits to their code:

```bash
python -m mpsify train.py --epochs 10 --lr 1e-3
# or, after install, the console script:
mpsify train.py --epochs 10 --lr 1e-3
```

Or drop one line at the top of your entry file:

```python
import mpsify  # patches torch on import
```

The wrapper (`python -m mpsify`) is preferred: it sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` *before* torch loads, which the import form
can't always guarantee.

## What it does

| CUDA thing | Retargeted to |
|---|---|
| `torch.cuda.is_available()` | `True` |
| `.cuda()`, `.to('cuda')`, `device='cuda'` | `mps` |
| `torch.device('cuda')` | `mps` |
| `torch.load(map_location='cuda')` | `mps` |
| `DataLoader(pin_memory=True)` | `pin_memory=False` (no pinned memory on MPS) |
| `torch.cuda.amp` autocast / `GradScaler` | no-op, fp32 (see knobs) |
| `nn.DataParallel` | identity (single device) |
| `nccl` backend | `gloo` |

Ops with no Metal kernel fall back to CPU automatically. `mpsify` catches those
fallbacks, warns once per op live, and prints a summary at exit — so you can
see exactly which ops are your latency hot spots.

Libraries with no Metal backend at all (`bitsandbytes`, `apex`, `deepspeed`,
`flash_attn`, `triton`) are detected and reported loudly instead of crashing
cryptically.

## Pre-flight check

See what mpsify will do to a script **without running it**:

```bash
python -m mpsify doctor train.py
```

Reports every CUDA call it will remap and any library it can't fix
(flash-attention, bitsandbytes, ...). Exit code is non-zero if something is
unshimmable — handy in CI.

## Loading a CUDA-trained checkpoint

Weights are just numbers — a model trained on CUDA runs on MPS unchanged. The
only snag is `torch.load` restoring tensors to their saved `cuda` device.
`mpsify.load` handles that (and upcasts fp16 → fp32, since MPS half-precision
has rough edges):

```python
import mpsify
state = mpsify.load("model_trained_on_a100.pth")   # lands on MPS
model.load_state_dict(state)
```

## Diagnosing slow ops

```bash
python -m mpsify --profile train.py
```

Runs a dispatch-level profiler that counts calls and times ops. This adds
per-op overhead — use it for a diagnostic pass, not production.

## Knobs

| Env var | Effect |
|---|---|
| `MPSIFY_AMP=1` | Re-enable AMP/autocast (default off = fp32, correct but slower). AMP on MPS is where correctness gets dicey. |
| `MPSIFY_QUIET=1` | Suppress live fallback warnings; keep the exit summary. |

## Tested on

Real pretrained models, output verified numerically identical to CPU
(max abs diff ~1e-6) on macOS + torch 2.5.1:

- torchvision `resnet18`, `efficientnet_b0`, `vit_b_16`
- timm `mobilenetv3_small_100`
- A 3-step fine-tuning loop (loss decreases, params on `mps:0`)
- Loading fp16/CUDA checkpoints via `mpsify.load`

## Scope

Handles pure-PyTorch repos (torchvision / timm models — ResNet, EfficientNet,
ViT, etc.). It does **not** translate custom `.cu`/Triton kernels or make
CUDA-only libraries (flash-attention, DeepSpeed, apex) actually work — those are
detected and reported, not fixed.

## License

MIT

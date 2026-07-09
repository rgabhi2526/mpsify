"""A deliberately CUDA-hardcoded training script — the kind mpsify targets.

Run it two ways:
    python examples/demo.py            # dies: no NVIDIA GPU
    python -m mpsify examples/demo.py  # runs on Apple MPS, zero edits
"""
import torch
import torch.nn as nn

assert torch.cuda.is_available(), "needs a CUDA GPU"     # fails on a Mac
device = torch.device("cuda")

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2)).cuda()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

X = torch.randn(64, 10).to("cuda")
y = torch.randint(0, 2, (64,)).cuda()

for step in range(5):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()
    print(f"step {step}  loss {loss.item():.4f}  device {next(model.parameters()).device}")

print("done — trained on", next(model.parameters()).device)

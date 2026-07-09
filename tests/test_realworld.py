"""Real pretrained models: prove CUDA-trained weights run correctly on MPS.

Downloads small pretrained checkpoints — auto-skips if offline or no MPS.
Run: python -m pytest tests/test_realworld.py -q
"""
import warnings

import pytest
import torch

import mpsify

warnings.filterwarnings("ignore")

MPS = torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS, reason="needs MPS")


def _download_ok(fn):
    try:
        return fn()
    except Exception as e:  # network / hub failure
        pytest.skip(f"download unavailable: {e}")


def _agree(model, x, atol):
    """Assert model output on MPS (via .cuda()) matches CPU reference."""
    model = model.eval()
    with torch.no_grad():
        ref = model(x)                       # CPU
        out = model.cuda()(x.cuda()).cpu()   # shim -> MPS
    assert next(model.parameters()).device.type == "mps"
    assert torch.allclose(out, ref, atol=atol), (out - ref).abs().max().item()


def test_torchvision_resnet18():
    import torchvision.models as M
    m = _download_ok(lambda: M.resnet18(weights=M.ResNet18_Weights.DEFAULT))
    _agree(m, torch.randn(1, 3, 224, 224), atol=1e-3)


def test_torchvision_efficientnet_b0():
    import torchvision.models as M
    m = _download_ok(lambda: M.efficientnet_b0(weights=M.EfficientNet_B0_Weights.DEFAULT))
    _agree(m, torch.randn(1, 3, 224, 224), atol=1e-3)


def test_torchvision_vit_b_16():
    import torchvision.models as M
    m = _download_ok(lambda: M.vit_b_16(weights=M.ViT_B_16_Weights.DEFAULT))
    _agree(m, torch.randn(1, 3, 224, 224), atol=1e-3)


def test_timm_mobilenetv3():
    timm = pytest.importorskip("timm")
    m = _download_ok(lambda: timm.create_model("mobilenetv3_small_100", pretrained=True))
    _agree(m, torch.randn(1, 3, 224, 224), atol=1e-3)


def test_load_cuda_checkpoint(tmp_path):
    # Simulate a checkpoint; mpsify.load should land it on MPS.
    sd = {"w": torch.randn(4, 4), "half": torch.randn(4).half()}
    p = tmp_path / "ckpt.pt"
    torch.save(sd, p)
    out = mpsify.load(p, weights_only=True)
    assert out["w"].device.type == "mps"
    assert out["half"].dtype == torch.float32  # fp16 upcast by default


def test_transformers_sentiment():
    """Real HF transformer: import, .to('cuda'), forward, and pipeline(device=0)."""
    pytest.importorskip("transformers")
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, pipeline)
    name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    tok = _download_ok(lambda: AutoTokenizer.from_pretrained(name))
    model = _download_ok(
        lambda: AutoModelForSequenceClassification.from_pretrained(
            name, use_safetensors=True)).to("cuda")
    assert next(model.parameters()).device.type == "mps"

    enc = tok("this movie was fantastic", return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits
    assert logits.device.type == "mps"
    assert model.config.id2label[logits.argmax().item()] == "POSITIVE"

    # pipeline(device=0) exercises `with torch.cuda.device(i):`
    pipe = pipeline("text-classification", model=model, tokenizer=tok, device=0)
    assert pipe("what a terrible waste of time")[0]["label"] == "NEGATIVE"


def test_training_micro_run():
    import torchvision.models as M
    m = M.resnet18(num_classes=10).cuda()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    x = torch.randn(2, 3, 64, 64).cuda()
    y = torch.randint(0, 10, (2,)).cuda()
    losses = []
    for _ in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert next(m.parameters()).device.type == "mps"
    assert losses[-1] < losses[0]  # learning happened

"""Self-checks for the mpsify shim. Run: python -m pytest tests/

No GPU required — device asserts adapt to whether MPS is present.
"""
import torch

import mpsify  # noqa: F401  -- import triggers patch()

MPS = torch.backends.mps.is_available()
_EXPECT = "mps" if MPS else "cpu"  # no MPS -> torch coerces mps->cpu at use


def test_cuda_is_available_lies():
    assert torch.cuda.is_available() is True
    assert torch.cuda.device_count() == 1


def test_device_remapped_at_use_site():
    # torch.device('cuda') stays a real cuda descriptor (keeps isinstance /
    # `device | str` working); remapping happens where it's used.
    if MPS:
        assert torch.zeros(2).to(torch.device("cuda")).device.type == "mps"
        assert torch.zeros(2, device=torch.device("cuda:3")).device.type == "mps"


def test_tensor_cuda_lands_on_mps():
    t = torch.zeros(2)
    if MPS:
        assert t.cuda().device.type == "mps"
        assert t.to("cuda").device.type == "mps"


def test_module_cuda():
    m = torch.nn.Linear(2, 2)
    if MPS:
        assert next(m.cuda().parameters()).device.type == "mps"


def test_dataloader_pin_memory_forced_off():
    ds = torch.utils.data.TensorDataset(torch.zeros(4, 2))
    dl = torch.utils.data.DataLoader(ds, batch_size=2, pin_memory=True)
    assert dl.pin_memory is False


def test_load_map_location_remapped(tmp_path):
    p = tmp_path / "t.pt"
    torch.save(torch.zeros(2), p)
    out = torch.load(p, map_location="cuda", weights_only=True)
    assert out.device.type == _EXPECT


def test_dataparallel_is_identity():
    m = torch.nn.Linear(2, 2)
    assert torch.nn.DataParallel(m) is m


def test_amp_disabled_by_default():
    # GradScaler is a no-op passthrough unless MPSIFY_AMP=1
    scaler = torch.cuda.amp.GradScaler()
    x = torch.ones(2)
    assert scaler.scale(x) is x
    assert scaler.get_scale() == 1.0


def test_pin_memory_method_is_noop():
    t = torch.zeros(2)
    assert t.pin_memory() is t  # would RuntimeError on MPS without the shim


def test_integer_and_indexed_devices():
    if MPS:
        assert torch.zeros(2).to(0).device.type == "mps"
        assert torch.zeros(2).cuda(0).device.type == "mps"
        assert torch.zeros(2).to("cuda:1").device.type == "mps"


def test_torch_device_stays_a_type():
    # Replacing torch.device with a function broke these — regression guards.
    assert isinstance(torch.device, type)
    assert isinstance(torch.device("cpu"), torch.device)
    _ = torch.device | str  # PEP-604 union (used in transformers annotations)


def test_factory_device_string_remapped():
    if MPS:
        assert torch.zeros(2, device="cuda").device.type == "mps"
        assert torch.tensor([1.0], device="cuda").device.type == "mps"


def test_cuda_device_context_is_noop():
    # `with torch.cuda.device(i):` is used by HF pipelines.
    with torch.cuda.device(0):
        pass


def test_load_upcasts_fp16(tmp_path):
    p = tmp_path / "c.pt"
    torch.save({"a": torch.randn(3).half()}, p)
    out = mpsify.load(p, weights_only=True)
    assert out["a"].dtype == torch.float32
    assert out["a"].device.type == _EXPECT

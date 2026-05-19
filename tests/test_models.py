"""Shape + smoke tests for every model (except mamba1d which needs CUDA + special install)."""
from __future__ import annotations

import pytest
import torch

from csnet.models import MODELS, build_model
from csnet.utils import NUM_CLASSES


@pytest.fixture
def tiny_input():
    """Batch of 4 synthetic spectra, length 512 (matches smoke config)."""
    torch.manual_seed(0)
    return torch.randn(4, 1, 512)


@pytest.mark.parametrize("name", [n for n in MODELS if n != "mamba1d"])
def test_forward_shape(name, tiny_input):
    model = build_model(name)
    model.eval()
    with torch.no_grad():
        out = model(tiny_input)
    assert out.shape == (4, NUM_CLASSES), f"{name}: expected (4, {NUM_CLASSES}), got {out.shape}"


@pytest.mark.parametrize("name", [n for n in MODELS if n != "mamba1d"])
def test_backward(name, tiny_input):
    model = build_model(name)
    model.train()
    out = model(tiny_input)
    target = torch.tensor([0, 3, 5, 6])
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()
    # Check that at least one parameter received a non-zero gradient
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, f"{name}: no parameter received a non-trivial gradient"


def test_mamba1d_raises_clear_error_without_install(tiny_input):
    """If mamba-ssm isn't installed, the wrapper should raise a helpful ImportError
    pointing at the right install commands — not a cryptic AttributeError."""
    try:
        import mamba_ssm  # noqa: F401
        pytest.skip("mamba-ssm IS installed; skipping the negative test")
    except Exception:
        pass

    with pytest.raises(ImportError) as exc_info:
        build_model("mamba1d")
    msg = str(exc_info.value)
    assert "mamba-ssm" in msg
    assert "conda" in msg or "pip" in msg

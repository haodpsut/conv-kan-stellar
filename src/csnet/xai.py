"""Integrated Gradients (Sundararajan et al., 2017) for 1D spectra.

For input x, baseline x', and model f producing logits, the Integrated
Gradient attribution for class c at pixel i is:

    IG_i(x; c) = (x_i - x'_i) * mean_{k=1..N} grad_{x_i} f_c(x' + k/N * (x - x'))

The baseline x' = 0 (zero spectrum) by default. N steps of Riemann sum
(default 50) trade accuracy for runtime. The output is a [seq_len] array
of contribution scores; positive means "pushed the prediction toward
class c", negative means "pushed away".

References
----------
Sundararajan, M., Taly, A. & Yan, Q. "Axiomatic Attribution for Deep Networks."
ICML 2017. https://arxiv.org/abs/1703.01365
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)   # [1, 1, L]
    if x.dim() == 2:
        return x.unsqueeze(1)                # [B, 1, L]
    return x


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
    batch_size: int = 16,
) -> np.ndarray:
    """Compute IG attributions for a SINGLE spectrum.

    Args
    ----
    model         : nn.Module producing logits [B, num_classes]
    x             : tensor [seq_len] or [1, seq_len] or [1, 1, seq_len]
    target_class  : int, which class to attribute toward
    baseline      : same shape as x, defaults to zeros
    n_steps       : Riemann sum steps (50 is standard)
    batch_size    : how many interpolation steps to evaluate per forward pass

    Returns
    -------
    attribution : numpy array [seq_len], contribution per pixel
    """
    model.eval()
    device = next(model.parameters()).device

    x = _ensure_batch(x).to(device)   # [1, 1, L]
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = _ensure_batch(baseline).to(device)

    # Build interpolation steps: alphas in (0, 1]
    alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=device)

    grads_accum = torch.zeros_like(x)
    for i in range(0, n_steps, batch_size):
        batch_alphas = alphas[i:i + batch_size]
        # [b, 1, L]: baseline + alpha * (x - baseline)
        interp = baseline + batch_alphas.view(-1, 1, 1) * (x - baseline)
        interp = interp.detach().requires_grad_(True)
        logits = model(interp)
        target = logits[:, target_class].sum()
        grad = torch.autograd.grad(target, interp, retain_graph=False)[0]
        grads_accum = grads_accum + grad.sum(dim=0, keepdim=True)

    avg_grads = grads_accum / n_steps
    attribution = (x - baseline) * avg_grads
    return attribution.squeeze().detach().cpu().numpy()

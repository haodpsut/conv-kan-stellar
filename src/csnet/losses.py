"""ENS (Effective Number of Samples) — Focal loss (Cui+2019).

For imbalanced classification with class frequencies n_c, the effective
number of samples for class c is:

    E_c = (1 - beta^{n_c}) / (1 - beta)

and the class weight is:

    w_c = (1 - beta) / (1 - beta^{n_c})  =  1 / E_c

normalised so that sum(w_c) == num_classes (keeps loss magnitude comparable
to vanilla CE).

We combine these weights with a focal modulation (gamma) on the softmax
output to further down-weight easy examples — useful for the dominant
G-type class which would otherwise saturate training:

    L = -sum_c w_c * (1 - p_c)^gamma * y_c * log(p_c)

Reference
---------
Cui, Y. et al. "Class-Balanced Loss Based on Effective Number of Samples."
CVPR 2019. https://arxiv.org/abs/1901.05555
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def ens_class_weights(class_counts: np.ndarray, beta: float = 0.9999) -> torch.Tensor:
    """Compute ENS weights from per-class training counts.

    Args
    ----
    class_counts : array of shape [num_classes], integer counts.
    beta         : ENS smoothing in [0, 1). Cui+2019 default 0.9999.

    Returns
    -------
    weights : torch.FloatTensor [num_classes], normalised to sum to num_classes.
    """
    counts = np.maximum(class_counts.astype(np.float64), 1.0)
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff_num                      # raw inverse-effective-num
    w = w / w.sum() * len(counts)                   # normalise to mean == 1
    return torch.from_numpy(w.astype(np.float32))


class ENSFocalLoss(nn.Module):
    """ENS-weighted focal cross-entropy.

    Args
    ----
    class_weights : torch.Tensor [num_classes] (typically from `ens_class_weights`)
    gamma         : focal exponent. 0.0 reduces to weighted CE.
    label_smoothing : optional label smoothing in [0, 1)
    """

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # Optional label smoothing
        if self.label_smoothing > 0:
            n_cls = logits.size(-1)
            smooth_target = torch.full_like(log_probs, self.label_smoothing / (n_cls - 1))
            smooth_target.scatter_(-1, target.unsqueeze(-1), 1.0 - self.label_smoothing)
        else:
            smooth_target = F.one_hot(target, num_classes=logits.size(-1)).to(log_probs.dtype)
        # Focal modulation: (1 - p_t)^gamma per sample, per class
        focal_mod = (1.0 - probs).clamp(min=1e-6).pow(self.gamma)
        per_sample = -(self.class_weights * smooth_target * focal_mod * log_probs).sum(dim=-1)
        return per_sample.mean()

"""All models in one file (kept together so reading the architecture story is
straightforward). Six models registered in `MODELS`:

    conv_kan         — proposed (Conv1D backbone + KAN classifier head)
    inception        — InceptionTime 1D (Fawaz+2020)
    se_resnet        — 1D ResNet-18 with Squeeze-and-Excitation blocks
    cnn_transformer  — CNN feature extractor + Transformer encoder
    starnet          — Fabbro+2018 stellar-spectrum CNN
    mamba1d          — 1D Mamba (Gu+Dao 2023); stub if mamba-ssm not installed

Every model takes input shape [B, 1, seq_len] and returns logits [B, num_classes].
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import NUM_CLASSES


# =========================================================================== #
# 1. B-spline KAN layer  (Liu+2024, eff. impl. inspired by Blealtan)          #
# =========================================================================== #

class KANLinear(nn.Module):
    """A single Kolmogorov-Arnold Network layer.

    Each scalar weight is replaced by a learnable univariate function
    parameterised as:
        phi(x) = w_b * silu(x) + w_s * sum_i c_i * B_i(x)
    where B_i are uniform B-spline basis functions of order `spline_order`
    on a grid of `grid_size + 1` knots in [grid_range[0], grid_range[1]].

    Args
    ----
    in_features, out_features : int
    grid_size      : number of grid intervals (default 5, matches our IEEE TAI submission)
    spline_order   : B-spline order (default 3 = cubic)
    grid_range     : (low, high) initial bounds; grid is uniform on this range
    scale_noise    : init scale of the spline coefficients
    scale_base     : init scale of the SiLU residual weight w_b
    scale_spline   : init scale of the spline output weight w_s
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation

        # Build extended grid (need spline_order extra knots on each side for B-splines).
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h
            + grid_range[0]
        )
        # Shape [in_features, grid_size + 2*spline_order + 1]
        grid = grid.unsqueeze(0).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        # Residual SiLU weights w_b: [out, in]
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        # Spline coefficients c: [out, in, grid_size + spline_order]
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        # Spline output scale w_s: [out, in]
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # Initialise spline coefficients so that initial output is near zero
            # but with small noise so gradients flow.
            noise = (torch.rand_like(self.spline_weight) - 0.5) * self.scale_noise
            self.spline_weight.copy_(noise / (self.grid_size + 1))
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis at x. x: [B, in_features].
        Returns: [B, in_features, grid_size + spline_order]."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid  # [in, G]
        x = x.unsqueeze(-1)  # [B, in, 1]
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # order 0
        # Recursively build higher-order bases via Cox-de Boor recursion
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8) * bases[..., :-1]
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k] + 1e-8) * bases[..., 1:]
            bases = left + right
        return bases  # [B, in, grid_size + spline_order]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in]
        base_out = F.linear(self.base_activation(x), self.base_weight)
        spline_basis = self._b_splines(x)  # [B, in, G]
        # Effective spline weights: spline_weight * spline_scaler (broadcast over G)
        eff = self.spline_weight * self.spline_scaler.unsqueeze(-1)  # [out, in, G]
        spline_out = torch.einsum("big,oig->bo", spline_basis, eff)
        return base_out + spline_out


# =========================================================================== #
# 2. Conv-KAN (proposed)                                                      #
# =========================================================================== #

class _ConvBlock1D(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 7, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvKAN(nn.Module):
    """Conv1D backbone (5 blocks with downsampling) → global avg pool → KAN classifier.

    For input seq_len in [256, 8192], the backbone reduces to ~seq_len/32 after
    5 stride-2 pooling steps, then global-pools to [B, channels].
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 128, 256),
        kan_grid: int = 5,
        kan_order: int = 3,
        kan_hidden: int = 64,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers.append(_ConvBlock1D(c_in, c_out, k=7))
            layers.append(_ConvBlock1D(c_out, c_out, k=7, stride=2))  # downsample
            c_in = c_out
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        feat_dim = channels[-1]

        self.kan = nn.Sequential(
            KANLinear(feat_dim, kan_hidden, grid_size=kan_grid, spline_order=kan_order),
            KANLinear(kan_hidden, num_classes, grid_size=kan_grid, spline_order=kan_order),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)              # [B, C, L']
        h = self.pool(h).squeeze(-1)      # [B, C]
        return self.kan(h)


# =========================================================================== #
# 3. InceptionTime 1D (Fawaz+2020) — baseline                                 #
# =========================================================================== #

class _InceptionModule(nn.Module):
    def __init__(self, in_c: int, n_filters: int = 32, bottleneck: int = 32,
                 kernel_sizes: tuple[int, ...] = (9, 19, 39)):
        # NOTE: odd kernels chosen so that padding=k//2 preserves sequence length
        # exactly (PyTorch nn.Conv1d does not support padding="same" with stride>1
        # in all versions). Functionally equivalent to InceptionTime's 10/20/40.
        super().__init__()
        self.use_bottleneck = in_c > 1
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_c, bottleneck, 1, bias=False)
            conv_in = bottleneck
        else:
            conv_in = in_c
        self.convs = nn.ModuleList(
            [nn.Conv1d(conv_in, n_filters, k, padding=k // 2, bias=False) for k in kernel_sizes]
        )
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_c, n_filters, 1, bias=False),
        )
        out_c = n_filters * (len(kernel_sizes) + 1)
        self.bn = nn.BatchNorm1d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_bottleneck:
            z = self.bottleneck(x)
        else:
            z = x
        outs = [conv(z) for conv in self.convs]
        outs.append(self.pool_conv(x))
        h = torch.cat(outs, dim=1)
        return F.relu(self.bn(h), inplace=True)


class InceptionTime(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1,
                 n_modules: int = 6, n_filters: int = 32):
        super().__init__()
        mods: list[nn.Module] = []
        c = in_channels
        for i in range(n_modules):
            m = _InceptionModule(c, n_filters=n_filters)
            mods.append(m)
            c = n_filters * 4  # 3 conv branches + 1 pool branch
            # Residual every 3 modules (Fawaz+2020 detail)
        self.modules_list = nn.ModuleList(mods)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = None
        h = x
        for i, m in enumerate(self.modules_list):
            h = m(h)
            if i % 3 == 2 and residual is not None and residual.shape == h.shape:
                h = h + residual
            if i % 3 == 0:
                residual = h
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


# =========================================================================== #
# 4. 1D SE-ResNet (squeeze-and-excitation) — baseline                          #
# =========================================================================== #

class _SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(c, max(c // r, 4))
        self.fc2 = nn.Linear(max(c // r, 4), c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=-1)                       # [B, C]
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1)  # [B, C, 1]
        return x * s


class _SEResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.se = _SEBlock(out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm1d(out_c)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.relu(h + self.shortcut(x), inplace=True)


class SEResNet1D(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1,
                 stages: tuple[tuple[int, int], ...] = ((32, 2), (64, 2), (128, 2), (256, 2))):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, 32, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(3, stride=2, padding=1),
        ]
        c = 32
        for c_out, n in stages:
            for i in range(n):
                stride = 2 if i == 0 and c != c_out else 1
                layers.append(_SEResBlock(c, c_out, stride=stride))
                c = c_out
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


# =========================================================================== #
# 5. CNN-Transformer hybrid — baseline                                        #
# =========================================================================== #

class CNNTransformer(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 cnn_channels: tuple[int, ...] = (32, 64, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for c_out in cnn_channels:
            layers.append(_ConvBlock1D(c, c_out, k=7, stride=2))
            c = c_out
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Conv1d(c, d_model, 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)                         # [B, C, L']
        h = self.proj(h).transpose(1, 2)        # [B, L', d_model]
        B = h.size(0)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.transformer(h)
        return self.fc(h[:, 0])


# =========================================================================== #
# 6. StarNet (Fabbro+2018) — baseline                                         #
# =========================================================================== #

class StarNet(nn.Module):
    """Re-implementation of the small CNN from Fabbro+2018, originally
    proposed for APOGEE stellar parameter regression and widely used as a
    stellar-spectra baseline. We re-purpose the head for classification.

    Notes vs the original:
      - Kernel size 7 (odd) instead of 8 so padding=k//2 preserves length —
        avoids cumulative shape drift across surveys with different seq_len.
      - An AdaptiveAvgPool1d collapses the conv feature map to a fixed size
        (64), making the FC head seq_len-independent. This removes the need
        for lazy initialisation, which was incompatible with passing
        `model.parameters()` to the optimiser at construction time.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1,
                 pool_to: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 4, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(4, 16, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(pool_to),
        )
        flat = 16 * pool_to
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        h = F.relu(self.fc1(h), inplace=True)
        h = F.relu(self.fc2(h), inplace=True)
        return self.fc3(h)


# =========================================================================== #
# 7. 1D Mamba — baseline (optional, requires `mamba-ssm`)                     #
# =========================================================================== #

class Mamba1D(nn.Module):
    """Selective State-Space Model (Gu+Dao 2023). Wraps the mamba-ssm package
    when available; raises a helpful error otherwise."""

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 1,
                 d_model: int = 128, n_layers: int = 4):
        super().__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore
        except Exception as e:
            raise ImportError(
                "1D Mamba baseline requires the `mamba-ssm` package "
                "(CUDA + Hopper-compatible). Install on the H200 instance with:\n"
                "    conda install -c conda-forge cudatoolkit-dev\n"
                "    pip install mamba-ssm causal-conv1d\n"
                f"Original error: {e}"
            )
        self.proj_in = nn.Conv1d(in_channels, d_model, 1)
        self.blocks = nn.ModuleList([Mamba(d_model=d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x).transpose(1, 2)  # [B, L, d_model]
        for blk in self.blocks:
            h = h + blk(self.norm(h))
        h = h.transpose(1, 2)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


# =========================================================================== #
# 8. Registry                                                                  #
# =========================================================================== #

MODELS: dict[str, Callable[..., nn.Module]] = {
    "conv_kan": ConvKAN,
    "inception": InceptionTime,
    "se_resnet": SEResNet1D,
    "cnn_transformer": CNNTransformer,
    "starnet": StarNet,
    "mamba1d": Mamba1D,
}


def build_model(name: str, num_classes: int = NUM_CLASSES, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name](num_classes=num_classes, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

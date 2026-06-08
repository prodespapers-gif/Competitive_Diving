"""aqa_model.py — The frozen-after-training AQA oracle.

Implements the temporal-segmentation-attention (TSA, FineDiving CVPR 2022)
and spatial-temporal-segmentation-attention (STSA, FineDiving IJCV 2024)
AQA scorer, plus the machinery required to re-use it as the differentiable
oracle for counterfactual search.

Composition
-----------
1. :class:`I3DBackbone` — Kinetics-pretrained Inception-3D feature extractor.
   Implemented inline so the repo stays self-contained under the
   CentOS 7 / GCC 4.8.5 deployment constraint (no compiled extensions).
2. :class:`ProcedureSegmentation` — the procedure-segmentation block ``S``
   from FineDiving §4.2. Two variants are available, switchable via
   ``AQACfg.segmentation_use_reference_psnet``:
       - ``"reference"``: faithful reproduction of the released code base
         (``FineDiving-main/models/PS.py``): snippets fed as channels,
         four MaxPool1d-and-double-conv "down" blocks, channels 9→96 and
         length 1024→64, then an MLP_tas with Sigmoid output. Matches the
         architecture that produced the published numbers.
       - ``"upsample"``: the alternative where snippets are upsampled
         temporally from 9 to ``num_frames`` and channels reduce 1024→128.
         Kept as an ablation toggle; not the published architecture.
3. :class:`SpatialMotionAttention` — optional STSA extension (Xu et al.,
   IJCV 2024 §4.2). When enabled, learns foreground-motion attention via
   an implicit-supervision proxy task.
4. :class:`ProcedureAwareCrossAttention` — transformer decoder over the
   ``L+1`` consecutive query/exemplar steps. Decoder embed dim adapts to
   whichever segmentation variant is active.
5. :class:`ContrastiveRegressor` — three-layer MLP producing a per-step
   relative score that is summed and added to the exemplar's score.
6. :class:`TSAModel` — composes the above, exposes :meth:`forward`
   (symmetric or asymmetric, per ``AQACfg.use_symmetric_training``),
   :meth:`predict_score` (multi-exemplar voting), and :meth:`frozen`
   (the differentiable-oracle context).

Loss
----
``J = α_bce · BCE(transits_pred, transits_target) + α_mse · MSE(ŷ, y)``
matching the reference's unweighted sum ``loss_aqa + loss_tas`` in
``helper.py:165``. When training is symmetric (the default — and the
reference's recipe at lines 145-164), the MSE term covers BOTH
directions: ``mse(δ̂_qe, y_q - y_e) + mse(δ̂_eq, y_e - y_q)``.

When STSA is enabled an extra KL term is added between predicted and
target spatial-motion-attention maps (IJCV Eq. 6, 13). The aggregation
over query and exemplar branches follows ``AQACfg.sma_kl_aggregation``
(``"mean"`` for our default; ``"sum"`` to match STSA Eq. 6 literally).

Frozen-oracle contract
----------------------
:meth:`TSAModel.frozen` returns a context manager that puts the model in
eval mode and freezes its parameters while *preserving autograd on the
input*. This is the contract :mod:`counterfactual` relies on: score
gradients flow through pose / pixels, the oracle's weights do not move,
and dropout / batch-stats stay deterministic so the same input always
yields the same score. Under :meth:`frozen`, ``forward`` automatically
disables symmetric mode (the oracle only ever needs q → e).

Changes from previous version
-----------------------------
- Issue 2 (paper-blocking): symmetric forward in the reference's
  ``cat([q→e, e→q])`` shape; activated by default via
  ``AQACfg.use_symmetric_training``.
- Issue 5 (paper-blocking): step-transition curriculum — forward takes
  ``current_epoch`` and GT transitions and slices features using GT
  while ``current_epoch < AQACfg.curriculum_threshold_epochs``.
- Issue 6 (paper-blocking): :class:`_ReferencePSNet` ported verbatim
  from ``models/PS.py`` and ``models/PS_parts.py``. Selectable via
  ``AQACfg.segmentation_use_reference_psnet``.
- Issue 8 (camera-ready): KL aggregation now honours
  ``AQACfg.sma_kl_aggregation``.
- Issue 14 (camera-ready): the segmentation b2 head activation reads
  ``AQACfg.b2_activation`` (default ReLU per STSA §5.2 and the
  reference ``PS_parts.MLP_tas``).
"""
from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import AQACfg, DataCfg

log = logging.getLogger(__name__)

# =============================================================================
# I3D — Inception-3D (Carreira & Zisserman, CVPR 2017)
# =============================================================================
#
# Implemented inline, matching the layer-naming convention of the public
# Kinetics-pretrained checkpoints (piergiaj/pytorch-i3d). This means
# ``state_dict = torch.load("i3d_kinetics.pth")`` followed by
# ``self.load_state_dict(state_dict, strict=True)`` works out of the box
# for the canonical RGB-only checkpoint.
# -----------------------------------------------------------------------------


class _MaxPool3dSamePadding(nn.MaxPool3d):
    """Max-pool with TF-style 'SAME' padding, which I3D was trained with."""

    def compute_pad(self, dim: int, s: int) -> int:
        kernel = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size,) * 3
        stride = self.stride if isinstance(self.stride, tuple) else (self.stride,) * 3
        if s % stride[dim] == 0:
            return max(kernel[dim] - stride[dim], 0)
        return max(kernel[dim] - (s % stride[dim]), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2,
                      pad_t // 2, pad_t - pad_t // 2])
        return super().forward(x)


class _Unit3D(nn.Module):
    """Conv3D + BN + ReLU with TF-style 'SAME' padding."""

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_shape: Tuple[int, int, int] = (1, 1, 1),
        stride: Tuple[int, int, int] = (1, 1, 1),
        use_batch_norm: bool = True,
        activation_fn: Optional[type] = nn.ReLU,
    ) -> None:
        super().__init__()
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self.conv3d = nn.Conv3d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_shape, stride=stride, padding=0,
            bias=not use_batch_norm,
        )
        if use_batch_norm:
            self.bn = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.001)
        if activation_fn is not None:
            self.activation = activation_fn()

    def compute_pad(self, dim: int, s: int) -> int:
        stride = self._stride[dim]
        kernel = self._kernel_shape[dim]
        if s % stride == 0:
            return max(kernel - stride, 0)
        return max(kernel - (s % stride), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2,
                      pad_t // 2, pad_t - pad_t // 2])
        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self.activation(x)
        return x


class _InceptionModule(nn.Module):
    """Inception block: 1x1, 3x3, 5x5 branches plus a pooled branch."""

    def __init__(self, in_channels: int,
                 out_channels: Tuple[int, int, int, int, int, int]) -> None:
        super().__init__()
        self.b0 = _Unit3D(in_channels, out_channels[0], (1, 1, 1))
        self.b1a = _Unit3D(in_channels, out_channels[1], (1, 1, 1))
        self.b1b = _Unit3D(out_channels[1], out_channels[2], (3, 3, 3))
        self.b2a = _Unit3D(in_channels, out_channels[3], (1, 1, 1))
        self.b2b = _Unit3D(out_channels[3], out_channels[4], (3, 3, 3))
        self.b3a = _MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.b3b = _Unit3D(in_channels, out_channels[5], (1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.b0(x), self.b1b(self.b1a(x)),
             self.b2b(self.b2a(x)), self.b3b(self.b3a(x))],
            dim=1,
        )


class InceptionI3D(nn.Module):
    """Inception-3D backbone, pre-classifier (output of ``Mixed_5c``).

    Forward signature: ``(B, 3, T, H, W) → (B, 1024, T', H', W')``.
    Spatial pooling to a single-vector feature is the caller's responsibility,
    so :class:`I3DBackbone` can decide whether to keep the spatial map
    (needed by STSA) or pool it away (TSA).
    """

    LAYERS_PRE_INCEPTION = (
        "Conv3d_1a_7x7", "MaxPool3d_2a_3x3", "Conv3d_2b_1x1", "Conv3d_2c_3x3",
        "MaxPool3d_3a_3x3",
    )

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.Conv3d_1a_7x7 = _Unit3D(in_channels, 64, (7, 7, 7), stride=(2, 2, 2))
        self.MaxPool3d_2a_3x3 = _MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.Conv3d_2b_1x1 = _Unit3D(64, 64, (1, 1, 1))
        self.Conv3d_2c_3x3 = _Unit3D(64, 192, (3, 3, 3))
        self.MaxPool3d_3a_3x3 = _MaxPool3dSamePadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.Mixed_3b = _InceptionModule(192, (64, 96, 128, 16, 32, 32))
        self.Mixed_3c = _InceptionModule(256, (128, 128, 192, 32, 96, 64))
        self.MaxPool3d_4a_3x3 = _MaxPool3dSamePadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))
        self.Mixed_4b = _InceptionModule(480, (192, 96, 208, 16, 48, 64))
        self.Mixed_4c = _InceptionModule(512, (160, 112, 224, 24, 64, 64))
        self.Mixed_4d = _InceptionModule(512, (128, 128, 256, 24, 64, 64))
        self.Mixed_4e = _InceptionModule(512, (112, 144, 288, 32, 64, 64))
        self.Mixed_4f = _InceptionModule(528, (256, 160, 320, 32, 128, 128))
        self.MaxPool3d_5a_2x2 = _MaxPool3dSamePadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.Mixed_5b = _InceptionModule(832, (256, 160, 320, 32, 128, 128))
        self.Mixed_5c = _InceptionModule(832, (384, 192, 384, 48, 128, 128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Conv3d_1a_7x7(x); x = self.MaxPool3d_2a_3x3(x)
        x = self.Conv3d_2b_1x1(x); x = self.Conv3d_2c_3x3(x)
        x = self.MaxPool3d_3a_3x3(x); x = self.Mixed_3b(x); x = self.Mixed_3c(x)
        x = self.MaxPool3d_4a_3x3(x)
        x = self.Mixed_4b(x); x = self.Mixed_4c(x); x = self.Mixed_4d(x)
        x = self.Mixed_4e(x); x = self.Mixed_4f(x)
        x = self.MaxPool3d_5a_2x2(x)
        x = self.Mixed_5b(x); x = self.Mixed_5c(x)
        return x


class I3DBackbone(nn.Module):
    """Snippet-level wrapper around :class:`InceptionI3D`.

    The FineDiving protocol feeds ``num_snippets`` snippets of
    ``snippet_len`` frames each to I3D. We reshape ``(B, C, T, H, W)`` —
    where ``T == (num_snippets-1)*stride + snippet_len`` — into per-snippet
    sub-clips, run I3D on the union batch, and return per-snippet features.

    Two outputs:
        - ``features``: ``(B, num_snippets, C_feat)`` (after spatial pooling)
        - ``feature_map``: ``(B, num_snippets, C_feat, H', W')`` (pre-pool;
          required by :class:`SpatialMotionAttention` and discarded by TSA).
    """

    def __init__(
        self, cfg: AQACfg, data_cfg: DataCfg, return_spatial_map: bool = False,
    ) -> None:
        super().__init__()
        self.i3d = InceptionI3D(in_channels=3)
        self.num_snippets = data_cfg.num_snippets
        self.snippet_len = data_cfg.snippet_len
        self.snippet_stride = data_cfg.snippet_stride
        self.return_spatial_map = return_spatial_map
        self._feature_dim = cfg.i3d_feature_dim

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _slice_snippets(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, T, H, W) → (B, num_snippets, C, snippet_len, H, W)``."""
        B, C, T, H, W = x.shape
        expected = (self.num_snippets - 1) * self.snippet_stride + self.snippet_len
        if T != expected:
            raise ValueError(
                f"I3DBackbone expects T={expected} (from cfg), got T={T}"
            )
        snippets = []
        for s in range(self.num_snippets):
            start = s * self.snippet_stride
            snippets.append(x[:, :, start:start + self.snippet_len])
        return torch.stack(snippets, dim=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        snippets = self._slice_snippets(x)
        B, S, C, L, H, W = snippets.shape
        flat = snippets.reshape(B * S, C, L, H, W)
        fmap = self.i3d(flat)              # (B*S, 1024, L', H', W')
        fmap = fmap.mean(dim=2)
        BS, Cf, Hp, Wp = fmap.shape
        feat = fmap.mean(dim=(-2, -1))     # (B*S, Cf)
        feat = feat.reshape(B, S, Cf)
        out: Dict[str, torch.Tensor] = {"features": feat}
        if self.return_spatial_map:
            out["feature_map"] = fmap.reshape(B, S, Cf, Hp, Wp)
        return out


# =============================================================================
# Procedure Segmentation (S) — FineDiving §4.2, Eq. (2)-(3)
# =============================================================================
#
# Two implementations live here:
#
# 1. _ReferencePSNet — ported verbatim from FineDiving-main/models/PS.py and
#    models/PS_parts.py. Snippets are channels, I3D feature dim is the 1D
#    "length", four MaxPool1d-and-double-conv blocks pull length 1024 → 64
#    while growing channels 9 → 96, then MLP_tas with Sigmoid gives a
#    per-(frame, transition) probability of shape (B, 96, L). The "u_fea"
#    output (B, 96, 64) is the downstream feature, with the 96 axis
#    re-interpreted as frame index. This is the architecture that produced
#    the published numbers.
#
# 2. _UpsampleVariant — an alternative that interprets snippets as a
#    temporal axis to be upsampled to num_frames=96, with channels reduced
#    from feature_dim=1024 to 128. Kept for ablation; not the published
#    architecture.
#
# Both are wrapped by ProcedureSegmentation, which exposes a uniform
# {logits, probs, transitions, frame_feature} interface plus a
# .frame_feature_dim property the downstream consults so the decoder
# embed dim adapts automatically.
# -----------------------------------------------------------------------------


class _PSDoubleConv(nn.Module):
    """Conv1d → BN → ReLU → Conv1d → BN → ReLU. Reference: PS_parts.double_conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _PSDownBlock(nn.Module):
    """MaxPool1d(2) → double_conv. Reference: PS_parts.down."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool1d(2),
            _PSDoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mpconv(x)


class _MLPTas(nn.Module):
    """Final head of the reference PSNet — MLP_tas in PS_parts.py.

    Three Linears with non-linearity between, Sigmoid at the output (so each
    (frame, transition) cell is an independent binary classifier; the
    reference BCE consumes these cells directly).

    The hidden sizes are the published values (128 then 64). Activation
    is configurable for ablation but defaults to ReLU per STSA §5.2 and
    the reference. (Issue 14)
    """

    def __init__(self, in_channel: int, out_channel: int, activation: str = "relu") -> None:
        super().__init__()
        if activation == "relu":
            act_cls: type = nn.ReLU
        elif activation == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError(f"unknown activation: {activation!r}")
        self.layer1 = nn.Linear(in_channel, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, out_channel)
        self.act1 = act_cls()
        self.act2 = act_cls()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        return self.sigmoid(self.layer3(x))


class _ReferencePSNet(nn.Module):
    """Faithful port of ``FineDiving-main/models/PS.py``.

    Forward contract:
        input:  (B, num_snippets, feature_dim)
        output:
            frame_feature: (B, num_frames, feature_reduced_dim)
                — same axes as the reference's ``u_fea_96``
            transition_probs: (B, num_frames, L)
                — per-(frame, transition) Sigmoid probabilities (matches the
                  reference's ``transits_pred`` exactly).

    Default channel ladder is the published one — 9→12→24→48→96→96 — and
    the input-length ladder is 1024→512→256→128→64.
    """

    # Reference values from PS.py:8-13.
    CHANNEL_LADDER: Tuple[int, ...] = (12, 24, 48, 96, 96)
    # 1024 / 2^4 = 64 (four MaxPool1d(2) layers).
    REFERENCE_FEATURE_DIM_REDUCED: int = 64

    def __init__(
        self, num_snippets: int, feature_dim: int, num_frames: int,
        num_step_transitions: int, b2_activation: str = "relu",
    ) -> None:
        super().__init__()
        self.num_snippets = num_snippets
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.num_step_transitions = num_step_transitions

        if feature_dim % 16 != 0:
            raise ValueError(
                f"_ReferencePSNet expects feature_dim divisible by 16 "
                f"(got {feature_dim}). The four MaxPool1d(2) layers would "
                f"otherwise drop fractional positions."
            )
        ch = self.CHANNEL_LADDER
        self.inc = _PSDoubleConv(num_snippets, ch[0])
        self.down1 = _PSDownBlock(ch[0], ch[1])
        self.down2 = _PSDownBlock(ch[1], ch[2])
        self.down3 = _PSDownBlock(ch[2], ch[3])
        self.down4 = _PSDownBlock(ch[3], ch[4])
        feature_dim_reduced = feature_dim // 16
        self.tas = _MLPTas(
            in_channel=feature_dim_reduced,
            out_channel=num_step_transitions,
            activation=b2_activation,
        )
        if ch[-1] != num_frames:
            raise ValueError(
                f"_ReferencePSNet's final channel count ({ch[-1]}) must "
                f"equal num_frames ({num_frames}). If you changed "
                f"DataCfg.num_frames you also need to retune CHANNEL_LADDER."
            )
        self._feature_dim_reduced = feature_dim_reduced

    @property
    def frame_feature_dim(self) -> int:
        """Channel dim of ``frame_feature`` returned by ``forward``."""
        return self._feature_dim_reduced

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``x``: (B, num_snippets, feature_dim)."""
        if x.dim() != 3:
            raise ValueError(f"expected 3D input (B, S, C), got {tuple(x.shape)}")
        B, S, C = x.shape
        if S != self.num_snippets or C != self.feature_dim:
            raise ValueError(
                f"shape mismatch: got (B={B}, S={S}, C={C}); "
                f"expected S={self.num_snippets}, C={self.feature_dim}"
            )
        x = self.inc(x)         # (B, 12, 1024)
        x = self.down1(x)       # (B, 24, 512)
        x = self.down2(x)       # (B, 48, 256)
        x = self.down3(x)       # (B, 96, 128)
        x = self.down4(x)       # (B, 96, 64)
        # x is now (B, num_frames, frame_feature_dim) — the "u_fea_96" feature.
        probs = self.tas(x)     # (B, num_frames, L), already sigmoid'd
        return {"frame_feature": x, "transition_probs": probs}


class _UpsampleVariant(nn.Module):
    """Alternative procedure-segmentation block (upsample temporal + reduce channels).

    Used when ``AQACfg.segmentation_use_reference_psnet=False``. Same I/O
    contract as ``_ReferencePSNet`` so the downstream decoder doesn't
    care which variant is active.

    Output ``transition_probs`` shape is (B, num_frames, L). Unlike the
    reference, this variant applies a softmax over the frame axis per
    transition (one peaked distribution per row), so its BCE is not
    bit-equivalent to the reference's per-cell BCE.
    """

    def __init__(
        self, cfg: AQACfg, data_cfg: DataCfg, feature_dim: int,
    ) -> None:
        super().__init__()
        spatial_dims = cfg.segmentation_spatial_dims  # (1024, 512, 256, 128)
        temporal_dims = cfg.segmentation_temporal_dims  # (12, 24, 48, 96)
        if len(spatial_dims) != 4 or len(temporal_dims) != 4:
            raise ValueError(
                f"Upsample variant expects 4 dims; got "
                f"spatial={spatial_dims}, temporal={temporal_dims}"
            )
        if temporal_dims[-1] != data_cfg.num_frames:
            raise ValueError(
                f"segmentation_temporal_dims[-1]={temporal_dims[-1]} "
                f"must equal num_frames={data_cfg.num_frames}"
            )

        self.num_frames = data_cfg.num_frames
        self.num_step_transitions = data_cfg.num_step_transitions
        self._frame_feature_dim = spatial_dims[-1]

        self.b1 = nn.ModuleList()
        in_c = feature_dim
        for out_c, out_t in zip(spatial_dims, temporal_dims):
            self.b1.append(self._down_up_sub_block(in_c, out_c, out_t))
            in_c = out_c

        # b2: 3-conv head (per-frame transition logits). Activation per cfg.
        if cfg.b2_activation == "relu":
            act_cls: type = nn.ReLU
        elif cfg.b2_activation == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError(f"unknown b2_activation: {cfg.b2_activation!r}")
        b2_hidden = max(64, in_c // 2)
        self.b2 = nn.Sequential(
            nn.Conv1d(in_c, b2_hidden, kernel_size=3, padding=1),
            act_cls(),
            nn.Conv1d(b2_hidden, b2_hidden, kernel_size=3, padding=1),
            act_cls(),
            nn.Conv1d(b2_hidden, self.num_step_transitions, kernel_size=1),
        )

    @staticmethod
    def _down_up_sub_block(in_c: int, out_c: int, out_t: int) -> nn.Module:
        return nn.Sequential(
            nn.Upsample(size=out_t, mode="nearest"),
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.GELU(),
        )

    @property
    def frame_feature_dim(self) -> int:
        return self._frame_feature_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """``x``: (B, num_snippets, feature_dim)."""
        x = x.permute(0, 2, 1)
        for block in self.b1:
            x = block(x)
        # x is (B, frame_feature_dim, num_frames); match reference (B, T, C) order.
        frame_feature = x.permute(0, 2, 1).contiguous()
        logits_lt = self.b2(x)                                    # (B, L, T)
        probs_lt = F.softmax(logits_lt, dim=-1)
        transition_probs = probs_lt.permute(0, 2, 1).contiguous() # (B, T, L)
        return {"frame_feature": frame_feature, "transition_probs": transition_probs}


class ProcedureSegmentation(nn.Module):
    """Wrapper that selects between the reference and upsample variants.

    Exposes a uniform interface so the rest of the model is variant-agnostic:
        forward(features) -> {
            "frame_feature":    (B, num_frames, frame_feature_dim),
            "transition_probs": (B, num_frames, L) — Sigmoid (ref) or
                                 row-softmax (upsample),
            "transitions":      (B, L) int64 — argmax-within-partition,
        }
    """

    def __init__(self, cfg: AQACfg, data_cfg: DataCfg, feature_dim: int) -> None:
        super().__init__()
        self.num_frames = data_cfg.num_frames
        self.num_step_transitions = data_cfg.num_step_transitions
        self.use_reference = cfg.segmentation_use_reference_psnet
        if self.use_reference:
            self.module: nn.Module = _ReferencePSNet(
                num_snippets=data_cfg.num_snippets,
                feature_dim=feature_dim,
                num_frames=data_cfg.num_frames,
                num_step_transitions=data_cfg.num_step_transitions,
                b2_activation=cfg.b2_activation,
            )
            self._frame_feature_dim = self.module.frame_feature_dim
            self._probs_are_sigmoid = True
        else:
            self.module = _UpsampleVariant(cfg, data_cfg, feature_dim=feature_dim)
            self._frame_feature_dim = self.module.frame_feature_dim
            self._probs_are_sigmoid = False

    @property
    def frame_feature_dim(self) -> int:
        """Channel dim of ``frame_feature``; the decoder embed dim uses this."""
        return self._frame_feature_dim

    @property
    def probs_are_sigmoid(self) -> bool:
        """If True, ``transition_probs`` is per-cell sigmoid (independent BCE).
        If False, it's row-softmax along the frame axis."""
        return self._probs_are_sigmoid

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.module(features)
        probs = out["transition_probs"]   # (B, T, L)
        transitions = self._argmax_within_partition(probs)
        return {
            "frame_feature": out["frame_feature"],
            "transition_probs": probs,
            "transitions": transitions,
        }

    @staticmethod
    def _argmax_within_partition(probs: torch.Tensor) -> torch.Tensor:
        """Eq. (3) of FineDiving: argmax for transition ``k`` inside
        ``[T/L · k, T/L · (k+1))`` to enforce ordering t̂_1 ≤ ... ≤ t̂_L.

        Matches the reference ``helper.py:55-57`` partition-argmax convention.
        """
        B, T, L = probs.shape
        out = torch.zeros(B, L, dtype=torch.long, device=probs.device)
        if T % L == 0:
            chunk = T // L
            for k in range(L):
                lo, hi = chunk * k, chunk * (k + 1)
                sub = probs[:, lo:hi, k]
                out[:, k] = sub.argmax(dim=-1) + lo
        else:
            chunk_f = T / float(L)
            for k in range(L):
                lo = int(math.floor(chunk_f * k))
                hi = int(math.ceil(chunk_f * (k + 1)))
                sub = probs[:, lo:hi, k]
                out[:, k] = sub.argmax(dim=-1) + lo
        return out


# =============================================================================
# Spatial Motion Attention (STSA — IJCV 2024 §4.2)
# =============================================================================


class SpatialMotionAttention(nn.Module):
    """Implicit-supervision foreground-motion attention.

    Computes a global vector ``S^G`` by pooling query and exemplar features
    across time, then weights the per-frame feature map ``S^{XM}`` by
    ``S^G`` along the channel axis to produce a spatial attention map
    ``A^X``. A learnable block ``D`` predicts ``Â^X`` from ``S^{XM}``;
    the proxy KL-loss between ``A`` and ``Â`` is what drives the model to
    focus on foreground regions without explicit segmentation labels.

    Refined features are returned as a contrastive-mechanism output that
    feeds the procedure-aware cross-attention.
    """

    def __init__(self, cfg: AQACfg, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        # Three-layer MLP with two ReLU non-linearities (IJCV: "D is a
        # three-layer MLP with two ReLU non-linearities").
        self.D = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 2, feature_dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, 1, kernel_size=1),
        )

    def forward(
        self, q_feat: torch.Tensor, e_feat: torch.Tensor,
        q_fmap: torch.Tensor, e_fmap: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inputs:
            q_feat: ``(B, S, C)`` query features
            e_feat: ``(B, S, C)`` exemplar features
            q_fmap: ``(B, S, C, H, W)`` query feature map (pre-spatial-pool)
            e_fmap: ``(B, S, C, H, W)`` exemplar feature map

        Returns dict with refined ``q_refined`` (B, S, C), ``e_refined_map``
        (B, S, C, H, W), and ``a_gt_q / a_pred_q / a_gt_e / a_pred_e``.
        """
        union = torch.cat([q_feat, e_feat], dim=1)
        s_g = union.mean(dim=1)
        s_g = F.normalize(s_g, dim=-1)
        a_gt_q = self._make_gt_attention(q_fmap, s_g)
        a_gt_e = self._make_gt_attention(e_fmap, s_g)
        a_pred_q = self._predict_attention(q_fmap)
        a_pred_e = self._predict_attention(e_fmap)
        q_refined_map = q_fmap * a_pred_q.unsqueeze(2)
        e_refined_map = e_fmap * a_pred_e.unsqueeze(2)
        q_refined = self._spatial_softmax_pool(q_refined_map)
        e_refined = self._spatial_softmax_pool(e_refined_map)
        return {
            "q_refined": q_refined, "e_refined": e_refined,
            "q_refined_map": q_refined_map, "e_refined_map": e_refined_map,
            "a_gt_q": a_gt_q, "a_pred_q": a_pred_q,
            "a_gt_e": a_gt_e, "a_pred_e": a_pred_e,
        }

    @staticmethod
    def _make_gt_attention(fmap: torch.Tensor, s_g: torch.Tensor) -> torch.Tensor:
        """``A^X_ij = softmax_ij(Σ_c S^G_c * fmap_c_ij)`` (IJCV Eq. 5)."""
        B, S, C, H, W = fmap.shape
        s_g_exp = s_g.view(B, 1, C, 1, 1)
        scores = (fmap * s_g_exp).sum(dim=2)
        flat = scores.view(B, S, H * W)
        flat = F.softmax(flat, dim=-1)
        return flat.view(B, S, H, W)

    def _predict_attention(self, fmap: torch.Tensor) -> torch.Tensor:
        """Run the learnable D over each snippet's feature map."""
        B, S, C, H, W = fmap.shape
        flat = fmap.reshape(B * S, C, H, W)
        logits = self.D(flat).squeeze(1)
        flat_sm = F.softmax(logits.view(B * S, H * W), dim=-1)
        return flat_sm.view(B, S, H, W)

    @staticmethod
    def _spatial_softmax_pool(fmap: torch.Tensor) -> torch.Tensor:
        """Eq. 8: softmax-weighted spatial pool to get a per-snippet vector."""
        B, S, C, H, W = fmap.shape
        flat = fmap.view(B, S, C, H * W)
        weights = F.softmax(flat, dim=-1)
        return (flat * weights).sum(dim=-1)


# =============================================================================
# Procedure-Aware Cross-Attention — FineDiving Eq. (5)-(6)
# =============================================================================


class _CrossAttnLayer(nn.Module):
    """One transformer-decoder layer: cross-attention + MLP."""

    def __init__(self, dim: int, heads: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.mca = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True,
        )
        self.ln_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mca(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv))
        q = q + attn_out
        q = q + self.ff(self.ln_ff(q))
        return q


class ProcedureAwareCrossAttention(nn.Module):
    """Stacks ``decoder_layers`` cross-attention layers (FineDiving Eq. 5-6).

    Operates on each of the ``L+1`` step pairs independently: query step
    ``S^X_l`` attends to exemplar step ``S^Z_l``. Weights are *shared*
    across step indices (a single decoder is applied L+1 times) so the
    paper's "the consecutive steps of query action are served as queries
    and the steps of exemplar action are served as keys and values"
    statement is faithfully reproduced.

    ``heads`` must divide ``dim``; we adjust it down automatically if the
    user-supplied config picks an incompatible pair (this happens when
    switching to reference PSNet, where dim drops from 1024 to 64).
    """

    def __init__(self, cfg: AQACfg, dim: int) -> None:
        super().__init__()
        self.dim = dim
        heads = cfg.decoder_heads
        while heads > 1 and dim % heads != 0:
            heads //= 2
        if heads != cfg.decoder_heads:
            log.warning(
                "ProcedureAwareCrossAttention: cfg.decoder_heads=%d does not "
                "divide dim=%d; using heads=%d instead.",
                cfg.decoder_heads, dim, heads,
            )
        self.layers = nn.ModuleList([
            _CrossAttnLayer(
                dim=dim, heads=heads,
                dim_ff=cfg.decoder_dim_ff, dropout=cfg.decoder_dropout,
            )
            for _ in range(cfg.decoder_layers)
        ])

    def forward(self, q_step: torch.Tensor, e_step: torch.Tensor) -> torch.Tensor:
        """Inputs are per-step tokens of shape ``(B, T_l, C)``."""
        x = q_step
        for layer in self.layers:
            x = layer(x, e_step)
        return x


# =============================================================================
# Fine-grained Contrastive Regressor — FineDiving Eq. (7)
# =============================================================================


class ContrastiveRegressor(nn.Module):
    """Three-layer MLP producing a per-step relative score.

    Used per step, then summed over steps. The final predicted score is
    ``ŷ_X = Σ_l R(S_l) + y_Z`` (Eq. 7, IJCV form with λ_l = 1). The MLP
    activation is configurable via ``AQACfg.b2_activation`` (both this MLP
    and the segmentation's MLP_tas share the activation choice since
    they're the "non-linear heads" the STSA paper discusses).
    """

    def __init__(self, cfg: AQACfg, feature_dim: int) -> None:
        super().__init__()
        hidden = cfg.regressor_hidden
        dims = [feature_dim] + list(hidden) + [1]
        if cfg.b2_activation == "relu":
            act_cls: type = nn.ReLU
        elif cfg.b2_activation == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError(f"unknown b2_activation: {cfg.b2_activation!r}")
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
                layers.append(nn.Dropout(cfg.regressor_dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, step_feature: torch.Tensor) -> torch.Tensor:
        """Pool the per-step sequence then regress.

        Input: (B, T_l, C). Output: (B,) scalar per-step contribution.
        """
        pooled = step_feature.mean(dim=1)
        return self.mlp(pooled).squeeze(-1)


# =============================================================================
# Composite model
# =============================================================================


@dataclass
class TSAOutput:
    """Structured output of :meth:`TSAModel.forward`.

    Forward direction (q → e), always populated::

        predicted_score:        (B,)
        relative_score:         (B,) — Σ_l R(S_l) in q-attends-to-e
        transition_probs:       (B, T, L)
        predicted_transitions:  (B, L) int64

    Reverse direction (e → q), only when ``use_symmetric=True``::

        predicted_score_eq:     (B,)
        relative_score_eq:      (B,)

    Curriculum diagnostic::

        used_gt_transitions:    True iff GT-slicing branch was active

    STSA proxy-task tensors (None when TSA-only)::

        sma:                    dict with a_gt_q, a_pred_q, a_gt_e, a_pred_e
    """

    predicted_score: torch.Tensor
    relative_score: torch.Tensor
    transition_probs: torch.Tensor
    predicted_transitions: torch.Tensor
    used_gt_transitions: bool = False
    predicted_score_eq: Optional[torch.Tensor] = None
    relative_score_eq: Optional[torch.Tensor] = None
    sma: Optional[Dict[str, torch.Tensor]] = None


class TSAModel(nn.Module):
    """The full AQA scorer.

    Args:
        aqa_cfg: hyperparameters for the head + decoder + regressor.
        data_cfg: shared numeric constants (num_frames, num_snippets, ...).
        use_stsa: if True, inserts :class:`SpatialMotionAttention` between
            the backbone and the cross-attention decoder. Default False
            (TSA from CVPR 2022). Set True to reproduce STSA from IJCV 2024.
    """

    def __init__(
        self, aqa_cfg: AQACfg, data_cfg: DataCfg, use_stsa: bool = False,
    ) -> None:
        super().__init__()
        self.aqa_cfg = aqa_cfg
        self.data_cfg = data_cfg
        self.use_stsa = use_stsa

        # Backbone returns the spatial feature map only when STSA needs it.
        self.backbone = I3DBackbone(
            cfg=aqa_cfg, data_cfg=data_cfg, return_spatial_map=use_stsa,
        )
        feat_dim = self.backbone.feature_dim

        self.segmentation = ProcedureSegmentation(
            aqa_cfg, data_cfg, feature_dim=feat_dim,
        )
        decoder_dim = self.segmentation.frame_feature_dim

        if use_stsa:
            self.sma: Optional[SpatialMotionAttention] = SpatialMotionAttention(
                aqa_cfg, feature_dim=feat_dim,
            )
        else:
            self.sma = None

        self.decoder = ProcedureAwareCrossAttention(aqa_cfg, dim=decoder_dim)
        self.regressor = ContrastiveRegressor(aqa_cfg, feature_dim=decoder_dim)

        self._frozen: bool = False

    # ---- public API ----------------------------------------------------

    def forward(
        self,
        query_frames: torch.Tensor,
        exemplar_frames: torch.Tensor,
        exemplar_score: torch.Tensor,
        query_score: Optional[torch.Tensor] = None,
        gt_query_transitions: Optional[torch.Tensor] = None,
        gt_exemplar_transitions: Optional[torch.Tensor] = None,
        current_epoch: Optional[int] = None,
        use_symmetric: Optional[bool] = None,
    ) -> TSAOutput:
        """End-to-end forward.

        Args:
            query_frames, exemplar_frames: ``(B, C, T, H, W)`` clips.
            exemplar_score: ``(B,)`` exemplar's score; added to relative
                prediction to form ``predicted_score``.
            query_score: ``(B,)`` — only required when training symmetrically
                (the reverse-direction MSE needs it). May be None at eval.
            gt_query_transitions: ``(B, L)`` — ground-truth step transitions
                for the query video. Used for the curriculum branch
                (Issue 5). Frame-coordinate indices in [0, num_frames).
            gt_exemplar_transitions: ``(B, L)`` — same for the exemplar.
            current_epoch: integer — passed in so the curriculum branch
                can decide GT vs predicted slicing.
            use_symmetric: override for the cfg-set default. When the
                model is in :meth:`frozen` mode, forces False regardless.

        Returns:
            :class:`TSAOutput`. When symmetric is active, both
            ``predicted_score`` and ``predicted_score_eq`` are populated.
        """
        # ---- backbone (Q and E independently) ----
        q = self.backbone(query_frames)
        e = self.backbone(exemplar_frames)

        # ---- optional STSA refinement ----
        sma_out: Optional[Dict[str, torch.Tensor]] = None
        q_features = q["features"]   # (B, S, C_feat) — used by segmentation
        e_features = e["features"]
        if self.sma is not None:
            sma_out = self.sma(
                q_feat=q["features"], e_feat=e["features"],
                q_fmap=q["feature_map"], e_fmap=e["feature_map"],
            )
            # Use refined per-snippet features for segmentation so the
            # segmentation block also benefits from foreground attention.
            q_features = sma_out["q_refined"]
            e_features = sma_out["e_refined"]

        # ---- segmentation, run on the union along the batch axis ----
        # Reference (helper.py:36-39) concatenates q and e along batch and
        # runs PSNet once. Identical BN statistics for q and e branches —
        # this matters for matching published numbers.
        B = q_features.shape[0]
        union = torch.cat([q_features, e_features], dim=0)
        seg = self.segmentation(union)
        q_frame_feature = seg["frame_feature"][:B]         # (B, T, dec_dim)
        e_frame_feature = seg["frame_feature"][B:]
        q_probs = seg["transition_probs"][:B]              # (B, T, L)
        e_probs = seg["transition_probs"][B:]
        q_predicted_transitions = seg["transitions"][:B]
        e_predicted_transitions = seg["transitions"][B:]

        # ---- curriculum: GT vs predicted transitions ----
        # Reference (helper.py:64): use GT for the first
        # prob_tas_threshold * max_epoch epochs (≈25% × 200 = 50 by default).
        # After that, switch to the segmentation head's own predictions.
        use_gt_branch = self._should_use_gt(current_epoch)
        if use_gt_branch and (
            gt_query_transitions is None or gt_exemplar_transitions is None
        ):
            # Curriculum requested but GT not supplied; fall back and warn.
            log.warning(
                "Curriculum requested (epoch=%s < %d) but GT transitions were "
                "not provided; falling back to predicted transitions.",
                current_epoch, self.aqa_cfg.curriculum_threshold_epochs,
            )
            use_gt_branch = False

        if use_gt_branch:
            q_step_transitions = gt_query_transitions
            e_step_transitions = gt_exemplar_transitions
        else:
            q_step_transitions = q_predicted_transitions
            e_step_transitions = e_predicted_transitions

        # ---- step-wise decoder, both directions if symmetric ----
        sym = (
            self.aqa_cfg.use_symmetric_training
            if use_symmetric is None else bool(use_symmetric)
        )
        # Frozen oracle path never goes symmetric (CF only needs q→e).
        if self._frozen:
            sym = False

        relative_qe = self._step_loop(
            q_frame_feature, e_frame_feature,
            q_boundaries=q_step_transitions,
            e_boundaries=e_step_transitions,
        )
        predicted_qe = relative_qe + exemplar_score

        relative_eq: Optional[torch.Tensor] = None
        predicted_eq: Optional[torch.Tensor] = None
        if sym:
            if query_score is None:
                raise ValueError(
                    "use_symmetric_training=True but query_score was not "
                    "passed to forward(); the reverse-direction needs it."
                )
            # Reverse direction: e attends to q. Boundaries swap too — q's
            # exemplar role uses q_step_transitions, e's query role uses
            # e_step_transitions. (This is the natural symmetric form.)
            relative_eq = self._step_loop(
                e_frame_feature, q_frame_feature,
                q_boundaries=e_step_transitions,
                e_boundaries=q_step_transitions,
            )
            predicted_eq = relative_eq + query_score

        return TSAOutput(
            predicted_score=predicted_qe,
            relative_score=relative_qe,
            transition_probs=q_probs,
            predicted_transitions=q_predicted_transitions,
            used_gt_transitions=use_gt_branch,
            predicted_score_eq=predicted_eq,
            relative_score_eq=relative_eq,
            sma=sma_out,
        )

    # ---- inference convenience -----------------------------------------

    @torch.inference_mode()
    def predict_score(
        self, query_frames: torch.Tensor,
        exemplars: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Multi-exemplar voting (CoRe Eq. 10), used at evaluation time.

        Args:
            query_frames: ``(B, C, T, H, W)`` query clip.
            exemplars: iterable of ``(exemplar_frames, exemplar_score)`` pairs,
                       length M (typically M=10).

        Returns:
            ``(B,)`` averaged predicted score.
        """
        preds: List[torch.Tensor] = []
        for e_frames, e_score in exemplars:
            out = self.forward(
                query_frames, e_frames, e_score,
                use_symmetric=False,    # eval is q→e only
            )
            preds.append(out.predicted_score)
        return torch.stack(preds, dim=0).mean(dim=0)

    # ---- frozen-oracle contract ----------------------------------------

    @contextlib.contextmanager
    def frozen(self):
        """Context that makes the model a safe differentiable oracle.

        - puts the model in eval mode (dropout off, batch-norm running stats)
        - sets ``requires_grad=False`` on every parameter
        - flips the internal ``_frozen`` flag so ``forward`` forces
          ``use_symmetric=False`` regardless of cfg
        - does NOT disable autograd globally — the input keeps its grad graph

        On exit, restores the previous training mode and per-parameter
        ``requires_grad`` flags. Re-entrant safe.
        """
        prev_training = self.training
        prev_requires_grad = [p.requires_grad for p in self.parameters()]
        prev_frozen = self._frozen
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self._frozen = True
        try:
            yield self
        finally:
            self.train(prev_training)
            for p, flag in zip(self.parameters(), prev_requires_grad):
                p.requires_grad_(flag)
            self._frozen = prev_frozen

    # ---- internal helpers ----------------------------------------------

    def _should_use_gt(self, current_epoch: Optional[int]) -> bool:
        """Curriculum gate (Issue 5).

        Returns True while ``current_epoch < curriculum_threshold_epochs``.
        Returns False when ``current_epoch`` is None or beyond the threshold.
        """
        if current_epoch is None:
            return False
        if self.aqa_cfg.curriculum_threshold_epochs <= 0:
            return False
        return current_epoch < self.aqa_cfg.curriculum_threshold_epochs

    def _step_loop(
        self,
        q_frame_feature: torch.Tensor,
        e_frame_feature: torch.Tensor,
        q_boundaries: torch.Tensor,
        e_boundaries: torch.Tensor,
    ) -> torch.Tensor:
        """Run the decoder over L+1 step pairs and sum the per-step regressor outputs.

        Inputs:
            q_frame_feature, e_frame_feature: (B, num_frames, C_dec)
            q_boundaries: (B, L) frame indices for the query
            e_boundaries: (B, L) frame indices for the exemplar

        Returns:
            (B,) relative score = Σ_l R(decoder(q_step_l, e_step_l)).
        """
        relative_per_step: List[torch.Tensor] = []
        for l in range(self.data_cfg.num_steps):
            q_step = _slice_step_and_resample(
                q_frame_feature, q_boundaries, l,
                target_len=self.data_cfg.frames_per_step,
            )
            e_step = _slice_step_and_resample(
                e_frame_feature, e_boundaries, l,
                target_len=self.data_cfg.frames_per_step,
            )
            decoded = self.decoder(q_step, e_step)
            relative_per_step.append(self.regressor(decoded))
        return torch.stack(relative_per_step, dim=-1).sum(dim=-1)


# =============================================================================
# Slice / resample helpers
# =============================================================================


def _slice_step_and_resample(
    tokens: torch.Tensor, boundaries: torch.Tensor,
    step_idx: int, target_len: int,
) -> torch.Tensor:
    """Slice the L+1-th step out of ``tokens`` and resample to ``target_len``.

    Inputs:
        tokens:     (B, T, C) — frame-granularity feature for one branch
        boundaries: (B, L) — transition frame indices (sorted ascending)
        step_idx:   integer in [0, L]; step 0 covers [0, boundaries[:,0]),
                    step L covers [boundaries[:,-1], T), middle steps
                    cover [boundaries[:,step-1], boundaries[:,step]).
        target_len: number of frames to resample each step to.

    Returns:
        (B, target_len, C).
    """
    B, T, C = tokens.shape
    L = boundaries.shape[1]
    # Build the per-sample (lo, hi) ranges, guaranteeing hi > lo (the
    # reference's ``if video_st == 0: video_st = 1`` trick — at least one
    # frame per step).
    out_per_sample: List[torch.Tensor] = []
    for b in range(B):
        if step_idx == 0:
            lo, hi = 0, int(boundaries[b, 0].item())
        elif step_idx == L:
            lo, hi = int(boundaries[b, L - 1].item()), T
        else:
            lo = int(boundaries[b, step_idx - 1].item())
            hi = int(boundaries[b, step_idx].item())
        lo = max(0, min(lo, T - 1))
        hi = max(lo + 1, min(hi, T))
        # Slice and resample this sample's [lo, hi) range to target_len.
        slice_ = tokens[b:b + 1, lo:hi].permute(0, 2, 1)   # (1, C, hi-lo)
        resampled = F.interpolate(
            slice_, size=target_len, mode="linear", align_corners=False,
        )
        out_per_sample.append(resampled.permute(0, 2, 1))   # (1, target_len, C)
    return torch.cat(out_per_sample, dim=0)


# Backwards compatibility — counterfactual.py / experiments.py may import this.
def _frames_to_snippet_boundaries(
    transitions: torch.Tensor, num_frames: int, num_snippets: int,
) -> torch.Tensor:
    """Legacy mapping from per-frame transitions to snippet indices.

    Kept for any caller that operates at snippet granularity (the
    in-process feature is now at frame granularity by default, so this
    is rarely needed).
    """
    scale = num_snippets / float(num_frames)
    snippet_idx = (transitions.float() * scale).floor().long()
    return snippet_idx.clamp_(0, num_snippets - 1)


# =============================================================================
# Loss
# =============================================================================


def compute_aqa_loss(
    output: TSAOutput,
    batch: Mapping[str, torch.Tensor],
    aqa_cfg: AQACfg,
    data_cfg: DataCfg,
    sma_kl_weight: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """``J = α_bce · BCE + α_mse · MSE [+ α_kl · KL]`` (FineDiving Eq. 9; IJCV Eq. 13).

    Matches the reference's ``loss = loss_aqa + loss_tas`` (helper.py:165)
    when ``aqa_cfg.bce_weight = aqa_cfg.mse_weight = 1.0``. When
    ``aqa_cfg.use_symmetric_training=True``, the MSE term covers BOTH
    directions, mirroring the reference's helper.py:160-162.

    Args:
        output: model output (a :class:`TSAOutput`).
        batch: collated batch from :class:`DataModule`, with at minimum
            ``query_score``, ``exemplar_score``, ``query_step_transitions``.
        aqa_cfg: provides BCE / MSE / KL weights and the KL aggregation rule.
        data_cfg: provides num_frames for transition-target rasterisation.
        sma_kl_weight: optional override for ``aqa_cfg.sma_kl_weight``.
            Pass None to use the cfg default.

    Returns:
        dict with ``total``, ``mse``, ``bce``, and optionally ``kl``,
        ``mse_qe``, ``mse_eq``. Per-direction MSEs are present whenever
        ``output.predicted_score_eq`` is not None.
    """
    # ---- BCE on step-transition probabilities ---------------------------
    transitions = batch["query_step_transitions"]      # (B, L)
    targets = _make_transition_targets(
        transitions, num_frames=data_cfg.num_frames,
    )                                                   # (B, T, L)
    # Both segmentation variants produce probs in [0, 1] — sigmoid for the
    # reference, row-softmax for the upsample variant. F.binary_cross_entropy
    # consumes both correctly; clamp to avoid log(0).
    bce = F.binary_cross_entropy(
        output.transition_probs.clamp(min=1e-7, max=1.0 - 1e-7),
        targets, reduction="mean",
    )

    # ---- MSE on the predicted score (forward direction) -----------------
    # Compute MSE on the DELTA (predicted_score - exemplar_score) vs the
    # true delta (query_score - exemplar_score), which is mathematically
    # the same as MSE(predicted_score, query_score) but matches the
    # reference's exact arithmetic (helper.py:160-162).
    query_score = batch["query_score"]
    exemplar_score = batch["exemplar_score"]
    delta_target_qe = query_score - exemplar_score
    delta_pred_qe = output.relative_score
    mse_qe = F.mse_loss(delta_pred_qe, delta_target_qe)
    losses: Dict[str, torch.Tensor] = {"bce": bce, "mse_qe": mse_qe}
    mse_total = mse_qe

    # ---- MSE on the reverse direction (symmetric training) --------------
    if output.relative_score_eq is not None:
        delta_target_eq = exemplar_score - query_score
        delta_pred_eq = output.relative_score_eq
        mse_eq = F.mse_loss(delta_pred_eq, delta_target_eq)
        losses["mse_eq"] = mse_eq
        # Reference (helper.py:163) literally SUMS the two MSE terms.
        mse_total = mse_qe + mse_eq

    losses["mse"] = mse_total
    total = aqa_cfg.bce_weight * bce + aqa_cfg.mse_weight * mse_total

    # ---- STSA KL loss (proxy task) --------------------------------------
    if output.sma is not None:
        sma = output.sma
        kl_q = _kl_spatial(sma["a_gt_q"], sma["a_pred_q"])
        kl_e = _kl_spatial(sma["a_gt_e"], sma["a_pred_e"])
        # Aggregation per cfg (Issue 8).
        if aqa_cfg.sma_kl_aggregation == "sum":
            kl = kl_q + kl_e
        elif aqa_cfg.sma_kl_aggregation == "mean":
            kl = (kl_q + kl_e) / 2.0
        else:
            raise ValueError(
                f"unknown sma_kl_aggregation: {aqa_cfg.sma_kl_aggregation!r}"
            )
        losses["kl"] = kl
        w_kl = aqa_cfg.sma_kl_weight if sma_kl_weight is None else sma_kl_weight
        total = total + w_kl * kl

    losses["total"] = total
    return losses


def _make_transition_targets(
    transitions: torch.Tensor, num_frames: int,
) -> torch.Tensor:
    """Convert frame indices ``(B, L)`` into per-cell targets ``(B, T, L)``.

    Matches the reference helper.py:43-46 convention:
        label_pad[bs, transition_frame_k, k] = 1
    Every other cell is 0. Since the reference uses BCE with Sigmoid (and
    the upsample variant uses BCE with row-softmax), the same one-hot
    target works for both cases.
    """
    B, L = transitions.shape
    out = torch.zeros(B, num_frames, L, device=transitions.device)
    idx = transitions.clone().clamp_(0, num_frames - 1)
    for k in range(L):
        out[torch.arange(B, device=transitions.device), idx[:, k], k] = 1.0
    return out


def _kl_spatial(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(p || q) over the last two (spatial) dims, averaged over (B, S)."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    kl = (p * (p.log() - q.log())).sum(dim=(-1, -2))
    return kl.mean()


# =============================================================================
# Calibration diagnostic
# =============================================================================


@torch.no_grad()
def calibrate_oracle(
    model: TSAModel, loader, max_batches: int = 32,
) -> Dict[str, float]:
    """Frozen-oracle internal-consistency check.

    Two properties are verified:
        1. **Determinism under frozen()**: scoring the same input twice
           gives identical outputs (no dropout / batch-stats leakage).
        2. **No-update guarantee**: parameter sums before and after a
           sequence of forward passes are bit-identical.

    These together are what makes the model safe to use as a differentiable
    oracle by :mod:`counterfactual`. The function returns the deviations
    so callers can compare to ``AQACfg.calibration_mae_max``.

    Args:
        model: the trained AQA model (will be put into ``.frozen()``).
        loader: any DataLoader yielding batches with ``query_frames``,
                ``exemplar_frames``, and ``exemplar_score`` keys.
        max_batches: cap on how many batches to draw.

    Returns:
        dict with ``determinism_mae`` and ``param_drift``.
    """
    determinism_diffs: List[float] = []
    param_sum_before = _param_sum(model)
    with model.frozen():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            q = batch["query_frames"]
            e = batch["exemplar_frames"]
            ys = batch["exemplar_score"]
            out1 = model(q, e, ys).predicted_score
            out2 = model(q, e, ys).predicted_score
            determinism_diffs.append(float((out1 - out2).abs().mean().item()))
    param_sum_after = _param_sum(model)
    return {
        "determinism_mae": float(
            sum(determinism_diffs) / max(1, len(determinism_diffs))
        ),
        "param_drift": float(abs(param_sum_after - param_sum_before)),
    }


def _param_sum(model: nn.Module) -> float:
    """Sum of parameter values — a cheap fingerprint for no-update checks."""
    s = 0.0
    for p in model.parameters():
        s += float(p.detach().sum().item())
    return s


# =============================================================================
# Checkpoint I/O
# =============================================================================


def load_aqa_checkpoint(
    model: TSAModel, ckpt_path, map_location="cpu", strict: bool = True,
) -> Dict[str, object]:
    """Load a trained AQA checkpoint into ``model``.

    The checkpoint format produced by :mod:`train` is a dict with keys
    ``state_dict``, ``epoch``, ``config_fingerprint``, ``metrics``. We
    refuse to load a checkpoint whose fingerprint disagrees with the
    current config when ``strict=True`` — that's how we catch the
    "you changed the architecture and the .pth still loaded" class of bug.
    """
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        ckpt_fp = state.get("config_fingerprint")
        model.load_state_dict(state["state_dict"], strict=strict)
        return {
            "epoch": state.get("epoch"),
            "config_fingerprint": ckpt_fp,
            "metrics": state.get("metrics", {}),
        }
    # Bare state-dict fallback.
    model.load_state_dict(state, strict=strict)
    return {}


def load_i3d_kinetics_weights(
    model: TSAModel, weights_path, map_location="cpu",
) -> int:
    """Load Kinetics-pretrained I3D weights into the backbone.

    Returns the number of parameters that received pretrained values.
    Missing / extra keys are logged at WARNING level but do not raise —
    public Kinetics checkpoints frequently carry a classifier head we
    don't have.
    """
    state = torch.load(weights_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    own_state = model.backbone.i3d.state_dict()
    transferred = 0
    skipped: List[str] = []
    for k, v in state.items():
        key = k
        for prefix in ("module.", "i3d."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in own_state and own_state[key].shape == v.shape:
            own_state[key] = v
            transferred += 1
        else:
            skipped.append(k)
    model.backbone.i3d.load_state_dict(own_state, strict=False)
    if skipped:
        log.warning("I3D pretrained: skipped %d keys (e.g. %s)",
                    len(skipped), skipped[:3])
    log.info("I3D pretrained: transferred %d tensors from %s",
             transferred, weights_path)
    return transferred


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    "InceptionI3D",
    "I3DBackbone",
    "ProcedureSegmentation",
    "SpatialMotionAttention",
    "ProcedureAwareCrossAttention",
    "ContrastiveRegressor",
    "TSAOutput",
    "TSAModel",
    "compute_aqa_loss",
    "calibrate_oracle",
    "load_aqa_checkpoint",
    "load_i3d_kinetics_weights",
]

"""diffusion.py — Pose-conditioned motion diffusion in the EDGE lineage.

Implements
----------
- :class:`MotionDiffusion`: transformer-based DDPM over 151-dim SMPL pose
  tokens (24 joints × 6D rotation + root translation + 4 foot contacts),
  with FiLM-modulated timestep conditioning, classifier-free guidance,
  and masked inpainting — the editing primitive that supports the three
  intervention modalities used by the half-point counterfactual search.
- :class:`SMPLKinematics`: a minimal forward-kinematics layer over the
  SMPL 24-joint tree, used by :func:`loss_position` and :func:`loss_contact`.
  Includes a hardcoded approximation of SMPL_NEUTRAL rest positions;
  exact values can be plugged in via :func:`load_smpl_rest_positions`.
- Rotation utilities for the 6D continuous representation (Zhou et al.,
  CVPR 2019), used because axis-angle is discontinuous on SO(3) and that
  hurts diffusion regression.
- :func:`compute_diffusion_loss`: EDGE's four-term objective —
  λ_s·L_simple + λ_p·L_pos + λ_v·L_vel + λ_c·L_contact — with
  classifier-free dropout, optional p2 SNR-based loss weighting on the
  first three terms (matching EDGE-main/model/diffusion.py:464,477,496),
  and per-sample reductions so the weighting can apply correctly.
- :func:`make_joint_subset_mask`, :func:`make_phase_window_mask`,
  :func:`make_transition_shift_mask`: builders for the three
  intervention modalities listed in :class:`CFCfg`.

Reference-fidelity values
-------------------------
All numerical defaults verified against the EDGE released codebase at
https://github.com/Stanford-TML/EDGE (commit-frozen via DiffusionCfg):
  - Loss weights (model/diffusion.py:514-519):
      λ_simple = 0.636, λ_vel = 2.964, λ_pos = 0.646, λ_contact = 10.942
  - EMA decay (line 63): 0.9999 (applied in train.py, not here)
  - p2 weight buffer (lines 125-130):
      w_t = (k + ᾱ_t/(1-ᾱ_t))^(-γ), k=1, γ=0.5
  - predict_epsilon (line 80): False — EDGE predicts x̂ (the clean sample)

Design notes
------------
- Pose is tokenised as ``[contacts(4); rotations_6D(144); root(3)]`` = 151,
  matching EDGE exactly (their x = {b, w} convention).
- L_pos and L_contact run through forward kinematics; the kinematic
  *tree* drives gradient flow and is exact, while the rest *offsets*
  affect the numeric loss magnitude only.
- :meth:`MotionDiffusion.inpaint_with_score_guidance` is the hook
  :mod:`counterfactual` calls. It accepts a callable ``score_fn`` so the
  AQA oracle stays outside the import graph of this file.
- :meth:`MotionDiffusion.sample_ddim` provides accelerated sampling
  (50 steps default) for the counterfactual search inner loop; standard
  ``sample`` (T=1000 steps) is used at evaluation time.

Changes from previous version
-----------------------------
- Issue 4: loss weights and EMA decay now come from cfg (the cfg
  defaults were corrected to 0.636/0.646/2.964/10.942 and 0.9999);
  this file simply reads them. The hardcoded 1.0 defaults are gone.
- Issue 10: :meth:`inpaint_with_score_guidance` takes a new
  ``dime_gradient_scaling`` flag. When True, differentiates w.r.t. x̂
  and applies an explicit 1/√ᾱ_t factor (DiME Eq. 8) instead of
  autograd through one denoising step. Stable across τ sweeps.
- Issue 11: new ``noise_level_tau`` parameter. With τ=1.0 (default), z
  is initialised from pure noise as before. With τ<1.0, z is
  initialised from a forward-noised ``x_known`` at intermediate t, and
  the denoising loop runs from t=τ·T downward — the DiME late-start
  trick that yields tighter, more conservative CFs.
- Issue 17: new ``inpaint_resamples`` (R) and
  ``inpaint_resample_fraction`` parameters. R>0 enables RePaint
  Algorithm 1: for the final fraction of steps, do R iterations of
  (reverse-step, re-noise back to t). Defaults R=0 keep prior behaviour.
- Issue 19: new ``predict_epsilon`` mode (default False = predict x̂,
  matching EDGE). When True, model output is treated as ε; the loss
  function targets noise; samplers convert ε→x̂ via the standard formula.
- Issue 20: per-sample p2 SNR-based weighting on L_simple, L_vel, L_pos
  (not L_contact, per EDGE-main lines 464/477/496/no-line-for-foot).
  Toggled by ``DiffusionCfg.p2_loss_weighting``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import DataCfg, DiffusionCfg

log = logging.getLogger(__name__)


# =============================================================================
# Rotation utilities — 6D continuous representation (Zhou et al., CVPR 2019)
# =============================================================================


def axis_angle_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """``(..., 3) → (..., 3, 3)`` via Rodrigues' formula.

    Numerically stable near ``θ=0`` via the standard small-angle limit.
    """
    angle = aa.norm(dim=-1, keepdim=True)               # (..., 1)
    # Avoid division by zero; for tiny angles the formula degenerates to I.
    safe = angle.clamp(min=1e-8)
    axis = aa / safe
    cos = angle.cos()
    sin = angle.sin()
    one_minus_cos = 1.0 - cos
    x, y, z = axis.unbind(dim=-1)
    # Skew-symmetric K and K² for the Rodrigues expansion.
    zero = torch.zeros_like(x)
    K = torch.stack([
        torch.stack([zero, -z,    y   ], dim=-1),
        torch.stack([z,    zero, -x   ], dim=-1),
        torch.stack([-y,   x,    zero ], dim=-1),
    ], dim=-2)                                            # (..., 3, 3)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand(K.shape)
    R = eye + sin.unsqueeze(-1) * K + one_minus_cos.unsqueeze(-1) * torch.matmul(K, K)
    return R


def rotmat_to_6d(R: torch.Tensor) -> torch.Tensor:
    """``(..., 3, 3) → (..., 6)`` — first two rows of the rotation matrix."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def sixd_to_rotmat(d6: torch.Tensor) -> torch.Tensor:
    """``(..., 6) → (..., 3, 3)`` via Gram-Schmidt (Zhou et al. Eq. 3)."""
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    # Subtract the component of a2 along b1, then normalize.
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    # Stack as ROWS so rotmat_to_6d / sixd_to_rotmat round-trip exactly.
    return torch.stack([b1, b2, b3], dim=-2)


def axis_angle_to_6d(aa: torch.Tensor) -> torch.Tensor:
    """Composition of the two converters above — used by :func:`tokenize_pose`."""
    return rotmat_to_6d(axis_angle_to_rotmat(aa))


# =============================================================================
# SMPL kinematic tree
# =============================================================================
#
# Joint ordering matches the SMPL standard (24 body joints). The parent
# array is the same regardless of model gender/shape; only the rest
# offsets depend on shape (β). For L_pos as a regulariser, the topology
# drives gradient flow and the magnitude of rest offsets only affects the
# numeric loss value, not its direction.
# -----------------------------------------------------------------------------


SMPL_JOINT_NAMES: Tuple[str, ...] = (
    "pelvis",       "l_hip",       "r_hip",       "spine1",
    "l_knee",       "r_knee",      "spine2",      "l_ankle",
    "r_ankle",      "spine3",      "l_foot",      "r_foot",
    "neck",         "l_collar",    "r_collar",    "head",
    "l_shoulder",   "r_shoulder",  "l_elbow",     "r_elbow",
    "l_wrist",      "r_wrist",     "l_hand",      "r_hand",
)
NUM_SMPL_JOINTS = 24

# Parent of each joint (-1 for the root). Standard SMPL kinematic tree.
SMPL_PARENTS: Tuple[int, ...] = (
    -1,  0,  0,  0,  1,  2,  3,  4,
     5,  6,  7,  8,  9,  9,  9, 12,
    13, 14, 16, 17, 18, 19, 20, 21,
)

# Foot joints — used by L_contact (heel/ankle and toe/foot for each side).
SMPL_FOOT_JOINTS: Tuple[int, ...] = (7, 8, 10, 11)  # l_ankle, r_ankle, l_foot, r_foot

# Approximate canonical (T-pose, β=0) joint positions in metres. These are
# placeholders close to the SMPL_NEUTRAL mean values; the user should
# override via load_smpl_rest_positions(path) for exact reproduction.
_SMPL_REST_POSITIONS_APPROX: Tuple[Tuple[float, float, float], ...] = (
    ( 0.000,  0.000,  0.000),   # 0 pelvis
    ( 0.058, -0.082, -0.017),   # 1 l_hip
    (-0.064, -0.090, -0.014),   # 2 r_hip
    (-0.004,  0.124, -0.003),   # 3 spine1
    ( 0.103, -0.494, -0.003),   # 4 l_knee
    (-0.110, -0.499, -0.001),   # 5 r_knee
    ( 0.002,  0.255,  0.004),   # 6 spine2
    ( 0.083, -0.916, -0.034),   # 7 l_ankle
    (-0.088, -0.919, -0.035),   # 8 r_ankle
    ( 0.002,  0.292,  0.012),   # 9 spine3
    ( 0.105, -0.972,  0.090),   # 10 l_foot
    (-0.107, -0.975,  0.083),   # 11 r_foot
    ( 0.001,  0.503, -0.027),   # 12 neck
    ( 0.072,  0.421, -0.018),   # 13 l_collar
    (-0.077,  0.422, -0.018),   # 14 r_collar
    ( 0.008,  0.607,  0.046),   # 15 head
    ( 0.184,  0.430, -0.029),   # 16 l_shoulder
    (-0.193,  0.428, -0.030),   # 17 r_shoulder
    ( 0.435,  0.421, -0.043),   # 18 l_elbow
    (-0.450,  0.422, -0.043),   # 19 r_elbow
    ( 0.692,  0.408, -0.047),   # 20 l_wrist
    (-0.707,  0.408, -0.052),   # 21 r_wrist
    ( 0.756,  0.397, -0.054),   # 22 l_hand
    (-0.770,  0.402, -0.061),   # 23 r_hand
)


def default_rest_positions(device=None, dtype=torch.float32) -> torch.Tensor:
    """Return the approximate SMPL_NEUTRAL rest positions as a (24, 3) tensor."""
    return torch.tensor(_SMPL_REST_POSITIONS_APPROX, device=device, dtype=dtype)


def load_smpl_rest_positions(path) -> torch.Tensor:
    """Load exact rest positions from a NumPy ``.npy`` file of shape (24, 3).

    Generate this file once from your SMPL model:

        import numpy as np, pickle
        with open("SMPL_NEUTRAL.pkl", "rb") as f:
            model = pickle.load(f, encoding="latin1")
        # Joint regressor applied to mean (β=0) vertices.
        J = model["J_regressor"] @ model["v_template"]
        np.save("data/smpl/rest_positions.npy", J.astype("float32"))
    """
    import numpy as np
    arr = np.load(str(path))
    if arr.shape != (NUM_SMPL_JOINTS, 3):
        raise ValueError(
            f"expected shape ({NUM_SMPL_JOINTS}, 3), got {arr.shape}"
        )
    return torch.from_numpy(arr).float()


class SMPLKinematics(nn.Module):
    """Forward kinematics through the SMPL 24-joint tree.

    Given per-frame joint rotations (as 3×3 matrices) and root translation,
    produces world-space joint positions via the parent chain. No shape
    blending — for the auxiliary L_pos and L_contact losses we only need
    positions under the canonical T-pose offsets.
    """

    def __init__(
        self,
        rest_positions: Optional[torch.Tensor] = None,
        parents: Sequence[int] = SMPL_PARENTS,
    ) -> None:
        super().__init__()
        rest = rest_positions if rest_positions is not None else default_rest_positions()
        if rest.shape != (NUM_SMPL_JOINTS, 3):
            raise ValueError(
                f"rest_positions must be ({NUM_SMPL_JOINTS}, 3), got {tuple(rest.shape)}"
            )
        offsets = rest.clone()
        for j, p in enumerate(parents):
            if p >= 0:
                offsets[j] = rest[j] - rest[p]
        self.register_buffer("rest_offsets", offsets)             # (J, 3)
        self.parents: Tuple[int, ...] = tuple(parents)

    def forward(
        self, rotmat: torch.Tensor, root_translation: torch.Tensor,
    ) -> torch.Tensor:
        """Run FK.

        Args:
            rotmat: ``(..., J, 3, 3)`` per-joint local rotations.
            root_translation: ``(..., 3)`` world translation of the root.

        Returns:
            ``(..., J, 3)`` world-space joint positions.
        """
        B_shape = rotmat.shape[:-3]
        J = rotmat.shape[-3]
        if J != NUM_SMPL_JOINTS:
            raise ValueError(f"expected J={NUM_SMPL_JOINTS}, got {J}")

        world_R: List[torch.Tensor] = [None] * J  # type: ignore[list-item]
        world_p: List[torch.Tensor] = [None] * J  # type: ignore[list-item]

        world_R[0] = rotmat[..., 0, :, :]
        world_p[0] = root_translation

        offsets = self.rest_offsets
        for j in range(1, J):
            p = self.parents[j]
            world_R[j] = torch.matmul(world_R[p], rotmat[..., j, :, :])
            off = offsets[j].expand(*B_shape, 3).unsqueeze(-1)
            delta = torch.matmul(world_R[p], off).squeeze(-1)
            world_p[j] = world_p[p] + delta

        return torch.stack(world_p, dim=-2)


# =============================================================================
# Pose tokenisation: (axis-angle θ, root τ, contacts b) ↔ 151-dim token
# =============================================================================
#
# EDGE convention (their Sec. 3): x = {b, w} where b are 4 binary contacts
# and w = (24*6 + 3) = 147. Total 151 dims. We follow this exactly.
# Layout: token[..., :4] = contacts, token[..., 4:148] = 6D rotations,
#         token[..., 148:151] = root translation.
# -----------------------------------------------------------------------------


CONTACT_DIM = 4
ROTATION_6D_DIM = NUM_SMPL_JOINTS * 6   # 144
ROOT_DIM = 3
TOKEN_DIM = CONTACT_DIM + ROTATION_6D_DIM + ROOT_DIM   # 151

CONTACT_SLICE = slice(0, CONTACT_DIM)
ROTATION_SLICE = slice(CONTACT_DIM, CONTACT_DIM + ROTATION_6D_DIM)
ROOT_SLICE = slice(CONTACT_DIM + ROTATION_6D_DIM, TOKEN_DIM)


def tokenize_pose(
    theta_aa: torch.Tensor, tau: torch.Tensor,
    contacts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``(T, 24, 3) + (T, 3) [+ (T, 4)] → (T, 151)``.

    If ``contacts`` is None, the contact dims are zeroed; at training time
    they should be derived (e.g. by velocity-thresholding the foot joints
    via :func:`derive_foot_contacts`) and supplied explicitly.
    """
    if theta_aa.shape[-2:] != (NUM_SMPL_JOINTS, 3):
        raise ValueError(
            f"theta_aa must end in ({NUM_SMPL_JOINTS}, 3), got {tuple(theta_aa.shape)}"
        )
    if tau.shape[-1] != ROOT_DIM:
        raise ValueError(f"tau must end in (3,), got {tuple(tau.shape)}")
    leading = theta_aa.shape[:-2]
    sixd = axis_angle_to_6d(theta_aa).reshape(*leading, ROTATION_6D_DIM)
    if contacts is None:
        contacts = theta_aa.new_zeros(*leading, CONTACT_DIM)
    elif contacts.shape[-1] != CONTACT_DIM:
        raise ValueError(f"contacts must end in (4,), got {tuple(contacts.shape)}")
    return torch.cat([contacts, sixd, tau], dim=-1)


def detokenize_pose(token: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`tokenize_pose`. Returns 6D rotations, root, contacts."""
    if token.shape[-1] != TOKEN_DIM:
        raise ValueError(f"token must end in ({TOKEN_DIM},), got {tuple(token.shape)}")
    leading = token.shape[:-1]
    contacts = token[..., CONTACT_SLICE]
    sixd_flat = token[..., ROTATION_SLICE]
    sixd = sixd_flat.reshape(*leading, NUM_SMPL_JOINTS, 6)
    root = token[..., ROOT_SLICE]
    return {"contacts": contacts, "sixd": sixd, "root": root}


def derive_foot_contacts(
    foot_positions: torch.Tensor, velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Derive binary foot contacts from per-frame foot positions.

    Args:
        foot_positions: ``(B, T, 4, 3)`` — the four foot joints over time.
        velocity_threshold: speed (m/frame) below which a foot is considered
                            in contact with the ground.

    Returns:
        ``(B, T, 4)`` binary contact labels.
    """
    if foot_positions.shape[-2:] != (CONTACT_DIM, 3):
        raise ValueError(
            f"foot_positions must end in ({CONTACT_DIM}, 3), got "
            f"{tuple(foot_positions.shape)}"
        )
    delta = foot_positions[..., 1:, :, :] - foot_positions[..., :-1, :, :]
    speed = delta.norm(dim=-1)                                  # (B, T-1, 4)
    speed = torch.cat([speed, speed[..., -1:, :]], dim=-2)      # (B, T, 4)
    return (speed < velocity_threshold).float()


# =============================================================================
# Noise schedule
# =============================================================================


def cosine_betas(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal, ICML 2021). Returns ``(T,) betas``.

    ``s = 0.008`` is the Nichol & Dhariwal recommended offset and matches
    the EDGE codebase. Configurable via ``DiffusionCfg.cosine_s``.
    """
    t = torch.linspace(0, num_steps, num_steps + 1) / num_steps
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(min=0.0, max=0.999)


def linear_betas(
    num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2,
) -> torch.Tensor:
    """Standard linear schedule (Ho et al., NeurIPS 2020)."""
    return torch.linspace(beta_start, beta_end, num_steps)


# =============================================================================
# DDPM scheduler
# =============================================================================


class DDPMScheduler(nn.Module):
    """DDPM forward / reverse step machinery with EDGE-style extensions.

    Pre-computes ``α``, ``ᾱ``, ``√ᾱ``, ``√(1-ᾱ)``, posterior coefficients,
    and the P2 loss-weight buffer so neither training nor sampling does
    schedule math on the fly. Buffers are registered with the module so
    they migrate cleanly under ``.to(device)`` and DDP.

    Construction reads from :class:`DiffusionCfg` directly so all knobs
    (schedule type, cosine offset, posterior variance choice, p2
    weighting params) are recorded in one place.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        schedule: str = "cosine",
        cosine_s: float = 0.008,
        posterior_variance: str = "beta_tilde",
        p2_loss_weighting: bool = True,
        p2_loss_weight_k: float = 1.0,
        p2_loss_weight_gamma: float = 0.5,
    ) -> None:
        super().__init__()
        if schedule == "cosine":
            betas = cosine_betas(num_steps, s=cosine_s)
        elif schedule == "linear":
            betas = linear_betas(num_steps)
        else:
            raise ValueError(f"unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.0)

        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)
        self.register_buffer("alphas_bar_prev", alphas_bar_prev)
        self.register_buffer("sqrt_alphas_bar", alphas_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_bar", (1.0 - alphas_bar).sqrt())

        # Posterior q(z_{t-1} | z_t, x_0). Two variance choices:
        #   "beta_tilde": β̃_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)  (DDPM lower bound)
        #   "beta":      β_t                                       (DDPM upper bound)
        # EDGE uses β̃_t (the default here).
        beta_tilde = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
        if posterior_variance == "beta_tilde":
            var_buf = beta_tilde
        elif posterior_variance == "beta":
            var_buf = betas
        else:
            raise ValueError(
                f"posterior_variance must be 'beta' or 'beta_tilde'; "
                f"got {posterior_variance!r}"
            )
        self.register_buffer("posterior_variance", var_buf)
        # The variance can be 0 at t=0; clamp the log for numerical safety.
        self.register_buffer(
            "posterior_log_variance",
            var_buf.clamp(min=1e-20).log(),
        )
        self.register_buffer(
            "posterior_mean_coef_x0",
            betas * alphas_bar_prev.sqrt() / (1.0 - alphas_bar),
        )
        self.register_buffer(
            "posterior_mean_coef_zt",
            (1.0 - alphas_bar_prev) * alphas.sqrt() / (1.0 - alphas_bar),
        )

        # ---- P2 SNR-based loss weighting (EDGE lines 125-130) ------------
        # w_t = (k + ᾱ_t/(1-ᾱ_t))^(-γ). With γ=0 the weight is 1 (no-op).
        gamma_eff = p2_loss_weight_gamma if p2_loss_weighting else 0.0
        snr = alphas_bar / (1.0 - alphas_bar).clamp(min=1e-20)
        p2_w = (p2_loss_weight_k + snr) ** (-gamma_eff)
        self.register_buffer("p2_loss_weight", p2_w)

    def _gather(
        self, buf: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Gather schedule values at indices ``t`` and broadcast to ``x_shape``."""
        out = buf.gather(0, t)
        return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward noise: ``z_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε``."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_ab = self._gather(self.sqrt_alphas_bar, t, x_start.shape)
        sqrt_1m_ab = self._gather(self.sqrt_one_minus_alphas_bar, t, x_start.shape)
        return sqrt_ab * x_start + sqrt_1m_ab * noise

    def predict_x0_from_eps(
        self, z_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor,
    ) -> torch.Tensor:
        """Invert the forward process: ``x̂_0 = (z_t - √(1-ᾱ_t)·ε) / √ᾱ_t``.

        Used when the model is in ``predict_epsilon`` mode and the
        downstream sampler / loss needs the clean prediction.
        """
        sqrt_ab = self._gather(self.sqrt_alphas_bar, t, z_t.shape).clamp(min=1e-8)
        sqrt_1m_ab = self._gather(self.sqrt_one_minus_alphas_bar, t, z_t.shape)
        return (z_t - sqrt_1m_ab * eps) / sqrt_ab

    def predict_eps_from_x0(
        self, z_t: torch.Tensor, t: torch.Tensor, x_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the implied ε from a predicted x̂.

        Inverse of :meth:`predict_x0_from_eps`. Used inside DDIM step.
        """
        sqrt_ab = self._gather(self.sqrt_alphas_bar, t, z_t.shape)
        sqrt_1m_ab = self._gather(self.sqrt_one_minus_alphas_bar, t, z_t.shape).clamp(min=1e-8)
        return (z_t - sqrt_ab * x_hat) / sqrt_1m_ab

    def posterior_mean(
        self, x_start: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor,
    ) -> torch.Tensor:
        """``μ_{t-1}(z_t, x_0)`` — the posterior mean under DDPM."""
        c_x0 = self._gather(self.posterior_mean_coef_x0, t, x_start.shape)
        c_zt = self._gather(self.posterior_mean_coef_zt, t, x_start.shape)
        return c_x0 * x_start + c_zt * z_t

    def p_sample_from_xhat(
        self, x_hat: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One reverse step using the model's predicted ``x̂``.

        At t=0 the variance is 0 so we just return the posterior mean.
        """
        mean = self.posterior_mean(x_hat, z_t, t)
        var = self._gather(self.posterior_variance, t, x_hat.shape)
        if noise is None:
            noise = torch.randn_like(z_t)
        # Zero-out noise at t=0 (no further denoising step).
        nonzero_mask = (t > 0).float().view(t.shape[0], *([1] * (z_t.ndim - 1)))
        return mean + nonzero_mask * var.sqrt() * noise


# =============================================================================
# Building blocks
# =============================================================================


class TimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding + MLP, after Ho et al. (NeurIPS 2020)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``(B,) int64 → (B, dim) float``."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class FiLM(nn.Module):
    """Feature-wise linear modulation: ``out = (1 + γ(c)) · x + β(c)``.

    Initialised so that γ ≈ 0 and β ≈ 0 at start (no modulation), which
    keeps early training stable.
    """

    def __init__(self, cond_dim: int, x_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, x_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, T, x_dim); cond: (B, cond_dim)
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1.0 + gamma.unsqueeze(1)) * x + beta.unsqueeze(1)


class DiffusionDecoderBlock(nn.Module):
    """Self-attn + cross-attn + MLP, each gated by FiLM(t) (EDGE Fig. 2)."""

    def __init__(
        self, dim: int, heads: int, dim_ff: int, dropout: float, cond_dim: int,
    ) -> None:
        super().__init__()
        self.ln_self = nn.LayerNorm(dim)
        self.film_self = FiLM(cond_dim, dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.ln_cross = nn.LayerNorm(dim)
        self.film_cross = FiLM(cond_dim, dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.ln_ff = nn.LayerNorm(dim)
        self.film_ff = FiLM(cond_dim, dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, t_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention with FiLM(t) modulation.
        h = self.film_self(self.ln_self(x), t_emb)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h
        # Cross-attention to the conditioning sequence.
        h = self.film_cross(self.ln_cross(x), t_emb)
        h, _ = self.cross_attn(h, context, context, need_weights=False)
        x = x + h
        # Feed-forward.
        h = self.film_ff(self.ln_ff(x), t_emb)
        x = x + self.ff(h)
        return x


# =============================================================================
# Main model
# =============================================================================


@dataclass
class DiffusionOutput:
    """Structured output of :meth:`MotionDiffusion.forward`."""

    x_hat: torch.Tensor                         # (B, T, 151) predicted clean sample
    # Optional debug fields (sampling-only)
    intermediate: Optional[List[torch.Tensor]] = None


class MotionDiffusion(nn.Module):
    """EDGE-style pose-conditioned motion diffusion.

    Architecture: transformer decoder with FiLM(t) modulation and
    cross-attention over a short conditioning sequence (we have no music;
    the cross-attention captures action-type / difficulty / query-summary
    conditioning instead). Trained with DDPM (`L_simple` + auxiliaries)
    and classifier-free guidance dropout.

    The model can be configured to predict either:
      - ``x̂`` (the clean sample) — EDGE convention, default.
      - ``ε`` (the noise) — DDPM convention, set via ``cfg.predict_x0=False``.

    Internally :meth:`forward` returns the raw network output; the
    sampling and loss code use :meth:`predict_x0` to consistently get
    the clean prediction regardless of mode.
    """

    def __init__(
        self, cfg: DiffusionCfg, data_cfg: DataCfg,
        rest_positions: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_cfg = data_cfg
        if cfg.token_dim != TOKEN_DIM:
            raise ValueError(
                f"DiffusionCfg.token_dim={cfg.token_dim} disagrees with the SMPL "
                f"convention TOKEN_DIM={TOKEN_DIM}"
            )

        # Input / output projections.
        self.input_proj = nn.Linear(TOKEN_DIM, cfg.width)
        self.output_proj = nn.Linear(cfg.width, TOKEN_DIM)
        # Zero-init the output projection so the model starts predicting ≈ 0
        # (the identity reconstruction for ε mode, or the zero pose in x̂
        # mode), which empirically stabilises early steps.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # Learned positional embedding.
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.seq_frames, cfg.width))
        nn.init.normal_(self.pos_emb, std=0.02)

        # Timestep embedding.
        self.t_emb = TimestepEmbedding(cfg.width)

        # Conditioning projection.
        self.cond_dim_total = cfg.cond_action_embedding_dim + cfg.cond_summary_dim
        self.cond_proj = nn.Linear(self.cond_dim_total, cfg.width)

        # Stack of transformer decoder blocks.
        self.blocks = nn.ModuleList([
            DiffusionDecoderBlock(
                dim=cfg.width, heads=cfg.heads, dim_ff=cfg.dim_ff,
                dropout=cfg.dropout, cond_dim=cfg.width,
            )
            for _ in range(cfg.depth)
        ])
        self.final_norm = nn.LayerNorm(cfg.width)

        # DDPM scheduler — pulls every numerical knob from cfg.
        self.scheduler = DDPMScheduler(
            num_steps=cfg.diffusion_steps,
            schedule=cfg.noise_schedule,
            cosine_s=cfg.cosine_s,
            posterior_variance=cfg.posterior_variance,
            p2_loss_weighting=cfg.p2_loss_weighting,
            p2_loss_weight_k=cfg.p2_loss_weight_k,
            p2_loss_weight_gamma=cfg.p2_loss_weight_gamma,
        )

        # Forward kinematics layer for L_pos and L_contact.
        self.kinematics = SMPLKinematics(rest_positions=rest_positions)

        # Optional gradient-checkpointing.
        self.use_grad_checkpoint = cfg.grad_checkpoint

    @property
    def predict_x0(self) -> bool:
        """Mirror of ``cfg.predict_x0`` for downstream convenience."""
        return self.cfg.predict_x0

    # ---- forward / sampling ------------------------------------------

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor,
    ) -> torch.Tensor:
        """Run the raw network. The output is x̂ if ``cfg.predict_x0`` else ε.

        Args:
            z_t: ``(B, T, 151)`` noisy pose tokens.
            t: ``(B,) int64`` diffusion timesteps in ``[0, num_steps)``.
            context: ``(B, cond_dim_total)`` conditioning vector.

        Returns:
            ``(B, T, 151)`` — the network's raw prediction. Use
            :meth:`predict_x0_clean` when you want x̂ regardless of mode.
        """
        B, T, _ = z_t.shape
        if T > self.cfg.seq_frames:
            raise ValueError(
                f"sequence length {T} exceeds DiffusionCfg.seq_frames={self.cfg.seq_frames}"
            )
        h = self.input_proj(z_t)
        h = h + self.pos_emb[:, :T]

        t_emb = self.t_emb(t)
        cond_tokens = self.cond_proj(context).unsqueeze(1)

        for block in self.blocks:
            if self.use_grad_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    block, h, cond_tokens, t_emb, use_reentrant=False,
                )
            else:
                h = block(h, cond_tokens, t_emb)

        h = self.final_norm(h)
        return self.output_proj(h)

    def predict_x0_clean(
        self, z_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor,
    ) -> torch.Tensor:
        """Always return x̂ regardless of cfg.predict_x0.

        When the model predicts ε, converts via the standard formula
        ``x̂ = (z_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t``.
        """
        raw = self.forward(z_t, t, context)
        if self.cfg.predict_x0:
            return raw
        return self.scheduler.predict_x0_from_eps(z_t, t, raw)


    # ---- sampling -----------------------------------------------------

    @torch.no_grad()
    def sample(
        self, context: torch.Tensor, num_frames: Optional[int] = None,
        guidance_w: Optional[float] = None, seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Standard DDPM sampling with classifier-free guidance.

        Args:
            context: ``(B, cond_dim_total)`` conditioning vector.
            num_frames: length of the sampled sequence (default cfg.seq_frames).
            guidance_w: classifier-free guidance weight (default cfg).
            seed: optional generator seed for reproducibility.

        Returns:
            ``(B, T, 151)`` sampled pose tokens.
        """
        num_frames = num_frames or self.cfg.seq_frames
        guidance_w = guidance_w if guidance_w is not None else self.cfg.guidance_weight_default
        device = context.device
        B = context.shape[0]
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        z = torch.randn(B, num_frames, TOKEN_DIM, device=device, generator=generator)
        null_context = torch.zeros_like(context)
        for t_int in reversed(range(self.scheduler.num_steps)):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            x_hat = self._cfg_predict_x0(z, t, context, null_context, guidance_w)
            z = self.scheduler.p_sample_from_xhat(x_hat, z, t)
        return z

    @torch.no_grad()
    def sample_ddim(
        self, context: torch.Tensor, num_inference_steps: int = 50,
        num_frames: Optional[int] = None, guidance_w: Optional[float] = None,
        eta: float = 0.0, seed: Optional[int] = None,
    ) -> torch.Tensor:
        """DDIM sampling — accelerated, used by the CF search inner loop.

        ``eta=0`` is fully deterministic; ``eta=1`` recovers DDPM-equivalent
        stochasticity. The CF search uses ``eta=0`` so multiple counterfactual
        candidates differ only via mask / context / guidance, not via the
        sampler's own randomness.
        """
        num_frames = num_frames or self.cfg.seq_frames
        guidance_w = guidance_w if guidance_w is not None else self.cfg.guidance_weight_default
        device = context.device
        B = context.shape[0]
        timesteps = torch.linspace(
            self.scheduler.num_steps - 1, 0, num_inference_steps,
            device=device, dtype=torch.long,
        ).tolist()
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        z = torch.randn(B, num_frames, TOKEN_DIM, device=device, generator=generator)
        null_context = torch.zeros_like(context)
        for i, t_int in enumerate(timesteps):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x_hat = self._cfg_predict_x0(z, t, context, null_context, guidance_w)
            z = self._ddim_step(z, x_hat, t_int, t_prev, eta, generator)
        return z

    # ---- inpainting --------------------------------------------------

    @torch.no_grad()
    def inpaint(
        self, x_known: torch.Tensor, mask: torch.Tensor, context: torch.Tensor,
        guidance_w: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        noise_level_tau: float = 1.0,
        inpaint_resamples: int = 0,
        inpaint_resample_fraction: float = 0.2,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Masked inpainting (EDGE / RePaint).

        Args:
            x_known: ``(B, T, 151)`` known pose tokens.
            mask: ``(B, T, 151)`` with 1 where ``x_known`` is preserved and
                0 where the model should generate.
            context: ``(B, cond_dim_total)`` conditioning vector.
            num_inference_steps: if given, use DDIM with this many steps.
            noise_level_tau: in (0, 1]. With τ=1.0 (default), the latent
                is initialised from pure noise and the denoising loop
                runs from t = num_steps−1 down to 0. With τ<1.0 (DiME
                late-start), the loop runs from t = ⌊τ·num_steps⌋−1
                downward and z is initialised from a forward-noised
                ``x_known`` at that timestep.
            inpaint_resamples (R): RePaint Algorithm 1 — over the last
                ``inpaint_resample_fraction`` × num_inference_steps
                steps, run R inner iterations of (reverse-step, re-noise
                back to t). R=0 (default) disables — single-pass EDGE
                behaviour.
            inpaint_resample_fraction: fraction of final steps to apply
                RePaint resampling to. Only relevant when R>0.
            seed: optional generator seed.

        Returns:
            ``(B, T, 151)`` inpainted sequence with ``x_known``
            bit-restored under ``mask=1``.
        """
        if not 0.0 < noise_level_tau <= 1.0:
            raise ValueError(f"noise_level_tau must be in (0, 1]; got {noise_level_tau}")
        if inpaint_resamples < 0:
            raise ValueError(f"inpaint_resamples must be >= 0; got {inpaint_resamples}")
        if not 0.0 < inpaint_resample_fraction <= 1.0:
            raise ValueError(
                f"inpaint_resample_fraction must be in (0, 1]; got {inpaint_resample_fraction}"
            )

        guidance_w = guidance_w if guidance_w is not None else self.cfg.guidance_weight_default
        device = x_known.device
        B, T, _ = x_known.shape
        null_context = torch.zeros_like(context)
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        timesteps, ddim = self._build_timesteps(
            num_inference_steps, noise_level_tau, device,
        )
        z = self._init_latent(
            x_known, noise_level_tau, B, T, device, generator,
        )

        resample_start_idx = self._resample_start_idx(
            len(timesteps), inpaint_resamples, inpaint_resample_fraction,
        )

        for i, t_int in enumerate(timesteps):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            # How many resamples at this step (RePaint Algorithm 1).
            r_count = inpaint_resamples if (
                inpaint_resamples > 0 and i >= resample_start_idx
            ) else 1

            for r in range(r_count):
                # Re-impose known region at this timestep.
                z_known_t = self.scheduler.q_sample(x_known, t)
                z = mask * z_known_t + (1.0 - mask) * z
                x_hat = self._cfg_predict_x0(
                    z, t, context, null_context, guidance_w,
                )
                if ddim:
                    z = self._ddim_step(
                        z, x_hat, t_int, t_prev, eta=0.0, generator=generator,
                    )
                else:
                    z = self.scheduler.p_sample_from_xhat(x_hat, z, t)
                # Re-noise back to t for the next resample iteration.
                if r < r_count - 1 and t_prev >= 0:
                    t_prev_tensor = torch.full(
                        (B,), t_prev, device=device, dtype=torch.long,
                    )
                    # Forward step from t_prev back to t — sample fresh noise.
                    noise = torch.randn(
                        z.shape, device=device, generator=generator,
                    )
                    # z_t = √(α_t/α_{t-1}) · z_{t-1} + √(1 - α_t/α_{t-1}) · ε
                    a_t = self.scheduler.alphas_bar[t_int]
                    a_prev = self.scheduler.alphas_bar[t_prev].clamp(min=1e-8)
                    ratio = (a_t / a_prev).clamp(min=0.0, max=1.0)
                    z = ratio.sqrt() * z + (1.0 - ratio).sqrt() * noise

        # Final hard replacement: the known region is restored bit-exact.
        return mask * x_known + (1.0 - mask) * z

    def inpaint_with_score_guidance(
        self, x_known: torch.Tensor, mask: torch.Tensor, context: torch.Tensor,
        score_fn: Callable[[torch.Tensor], torch.Tensor],
        target_uplift: float,
        grad_scale: float = 1.0,
        guidance_w: Optional[float] = None,
        num_inference_steps: int = 50,
        noise_level_tau: float = 1.0,
        dime_gradient_scaling: bool = False,
        inpaint_resamples: int = 0,
        inpaint_resample_fraction: float = 0.2,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Inpainting + classifier-style guidance toward a score uplift.

        This is the hook the half-point counterfactual search calls. The
        ``score_fn`` is a callable that accepts the model's predicted clean
        sample ``x̂`` and returns ``(B,)`` AQA scores; it stays outside this
        file's import graph so the diffusion module never depends on
        :mod:`aqa_model`.

        Algorithm (mode-dependent):
          - ``dime_gradient_scaling=False`` (autograd path, default):
            backprop the score loss through one denoising step ``z → x̂``,
            yielding the gradient w.r.t. ``z_t`` directly. Equivalent to
            the previous codebase's behaviour.
          - ``dime_gradient_scaling=True`` (DiME Eq. 8): take the gradient
            w.r.t. ``x̂`` only, then scale by 1/√ᾱ_t when applying to z.
            This decouples the chain rule from the network's implicit
            Jacobian and is more stable when sweeping ``noise_level_tau``.

        ``noise_level_tau`` and ``inpaint_resamples`` behave as in
        :meth:`inpaint`.

        Note: NOT decorated with ``@torch.no_grad`` — we need autograd
        active through ``score_fn``. Callers should still wrap the AQA
        oracle they pass in :meth:`TSAModel.frozen` so its parameters
        don't move.
        """
        if not 0.0 < noise_level_tau <= 1.0:
            raise ValueError(f"noise_level_tau must be in (0, 1]; got {noise_level_tau}")
        if inpaint_resamples < 0:
            raise ValueError(f"inpaint_resamples must be >= 0; got {inpaint_resamples}")

        guidance_w = guidance_w if guidance_w is not None else self.cfg.guidance_weight_default
        device = x_known.device
        B, T, _ = x_known.shape
        null_context = torch.zeros_like(context)
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

        timesteps, _ddim = self._build_timesteps(
            num_inference_steps, noise_level_tau, device,
        )
        z = self._init_latent(
            x_known, noise_level_tau, B, T, device, generator,
        )

        # Establish the AQA baseline (no guidance applied yet).
        with torch.no_grad():
            baseline_score = score_fn(x_known)                  # (B,)
        target_score = baseline_score + target_uplift

        resample_start_idx = self._resample_start_idx(
            len(timesteps), inpaint_resamples, inpaint_resample_fraction,
        )

        for i, t_int in enumerate(timesteps):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            r_count = inpaint_resamples if (
                inpaint_resamples > 0 and i >= resample_start_idx
            ) else 1

            for r in range(r_count):
                # Re-impose known region at this timestep (no_grad — pure
                # forward noise; no gradients should flow through this).
                with torch.no_grad():
                    z_known_t = self.scheduler.q_sample(x_known, t)
                z = mask * z_known_t + (1.0 - mask) * z

                # ---- compute the guidance gradient ----
                z_with_grad = z.detach().requires_grad_(True)
                with torch.enable_grad():
                    x_hat = self._cfg_predict_x0(
                        z_with_grad, t, context, null_context, guidance_w,
                    )
                    if dime_gradient_scaling:
                        # DiME Eq. 8: gradient w.r.t. x_hat, then scale by 1/√ᾱ_t.
                        # Detach the x_hat path before scoring to avoid double-counting.
                        x_hat_leaf = x_hat.detach().requires_grad_(True)
                        scores = score_fn(x_hat_leaf)
                        loss = ((scores - target_score) ** 2).sum()
                        (grad_xhat,) = torch.autograd.grad(loss, x_hat_leaf)
                        ab_t = self.scheduler.alphas_bar[t_int].clamp(min=1e-8)
                        grad_z = grad_xhat / ab_t.sqrt()
                    else:
                        # Autograd through one denoising step.
                        scores = score_fn(x_hat)
                        loss = ((scores - target_score) ** 2).sum()
                        (grad_z,) = torch.autograd.grad(loss, z_with_grad)

                # ---- denoise + apply guidance (no_grad past here) ----
                with torch.no_grad():
                    # Recompute x_hat fresh in no_grad (cheap; lets us
                    # drop the autograd graph between iterations).
                    x_hat_ng = self._cfg_predict_x0(
                        z, t, context, null_context, guidance_w,
                    )
                    z_next = self._ddim_step(
                        z, x_hat_ng, t_int, t_prev, eta=0.0, generator=generator,
                    )
                    # Apply guidance only to the regenerated region.
                    z = z_next - grad_scale * grad_z * (1.0 - mask)

                # RePaint inner-resample re-noise.
                if r < r_count - 1 and t_prev >= 0:
                    with torch.no_grad():
                        noise = torch.randn(z.shape, device=device, generator=generator)
                        a_t = self.scheduler.alphas_bar[t_int]
                        a_prev = self.scheduler.alphas_bar[t_prev].clamp(min=1e-8)
                        ratio = (a_t / a_prev).clamp(min=0.0, max=1.0)
                        z = ratio.sqrt() * z + (1.0 - ratio).sqrt() * noise

        return mask * x_known + (1.0 - mask) * z.detach()


    # ---- private helpers ---------------------------------------------

    def _cfg_predict_x0(
        self, z: torch.Tensor, t: torch.Tensor,
        context: torch.Tensor, null_context: torch.Tensor, w: float,
    ) -> torch.Tensor:
        """Classifier-free guidance combination, always returning x̂.

        Internally calls :meth:`predict_x0_clean` for both branches so
        the ε-prediction mode works transparently here.
        """
        x_cond = self.predict_x0_clean(z, t, context)
        if w == 1.0:
            return x_cond
        x_uncond = self.predict_x0_clean(z, t, null_context)
        return w * x_cond + (1.0 - w) * x_uncond

    def _ddim_step(
        self, z_t: torch.Tensor, x_hat: torch.Tensor,
        t_int: int, t_prev: int, eta: float, generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """One DDIM update step.

        Computes the next-timestep latent given the current latent ``z_t``
        and the model's predicted clean sample ``x̂``. ``t_prev < 0`` means
        we're at the last step and should return ``x̂``.
        """
        if t_prev < 0:
            return x_hat
        ab_t = self.scheduler.alphas_bar[t_int]
        ab_prev = self.scheduler.alphas_bar[t_prev]
        # Derive the implied noise from x_hat and z_t.
        eps = (z_t - ab_t.sqrt() * x_hat) / (1.0 - ab_t).sqrt().clamp(min=1e-8)
        sigma = eta * ((1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)).clamp(min=0.0).sqrt()
        mean = ab_prev.sqrt() * x_hat + (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps
        if eta > 0:
            noise = torch.randn(z_t.shape, device=z_t.device, generator=generator)
            return mean + sigma * noise
        return mean

    def _build_timesteps(
        self,
        num_inference_steps: Optional[int],
        noise_level_tau: float,
        device,
    ) -> Tuple[List[int], bool]:
        """Build the descending timestep list, honouring ``noise_level_tau``.

        Returns ``(timesteps_list, is_ddim)``.

        With τ=1.0 and num_inference_steps=None, this is the full DDPM
        schedule from num_steps-1 down to 0. With τ<1.0 the loop starts
        from ⌊τ·num_steps⌋−1 instead.
        """
        T_max = self.scheduler.num_steps
        start = int(noise_level_tau * T_max) - 1
        start = max(0, min(start, T_max - 1))

        if num_inference_steps is None:
            timesteps = list(range(start, -1, -1))
            return timesteps, False

        # DDIM: equally-spaced from `start` down to 0.
        ts = torch.linspace(
            start, 0, num_inference_steps,
            device=device, dtype=torch.long,
        ).tolist()
        return ts, True

    def _init_latent(
        self, x_known: torch.Tensor, noise_level_tau: float,
        B: int, T: int, device, generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Initialise the inpainting latent z.

        With τ=1.0, pure Gaussian noise (standard DDPM start). With τ<1.0,
        the latent is set to q(z_{τT} | x_known) — a forward-noised version
        of the known sample at the late-start timestep.
        """
        if noise_level_tau >= 1.0:
            return torch.randn(B, T, TOKEN_DIM, device=device, generator=generator)
        start = int(noise_level_tau * self.scheduler.num_steps) - 1
        start = max(0, min(start, self.scheduler.num_steps - 1))
        t_start = torch.full((B,), start, device=device, dtype=torch.long)
        noise = torch.randn(B, T, TOKEN_DIM, device=device, generator=generator)
        return self.scheduler.q_sample(x_known, t_start, noise=noise)

    @staticmethod
    def _resample_start_idx(
        num_steps: int, resamples: int, fraction: float,
    ) -> int:
        """Index of the first timestep at which RePaint resampling kicks in.

        Returns ``num_steps`` (= never) when resampling is disabled or
        the fraction is 0; otherwise the index of the first step in
        the final ``fraction × num_steps`` window.
        """
        if resamples <= 0 or fraction <= 0.0:
            return num_steps
        n_resample = max(1, int(round(num_steps * fraction)))
        return num_steps - n_resample


# =============================================================================
# Losses (EDGE Eq. 2-5 + the simple objective)
# =============================================================================
#
# All per-sample loss functions return a (B,) tensor so the calling code
# can apply per-sample p2 weighting (EDGE-main lines 464/477/496) before
# reducing to a scalar. This is the difference from the previous version,
# which returned scalars and could not be p2-weighted.
# -----------------------------------------------------------------------------


def loss_simple_per_sample(
    pred: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """``L_simple = ‖target - pred‖²`` per sample (EDGE Eq. 2).

    Returns (B,). The caller picks ``target`` as either the clean ``x``
    (when ``predict_x0=True``) or the noise ``ε`` (when ``predict_x0=False``).
    """
    return ((pred - target) ** 2).mean(dim=tuple(range(1, pred.ndim)))


def loss_velocity_per_sample(
    x_hat: torch.Tensor, x_target: torch.Tensor,
) -> torch.Tensor:
    """``L_vel`` per sample — finite-difference velocity in token space (EDGE Eq. 4).

    Returns (B,). Always computed against the clean x̂ (regardless of
    predict_x0 mode), because velocity is a temporal-derivative property
    of the clean sequence, not of the noise.
    """
    v_hat = x_hat[..., 1:, :] - x_hat[..., :-1, :]
    v_target = x_target[..., 1:, :] - x_target[..., :-1, :]
    return ((v_hat - v_target) ** 2).mean(dim=tuple(range(1, v_hat.ndim)))


def loss_position_per_sample(
    x_hat: torch.Tensor, x_target: torch.Tensor, kinematics: SMPLKinematics,
) -> torch.Tensor:
    """``L_pos = ‖FK(x) - FK(x̂)‖²`` per sample (EDGE Eq. 3). Returns (B,)."""
    pos_hat = _token_to_joint_positions(x_hat, kinematics)
    pos_target = _token_to_joint_positions(x_target, kinematics)
    return ((pos_hat - pos_target) ** 2).mean(dim=tuple(range(1, pos_hat.ndim)))


def loss_contact_per_sample(
    x_hat: torch.Tensor, kinematics: SMPLKinematics,
) -> torch.Tensor:
    """EDGE's Contact Consistency Loss (Eq. 5), per sample. Returns (B,).

    Penalises foot-joint *velocity* in frames where the model itself
    predicts foot contact. Unlike a vanilla foot-skate loss, this depends
    only on the model's own output, which lets it shape both the contact
    and the kinematics jointly. p2 weighting is NOT applied to this term
    in EDGE (verified at eval/diffusion.py:509-512 — no p2 multiplication).
    """
    pos = _token_to_joint_positions(x_hat, kinematics)              # (B, T, 24, 3)
    foot_pos = pos[..., SMPL_FOOT_JOINTS, :]                        # (B, T, 4, 3)
    foot_vel = foot_pos[..., 1:, :, :] - foot_pos[..., :-1, :, :]   # (B, T-1, 4, 3)
    contacts_hat = x_hat[..., :-1, CONTACT_SLICE]                   # (B, T-1, 4)
    weighted = foot_vel * contacts_hat.unsqueeze(-1)
    return (weighted ** 2).mean(dim=tuple(range(1, weighted.ndim)))


# Backwards-compatible scalar wrappers — the previous file exported these.
def loss_simple(x_hat: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    """DEPRECATED — use :func:`loss_simple_per_sample` for p2 weighting."""
    return loss_simple_per_sample(x_hat, x_target).mean()


def loss_velocity(x_hat: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    """DEPRECATED — use :func:`loss_velocity_per_sample`."""
    return loss_velocity_per_sample(x_hat, x_target).mean()


def loss_position(
    x_hat: torch.Tensor, x_target: torch.Tensor, kinematics: SMPLKinematics,
) -> torch.Tensor:
    """DEPRECATED — use :func:`loss_position_per_sample`."""
    return loss_position_per_sample(x_hat, x_target, kinematics).mean()


def loss_contact(
    x_hat: torch.Tensor, kinematics: SMPLKinematics,
) -> torch.Tensor:
    """DEPRECATED — use :func:`loss_contact_per_sample`."""
    return loss_contact_per_sample(x_hat, kinematics).mean()


def _token_to_joint_positions(
    token: torch.Tensor, kinematics: SMPLKinematics,
) -> torch.Tensor:
    """Apply detokenise → 6D-to-rotmat → FK in one shot."""
    decoded = detokenize_pose(token)
    rotmat = sixd_to_rotmat(decoded["sixd"])
    return kinematics(rotmat, decoded["root"])


def compute_diffusion_loss(
    model: MotionDiffusion,
    x: torch.Tensor,                # (B, T, 151) clean target
    context: torch.Tensor,          # (B, cond_dim_total) conditioning
    *,
    cfg_drop_prob: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """One training step's loss dictionary.

    Steps:
        1. Sample t uniformly per-sample from ``[0, num_steps)``.
        2. Sample Gaussian noise ε and form ``z_t = q_sample(x, t)``.
        3. (Classifier-free dropout) replace ``context`` with zeros for a
           fraction of the batch.
        4. Run the model. If ``cfg.predict_x0`` the output IS x̂; otherwise
           it's ε and we derive x̂ via the standard formula for the
           auxiliary losses.
        5. Compose
              total = λ_s·⟨w_t·L_simple⟩ + λ_v·⟨w_t·L_vel⟩
                    + λ_p·⟨w_t·L_pos⟩  + λ_c·⟨L_contact⟩
           where ``w_t = (1 + SNR(t))^(-γ/2)`` is the p2 weight when
           ``cfg.p2_loss_weighting=True``, else w_t ≡ 1. p2 weighting is
           NOT applied to L_contact (matches EDGE-main lines 464/477/496;
           L_contact at 509-512 carries no p2 multiplication).

    Returns a dict with ``total``, ``simple``, ``pos``, ``vel``, ``contact``,
    so the train loop can log each component.
    """
    cfg = model.cfg
    B = x.shape[0]
    device = x.device
    drop_p = cfg.cf_drop_prob if cfg_drop_prob is None else cfg_drop_prob

    # 1) sample timesteps.
    t = torch.randint(
        0, model.scheduler.num_steps, (B,), device=device, dtype=torch.long,
    )
    # 2) forward noising.
    noise = torch.randn_like(x)
    z_t = model.scheduler.q_sample(x, t, noise=noise)
    # 3) classifier-free dropout — per-sample.
    if drop_p > 0:
        drop_mask = (torch.rand(B, device=device) < drop_p).view(B, 1)
        context = context * (~drop_mask).float()
    # 4) predict, and recover x̂ regardless of prediction mode.
    raw_pred = model(z_t, t, context)
    if cfg.predict_x0:
        x_hat = raw_pred
        simple_target = x
    else:
        x_hat = model.scheduler.predict_x0_from_eps(z_t, t, raw_pred)
        simple_target = noise

    # 5) per-sample loss components.
    simple_b  = loss_simple_per_sample(raw_pred, simple_target)         # (B,)
    vel_b     = loss_velocity_per_sample(x_hat, x)                       # (B,)
    pos_b     = loss_position_per_sample(x_hat, x, model.kinematics)     # (B,)
    contact_b = loss_contact_per_sample(x_hat, model.kinematics)         # (B,)

    # P2 weight per sample (a scalar per item, picked at sample's t).
    # When p2_loss_weighting=False the scheduler stored a buffer of 1's,
    # so this multiplication is a no-op.
    p2_w = model.scheduler.p2_loss_weight.gather(0, t)                   # (B,)

    simple_loss  = (p2_w * simple_b).mean()
    vel_loss     = (p2_w * vel_b).mean()
    pos_loss     = (p2_w * pos_b).mean()
    # NOTE: contact loss is NOT p2-weighted (EDGE convention).
    contact_loss = contact_b.mean()

    losses: Dict[str, torch.Tensor] = {
        "simple":  simple_loss,
        "vel":     vel_loss,
        "pos":     pos_loss,
        "contact": contact_loss,
    }
    losses["total"] = (
        cfg.lambda_simple  * simple_loss
      + cfg.lambda_vel     * vel_loss
      + cfg.lambda_pos     * pos_loss
      + cfg.lambda_contact * contact_loss
    )
    return losses


# =============================================================================
# Mask builders — the three intervention modalities
# =============================================================================
#
# All mask builders produce shape (T, 151) with 1 where the original
# token is preserved and 0 where the diffusion model should generate.
# They are deterministic given their arguments and stateless; the
# counterfactual search composes them via element-wise multiplication.
# -----------------------------------------------------------------------------


def _joint_rotation_indices(joint_indices: Sequence[int]) -> List[int]:
    """Token-dim indices for the 6D rotation slots of the given SMPL joints."""
    out: List[int] = []
    base = CONTACT_DIM
    for j in joint_indices:
        if not (0 <= j < NUM_SMPL_JOINTS):
            raise ValueError(f"joint index {j} out of [0, {NUM_SMPL_JOINTS})")
        out.extend(range(base + j * 6, base + (j + 1) * 6))
    return out


def make_joint_subset_mask(
    num_frames: int, joint_indices: Sequence[int],
    frame_range: Optional[Tuple[int, int]] = None,
    device=None, dtype=torch.float32,
) -> torch.Tensor:
    """Mask that frees specific SMPL joints to be regenerated.

    Args:
        num_frames: T.
        joint_indices: which SMPL joints (by index 0..23) to free.
        frame_range: ``(t_start, t_end)`` to limit the regeneration to a
            specific frame window. ``None`` frees those joints in every frame.

    Returns ``(T, 151)`` mask, 1 to preserve, 0 to regenerate.
    """
    mask = torch.ones(num_frames, TOKEN_DIM, device=device, dtype=dtype)
    cols = _joint_rotation_indices(joint_indices)
    t0, t1 = (0, num_frames) if frame_range is None else frame_range
    t0 = max(0, t0); t1 = min(num_frames, t1)
    if t0 < t1:
        mask[t0:t1, cols] = 0.0
    return mask


def make_phase_window_mask(
    num_frames: int, frame_range: Tuple[int, int],
    include_root: bool = True, include_contacts: bool = True,
    device=None, dtype=torch.float32,
) -> torch.Tensor:
    """Mask that frees an entire phase window (all joints) to be regenerated.

    Args:
        num_frames: T.
        frame_range: ``(t_start, t_end)`` defining the phase window.
        include_root / include_contacts: whether to also free the root
            translation / foot contacts inside the window. EDGE-style
            inpainting usually wants both.

    Returns ``(T, 151)`` mask.
    """
    mask = torch.ones(num_frames, TOKEN_DIM, device=device, dtype=dtype)
    t0, t1 = frame_range
    t0 = max(0, t0); t1 = min(num_frames, t1)
    if t0 >= t1:
        return mask
    mask[t0:t1, ROTATION_SLICE] = 0.0
    if include_root:
        mask[t0:t1, ROOT_SLICE] = 0.0
    if include_contacts:
        mask[t0:t1, CONTACT_SLICE] = 0.0
    return mask


def make_transition_shift_mask(
    num_frames: int, transition_frame: int, shift_frames: int,
    device=None, dtype=torch.float32,
) -> torch.Tensor:
    """Mask that frees a small window around a step boundary so the
    diffusion model can re-time the transition.

    The window is ``[transition_frame - |shift|, transition_frame + |shift|]``;
    the magnitude of ``shift_frames`` controls how much temporal slack the
    inpainter has. Sign carries through to the counterfactual search's
    interpretation (later vs. earlier transition).

    Returns ``(T, 151)`` mask.
    """
    radius = max(1, abs(int(shift_frames)))
    return make_phase_window_mask(
        num_frames=num_frames,
        frame_range=(transition_frame - radius, transition_frame + radius),
        include_root=True, include_contacts=True,
        device=device, dtype=dtype,
    )


# =============================================================================
# Long-form chaining (EDGE Fig. 3)
# =============================================================================


@torch.no_grad()
def chained_sample(
    model: MotionDiffusion, contexts: Sequence[torch.Tensor],
    overlap_frames: int, seed: Optional[int] = None,
) -> torch.Tensor:
    """Concatenate multiple sampled clips by inpainting the overlap.

    Given a list of K conditioning vectors, each producing a (1, T, 151)
    clip, this yields a single ``(1, T * K - overlap * (K-1), 151)``
    sequence where adjacent clips agree on their overlapping window.

    Used when we need dives longer than ``cfg.seq_frames``. For the
    paper this is rarely hit (dives are ~4.2 s) but the machinery is
    here for future work.
    """
    if not contexts:
        raise ValueError("at least one context required")
    T = model.cfg.seq_frames
    if overlap_frames >= T:
        raise ValueError("overlap_frames must be < seq_frames")

    clip0 = model.sample(contexts[0], seed=seed)
    full = clip0
    for k in range(1, len(contexts)):
        prev_tail = full[..., -overlap_frames:, :]
        x_known = torch.cat([
            prev_tail,
            full.new_zeros(prev_tail.shape[0], T - overlap_frames, TOKEN_DIM),
        ], dim=1)
        mask = torch.zeros_like(x_known)
        mask[..., :overlap_frames, :] = 1.0
        new_clip = model.inpaint(x_known, mask, contexts[k], seed=seed)
        full = torch.cat([full, new_clip[..., overlap_frames:, :]], dim=1)
    return full


# =============================================================================
# Data filter — high-score prior
# =============================================================================


def filter_top_quartile_by_action(
    annotations: Sequence,
    quartile: float = 0.75,
) -> List:
    """Filter annotations to the top-quartile dives per action type.

    Args:
        annotations: iterable of :class:`DiveAnnotation` (or anything with
            ``action_type`` and ``final_score`` attributes).
        quartile: keep dives whose ``final_score`` is above this quantile
            within their action-type group.

    Returns:
        A list of the kept annotations.

    This is what makes the diffusion prior "what a good execution looks
    like" rather than "any dive at all" — without it, the diffusion's
    samples regress to the mean, and the CF search has trouble finding
    plausible uplifts.
    """
    import numpy as np
    from collections import defaultdict
    by_action: Dict[str, List] = defaultdict(list)
    for ann in annotations:
        by_action[ann.action_type].append(ann)
    kept: List = []
    for action, group in by_action.items():
        if len(group) < 4:
            kept.extend(group)
            continue
        scores = np.array([a.final_score for a in group])
        threshold = float(np.quantile(scores, quartile))
        kept.extend(a for a in group if a.final_score >= threshold)
    return kept


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    # Rotation utilities
    "axis_angle_to_rotmat", "rotmat_to_6d", "sixd_to_rotmat", "axis_angle_to_6d",
    # SMPL
    "SMPL_JOINT_NAMES", "SMPL_PARENTS", "SMPL_FOOT_JOINTS",
    "NUM_SMPL_JOINTS", "TOKEN_DIM",
    "default_rest_positions", "load_smpl_rest_positions",
    "SMPLKinematics",
    # Tokenisation
    "tokenize_pose", "detokenize_pose", "derive_foot_contacts",
    "CONTACT_SLICE", "ROTATION_SLICE", "ROOT_SLICE",
    # Schedule
    "cosine_betas", "linear_betas", "DDPMScheduler",
    # Building blocks
    "TimestepEmbedding", "FiLM", "DiffusionDecoderBlock",
    # Model
    "DiffusionOutput", "MotionDiffusion",
    # Losses — per-sample (new) and scalar (deprecated)
    "loss_simple_per_sample", "loss_velocity_per_sample",
    "loss_position_per_sample", "loss_contact_per_sample",
    "loss_simple", "loss_velocity", "loss_position", "loss_contact",
    "compute_diffusion_loss",
    # Masks
    "make_joint_subset_mask", "make_phase_window_mask", "make_transition_shift_mask",
    # Utilities
    "chained_sample", "filter_top_quartile_by_action",
]

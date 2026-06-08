"""configs.py — Frozen dataclass configurations.

Single source of truth for every hyperparameter, path, and protocol choice
in the project. Every run snapshots its resolved ``Cfg`` into
``experiments/<run_name>/config.yaml`` so that any number reported in the
paper can be reproduced from the (config, git-commit) pair recorded next
to its outputs.

Design rules
------------
1. All configs are ``@dataclass(frozen=True)``: they hash, cannot be
   mutated mid-run, and are safe to share across DDP workers.
2. YAML is used for human-readable snapshots; round-tripping
   ``Cfg → YAML → Cfg`` is exact for every field declared here.
3. ``Cfg.seed_everything`` covers Python ``random``, NumPy, PyTorch CPU,
   PyTorch CUDA, the CUDA workspace, and cuDNN — the five places where
   non-determinism leaks from in our stack.
4. Paths are stored as ``pathlib.Path``; relative paths resolve against
   ``DataCfg.repo_root`` at access time, so YAML stays portable across
   machines.
5. Numeric counts that must agree across modules (SMPL joint count,
   number of step transitions, FPS, ...) live here once and are imported
   everywhere — never re-declared.
6. Every non-obvious value cites its source: paper equation, reference
   repository file:line, or design rationale. The provenance is
   machine-reproducible.

Provenance of the published recipes
-----------------------------------
- FineDiving AQA: Xu et al., CVPR 2022. Reference code at
  https://github.com/xujinglin/FineDiving. Key files verified against:
  ``tools/builder.py`` (optimiser), ``tools/helper.py`` (loss & curriculum),
  ``models/PS.py`` (procedure segmentation), ``FineDiving_TSA.yaml``
  (hyperparameters).
- STSA (FineDiving+): Xu et al., IJCV 2024. Loss weights {α₁,α₂,α₃} =
  {10,1,1} (Eq. 13). NOT the default here; we follow the CVPR'22 protocol
  which uses unweighted addition. Switch to STSA via ``AQACfg.bce_weight=10``.
- EDGE diffusion: Tseng et al., CVPR 2023. Reference code at
  https://github.com/Stanford-TML/EDGE. Loss weights and EMA decay verified
  against ``model/diffusion.py:63`` and ``model/diffusion.py:514-519``.
- DiME diffusion-CF: Jeanneret et al., ACCV 2022. The ``noise_level_tau``
  and ``dime_gradient_scaling`` knobs implement their Eq. 8 and
  late-start trick.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np
import torch
import yaml

T = TypeVar("T")


# =============================================================================
# DataCfg
# =============================================================================


@dataclass(frozen=True)
class DataCfg:
    """Datasets, splits, frame sampling, exemplar selection."""

    # Repository root — resolved at construction so YAML stays portable.
    repo_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    # ---- Annotation roots ------------------------------------------------
    finediving_anno_root: Path = Path("data/annotations/finediving")
    finediving_frames_root: Path = Path("data/frames/finediving")
    # FineDiving+ (Xu et al., IJCV 2024). Superset of FineDiving; used for
    # cross-scenario / cross-discipline / cross-height splits.
    finediving_plus_anno_root: Path = Path("data/annotations/finediving_plus")
    # MTL-AQA (Parmar & Tran Morris, CVPR 2019). Cross-dataset AQA only —
    # it lacks step-level boundaries, so cannot host counterfactual or
    # attribution evaluation.
    mtl_aqa_anno_root: Path = Path("data/annotations/mtl_aqa")
    # Pre-extracted 3D SMPL poses (one .npz per dive). Built offline because
    # PHALP's compile chain is incompatible with our deployment environment
    # (CentOS 7, GCC 4.8.5).
    pose_root: Path = Path("data/poses")
    # Half-point fault annotations — the contribution of this paper.
    faults_root: Path = Path("data/faults")
    faults_file: str = "finediving_hf.json"
    # 0.9% of clips failed pose extraction (water-spray occlusion at entry);
    # excluded from pose-conditioned experiments and reported separately.
    pose_failed_list: Path = Path("data/poses/failed.txt")

    # ---- Frame filename pattern (verified against reference repo) -------
    # The FineDiving release uses 5-digit zero-padded frame names
    # (``00001.jpg``); the data_preparation/data_process.py script writes
    # ``%05d.jpg``. Keep configurable so the same loader works with
    # ``%08d.jpg`` if the user re-extracts with a different ffmpeg pattern.
    frame_filename_pattern: str = "{:05d}.jpg"

    # ---- Frame sampling (matches the FineDiving baseline exactly) -------
    # (num_snippets - 1) * stride + snippet_len covers num_frames frames.
    # Verified against ``FineDiving_TSA.yaml`` (frame_length: 96) and
    # ``tools/helper.py`` (9 snippets, stride 10, snippet_len 16).
    num_frames: int = 96
    num_snippets: int = 9
    snippet_len: int = 16
    snippet_stride: int = 10

    # ---- Image transforms (verified against reference builder.py) -------
    # The reference uses ``Resize((200,112)) → RandomCrop(112)`` at train and
    # ``Resize((200,112)) → CenterCrop(112)`` at eval, giving a 112×112
    # input to I3D. Despite I3D being Kinetics-pretrained at 224, the
    # AQA literature uses 112 to keep the 96-frame batch tractable.
    # Switching to 224 is a non-published configuration.
    image_resize_hw: Tuple[int, int] = (200, 112)  # (height, width); see Resize((H,W))
    image_size: int = 112                          # final crop size
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Optional ±N-frame jitter at training time. The reference repository
    # does NOT jitter — sampling is deterministic ``np.linspace`` for both
    # train and eval. Keep configurable for ablation; default off to match.
    frame_index_jitter: int = 0

    # ---- Step decomposition (FineDiving paper, Sec. 4.2) ---------------
    num_step_transitions: int = 2   # L
    frames_per_step: int = 5        # T_l; "fix_size" in the reference YAML
    num_steps: int = 3              # L + 1

    # ---- Pose representation (SMPL, axis-angle per joint) --------------
    pose_fps: int = 30
    pose_joint_count: int = 24
    pose_rotation_dim: int = 3
    pose_root_dim: int = 3

    # ---- Splits ---------------------------------------------------------
    # Standard FineDiving 75 / 25 train / test, with a deterministic 10%
    # of train held out for validation. The val split seed is fixed so
    # the same dives are held out across re-runs and re-clones. Holdout
    # is stratified by action_type — see ``data._stratified_val_holdout``.
    val_fraction_of_train: float = 0.10
    val_split_seed: int = 0
    val_stratify_by_action_type: bool = True

    # ---- Score normalisation (R-ℓ2 only; not Spearman) -----------------
    score_min: float = 0.0
    score_max: float = 114.85   # FineDiving+ maximum recorded score (IJCV §4)

    # ---- Judges' execution scores --------------------------------------
    num_judges: int = 7                  # FINA standard
    judge_trim_top_bottom: int = 1       # drop top-1 and bottom-1 before sum
    # The released FineDiving pickle stores final_score directly but does
    # not always carry individual judges' scores. When absent, our loaders
    # leave ``DiveAnnotation.judges_scores`` empty; FineDiving-HF supplies
    # them in its own JSON. Set this False if you have no judges' scores
    # and want loaders to skip the seven-value check.
    require_judges_scores: bool = False

    # ---- Exemplar selection --------------------------------------------
    # Shared by AQA contrastive regression and the diffusion "high-score
    # prior". Action-type conditioning is the default per the FineDiving
    # paper (their "w/ DN" setting); this is required to reach the
    # published ρ ≈ 0.92 — without it, ρ drops to ≈ 0.89.
    exemplar_strategy: str = "action_type"     # action_type | difficulty | random
    exemplar_voting_M: int = 10                # inference; training uses 1
    exemplar_difficulty_window: float = 0.2    # used as primary or fallback
    # When the primary strategy returns an empty pool (e.g. cross-height
    # transfer: some action types are platform-only), fall back to this
    # secondary strategy. Set to "raise" to reproduce the strict behaviour.
    exemplar_fallback: str = "difficulty"      # difficulty | random | raise

    # ---- DataLoader -----------------------------------------------------
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # ---------------------------------------------------------------------

    def __post_init__(self) -> None:
        expected = (self.num_snippets - 1) * self.snippet_stride + self.snippet_len
        if expected != self.num_frames:
            raise ValueError(
                f"num_frames={self.num_frames} inconsistent with "
                f"(num_snippets={self.num_snippets}-1)*stride={self.snippet_stride}"
                f"+snippet_len={self.snippet_len}={expected}"
            )
        if self.num_steps != self.num_step_transitions + 1:
            raise ValueError("num_steps must equal num_step_transitions + 1")
        if self.exemplar_strategy not in {"action_type", "difficulty", "random"}:
            raise ValueError(f"unknown exemplar_strategy: {self.exemplar_strategy}")
        if self.exemplar_fallback not in {"difficulty", "random", "raise"}:
            raise ValueError(f"unknown exemplar_fallback: {self.exemplar_fallback}")
        if not 0.0 < self.val_fraction_of_train < 1.0:
            raise ValueError("val_fraction_of_train must be in (0, 1)")
        if self.exemplar_voting_M < 1:
            raise ValueError("exemplar_voting_M must be >= 1")
        if self.num_judges < 3 or self.num_judges % 2 == 0:
            raise ValueError("num_judges should be odd and >= 3 (FINA: 7)")
        if self.frame_index_jitter < 0:
            raise ValueError("frame_index_jitter must be >= 0")
        if not self.frame_filename_pattern.endswith(".jpg"):
            raise ValueError(
                "frame_filename_pattern must end in .jpg "
                f"(got {self.frame_filename_pattern!r})"
            )
        if len(self.image_resize_hw) != 2 or any(s <= 0 for s in self.image_resize_hw):
            raise ValueError(
                f"image_resize_hw must be a (H,W) pair of positive ints; "
                f"got {self.image_resize_hw}"
            )

    def path(self, p: Path | str) -> Path:
        """Resolve a possibly-relative path against ``repo_root``."""
        p = Path(p)
        return p if p.is_absolute() else self.repo_root / p


# =============================================================================
# AQACfg
# =============================================================================


@dataclass(frozen=True)
class AQACfg:
    """The frozen-after-training oracle: I3D + TSA contrastive regression.

    Reproduces the FineDiving paper's numbers within a percentage point;
    once trained, it is frozen and re-used as the differentiable oracle
    for counterfactual evaluation.

    The training recipe matches Xu et al. (CVPR 2022) precisely: I3D
    Kinetics-pretrained at LR=1e-4, head modules at LR=1e-3, Adam with
    weight_decay=0, BCE+MSE loss with equal weight, M=10 exemplar voting,
    and a 25%-of-epochs curriculum where the segmentation head is
    bootstrapped using ground-truth step transitions before switching to
    its own predictions.
    """

    # I3D backbone (Carreira & Zisserman, CVPR 2017), Kinetics-400 pretrained.
    i3d_pretrained: Path = Path("data/checkpoints/i3d_kinetics.pth")
    i3d_feature_dim: int = 1024

    # Procedure-segmentation module S (Eq. 2-3, FineDiving paper).
    # Down-up sub-block dims: spatial dim drops, temporal dim grows.
    segmentation_spatial_dims: Tuple[int, ...] = (1024, 512, 256, 128)
    segmentation_temporal_dims: Tuple[int, ...] = (12, 24, 48, 96)
    # The reference repo's PSNet (models/PS.py) has a different semantic
    # interpretation: 9 snippets are CHANNELS, 1024 features are LENGTH,
    # MaxPool reduces length 1024 → 64. Set this True to use that variant
    # instead. See aqa_model.py for both implementations.
    segmentation_use_reference_psnet: bool = True

    # Procedure-aware cross-attention decoder (Eq. 5-6).
    # decoder_layers=3 matches reference builder.py: decoder_fuser(num_layers=3).
    decoder_layers: int = 3
    decoder_heads: int = 8
    decoder_dim_ff: int = 2048
    decoder_dropout: float = 0.1

    # b2 head activation. IJCV STSA §5.2 specifies ReLU; the reference
    # PS_parts.py uses ReLU. Use GELU only for ablation.
    b2_activation: str = "relu"   # relu | gelu

    # Fine-grained contrastive regressor (three-layer MLP; Eq. 7).
    regressor_hidden: Tuple[int, ...] = (256, 128)
    regressor_dropout: float = 0.1

    # ---- Loss weights ---------------------------------------------------
    # CVPR'22 (FineDiving) Eq. 9: J = Σ BCE + MSE (unweighted; the reference
    # ``tools/helper.py`` line 165 computes ``loss = loss_aqa + loss_tas``).
    # IJCV'24 (STSA) Eq. 13 uses {α₁,α₂,α₃} = {10,1,1}; if you target the
    # STSA recipe, set bce_weight=10.0 and sma_kl_weight=1.0.
    bce_weight: float = 1.0
    mse_weight: float = 1.0
    # KL term on the SMA proxy (STSA only). The reference paper SUMS the
    # query and exemplar branches; our code averages, so this weight is on
    # the *averaged* KL — multiply by 2 to match the paper's α₃ exactly.
    sma_kl_weight: float = 1.0
    sma_kl_aggregation: str = "mean"  # mean | sum — "sum" matches STSA Eq. 6 literally

    # ---- Symmetric training (FineDiving 2022, Table 3) ------------------
    # F+R⋆ (asymmetric, single direction) underperforms F+R (symmetric,
    # both directions). Reference helper.py:145-164 trains symmetrically.
    # Setting this to False enables the asymmetric ablation.
    use_symmetric_training: bool = True

    # ---- Step-transition curriculum (reference helper.py:65) ------------
    # For the first ``curriculum_threshold_epochs`` epochs, slice features
    # using ground-truth step transitions rather than the segmentation
    # head's predictions. This avoids the chicken-and-egg between an
    # untrained segmenter and an untrained regressor. The reference YAML
    # uses prob_tas_threshold=0.25 with max_epoch=200 → 50 epochs.
    curriculum_threshold_epochs: int = 50

    # ---- Frozen-oracle settings ----------------------------------------
    # Once trained the scorer is frozen; this knob lets the diffusion-CF
    # sampler request gradient pass-through without polluting training.
    freeze_after_training: bool = True

    # Calibration sanity check: faithfulness MAE on a held-out slice. If
    # the frozen oracle's claimed Δ-score disagrees with re-evaluation by
    # more than this, the CF pipeline aborts loudly rather than producing
    # unreliable attributions.
    calibration_mae_max: float = 0.25

    # Checkpoint path written by train.py and read by evaluate.py / cf.
    checkpoint_path: Path = Path("experiments/aqa/best.pt")

    def __post_init__(self) -> None:
        if self.b2_activation not in {"relu", "gelu"}:
            raise ValueError(f"b2_activation must be relu|gelu; got {self.b2_activation}")
        if self.sma_kl_aggregation not in {"mean", "sum"}:
            raise ValueError(
                f"sma_kl_aggregation must be mean|sum; got {self.sma_kl_aggregation}"
            )
        if self.curriculum_threshold_epochs < 0:
            raise ValueError("curriculum_threshold_epochs must be >= 0")


# =============================================================================
# DiffusionCfg
# =============================================================================


@dataclass(frozen=True)
class DiffusionCfg:
    """Pose-conditioned motion diffusion in the EDGE lineage (CVPR 2023).

    Trained on the top-quartile dives per dive type so the prior is "what
    a good execution of this specific dive looks like." Masked inpainting
    (m ⊙ q(x_known) + (1-m) ⊙ ẑ) is the editing primitive that supports
    all three intervention modalities used by the counterfactual search:
    joint-subset, phase-window, and transition-timestamp masks.

    All numerical defaults verified against the EDGE released codebase at
    https://github.com/Stanford-TML/EDGE (``model/diffusion.py``).
    """

    # ---- Sequence representation ----------------------------------------
    # EDGE pose token: 24 SMPL joints × 6D rotation + 3D root + 4 contacts.
    # We follow the same 151-dim convention.
    seq_seconds: float = 5.0
    seq_fps: int = 30
    seq_frames: int = 150
    pose_repr_dim: int = 24 * 6 + 3   # 147: 6D-rot joints + root translation
    contact_dim: int = 4              # heel/toe per foot
    token_dim: int = 24 * 6 + 3 + 4   # 151

    # ---- Transformer decoder --------------------------------------------
    width: int = 512
    depth: int = 8
    heads: int = 8
    dim_ff: int = 1024
    dropout: float = 0.1

    # ---- DDPM noising schedule ------------------------------------------
    # EDGE paper says "a monotonically decreasing schedule"; we use cosine
    # following Nichol & Dhariwal 2021 (Improved DDPM). DDPM (Ho 2020)
    # itself uses linear β.
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine"          # cosine | linear
    cosine_s: float = 0.008                  # Nichol & Dhariwal 2021, §3
    # EDGE predicts the clean sample x̂₀, not the noise ε (DDPM default).
    # Reference ``model/diffusion.py`` calls this ``predict_epsilon=False``.
    predict_x0: bool = True
    # Reverse-step variance choice. β_t (DDPM upper bound) and β̃_t (lower
    # bound) give similar results in the original DDPM ablation. EDGE
    # codebase uses β̃_t implicitly.
    posterior_variance: str = "beta_tilde"  # beta | beta_tilde

    # ---- Classifier-free guidance ---------------------------------------
    # Train with 25% unconditional drop (EDGE §3.2), sample with w > 1.
    cf_drop_prob: float = 0.25
    guidance_weight_default: float = 2.0    # w in EDGE; their headline value

    # ---- Conditioning ---------------------------------------------------
    # For this paper we condition on (action-type embedding, query summary).
    # No Jukebox features (no music here), but we keep the cross-attention
    # interface intact for parity with EDGE.
    cond_action_embedding_dim: int = 128
    cond_summary_dim: int = 256

    # ---- Auxiliary losses (EDGE Eq. 3-5) --------------------------------
    # Verified against EDGE-main/model/diffusion.py:514-519 — these are the
    # values committed to in the released codebase, not in the paper text.
    lambda_simple: float = 0.636
    lambda_pos: float = 0.646        # FK position loss
    lambda_vel: float = 2.964        # velocity in token space
    lambda_contact: float = 10.942   # Contact Consistency Loss (CCL)

    # ---- p2 loss weighting (EDGE-main/model/diffusion.py:125-130) ------
    # Per-timestep weighting by SNR; down-weights early/late timesteps.
    # The EDGE codebase enables this; small effect.
    p2_loss_weighting: bool = True
    p2_loss_weight_k: float = 1.0
    p2_loss_weight_gamma: float = 0.5

    # ---- Foot contact derivation threshold -----------------------------
    # Used during training data preparation to derive 4-binary foot contacts
    # from raw joint velocities. EDGE-codebase value, not in the paper.
    contact_velocity_threshold: float = 0.05

    # ---- High-score prior: train on top quartile per dive type ---------
    high_score_quartile: float = 0.75

    # ---- Long-form sampling (EDGE Fig. 3 chaining) ----------------------
    # Diving clips are ~4.2 s, so this code path almost never fires. We
    # use inpainting at the seam; EDGE uses a linear cross-fade. The
    # divergence is documented in the paper's limitations.
    overlap_seconds: float = 2.5
    overlap_frames: int = 75

    # ---- Optimisation ---------------------------------------------------
    grad_checkpoint: bool = True             # 24 GB VRAM on 3090

    # ---- Output ---------------------------------------------------------
    checkpoint_path: Path = Path("experiments/diffusion/best.pt")

    def __post_init__(self) -> None:
        if self.noise_schedule not in {"cosine", "linear"}:
            raise ValueError(f"noise_schedule must be cosine|linear; got {self.noise_schedule}")
        if self.posterior_variance not in {"beta", "beta_tilde"}:
            raise ValueError(
                f"posterior_variance must be beta|beta_tilde; got {self.posterior_variance}"
            )
        if not 0.0 <= self.cf_drop_prob <= 1.0:
            raise ValueError("cf_drop_prob must be in [0, 1]")
        if self.token_dim != self.pose_repr_dim + self.contact_dim:
            raise ValueError(
                f"token_dim={self.token_dim} != pose_repr_dim+contact_dim="
                f"{self.pose_repr_dim + self.contact_dim}"
            )


# =============================================================================
# CFCfg — the central algorithm of this paper
# =============================================================================


@dataclass(frozen=True)
class CFCfg:
    """Half-point counterfactual search hyperparameters.

    Given (query dive, frozen AQA, diffusion prior), search over intervention
    supports {phase, joint group, timing window} for the minimum-norm pose
    edit that uplifts AQA by ≥ 0.5 under PFC and minimality constraints.
    The half-point is the atomic award unit in FINA execution scoring and
    is therefore the natural quantum of counterfactual attribution.

    Three of the knobs below ship straight out of DiME (Jeanneret et al.
    ACCV 2022): ``noise_level_tau`` (their late-start trick),
    ``dime_gradient_scaling`` (their Eq. 8 with 1/√ᾱ_t scaling), and
    ``iterative_guidance_weights`` (their λ_c escalation list).
    """

    # ---- Uplift target --------------------------------------------------
    uplift_target: float = 0.5      # one half-point
    uplift_tolerance: float = 0.10  # accept [0.40, 0.75] as "one half-point"
    uplift_max: float = 0.75        # reject overshoots — atoms only

    # ---- Plausibility constraints --------------------------------------
    # We report TWO PFC variants: EDGE-verbatim (Eqs. 11-12) as the primary
    # metric, plus our diving-adapted "airborne lateral acceleration" as a
    # secondary domain-specific signal. The EDGE formula is the one cited
    # for comparison against published numbers.
    pfc_metric: str = "edge"        # edge | lateral_accel | both
    pfc_threshold: float = 2.0
    discriminator_threshold: float = 0.5

    # ---- Minimality cap ------------------------------------------------
    # L2 norm of the intervention in pose space (sum across joints, frames).
    minimality_norm_cap: float = 5.0

    # ---- Intervention modalities ---------------------------------------
    intervention_modalities: Tuple[str, ...] = (
        "joint_subset",        # mask a joint group across all frames of a step
        "phase_window",        # mask all joints in a single phase
        "transition_timestamp" # shift a step boundary by a few frames
    )
    # The joint-group lexicon is defined in attribution.py; this is just a
    # length for budget sizing here, so CFCfg stays free of the taxonomy.
    num_joint_groups: int = 8

    # ---- Fault gating by dive type (NSAQA-style) -----------------------
    # NSAQA's Algorithm 1 skips knee-straightness microprograms on tuck
    # dives because tuck explicitly bends the knee. We apply the same
    # gating in attribution; the matrix lives in attribution.py.
    enforce_fault_dive_gating: bool = True

    # ---- Search budget per half-point peel-off -------------------------
    search_max_candidates: int = 32     # (phase × joint-group) combinations
    search_inpaint_steps: int = 50      # DDIM steps for guided sampling
    search_guidance_weight: float = 3.0
    score_grad_scale: float = 1.0       # weight on AQA-score gradient term

    # ---- DiME knobs (Jeanneret et al., ACCV 2022) ----------------------
    # Late-start: initialise z from the query at intermediate τ rather than
    # from pure noise. DiME uses τ = 60/200 = 0.3. Default 1.0 = pure noise
    # (no late-start). Lower values give tighter, more conservative CFs.
    noise_level_tau: float = 1.0
    # Use DiME's Eq. 8 (1/√ᾱ_t) gradient scaling instead of autograd
    # through one denoising step. The autograd path is mathematically
    # equivalent but scale-fragile across τ; the DiME path is more stable
    # when you sweep τ during ablations.
    dime_gradient_scaling: bool = False
    # Iterative λ_c escalation: DiME tries {8, 10, 15} before declaring
    # failure on a candidate. Cheaper than a hyperparameter sweep.
    iterative_guidance_weights: Tuple[float, ...] = (3.0, 5.0, 8.0)

    # ---- RePaint-style resampling (Lugmayr et al., CVPR 2022) -----------
    # 0 = disabled (single-pass inpainting, EDGE-style). >0 enables R
    # inner resample loops over the final fraction of denoising steps.
    inpaint_resamples: int = 0
    inpaint_resample_fraction: float = 0.2   # last 20% of steps

    # ---- Outer decomposition -------------------------------------------
    # Peel half-points off the score gap iteratively. Stop when the
    # residual gap is below residual_threshold, or when we hit the cap.
    max_peeled_half_points: int = 12    # 6-point gap is the realistic upper bound
    residual_threshold: float = 0.25

    def __post_init__(self) -> None:
        if self.pfc_metric not in {"edge", "lateral_accel", "both"}:
            raise ValueError(f"pfc_metric must be edge|lateral_accel|both; got {self.pfc_metric}")
        if not 0.0 < self.noise_level_tau <= 1.0:
            raise ValueError("noise_level_tau must be in (0, 1]")
        if self.inpaint_resamples < 0:
            raise ValueError("inpaint_resamples must be >= 0")
        if not 0.0 < self.inpaint_resample_fraction <= 1.0:
            raise ValueError("inpaint_resample_fraction must be in (0, 1]")
        if not self.iterative_guidance_weights:
            raise ValueError("iterative_guidance_weights must be non-empty")


# =============================================================================
# EvalCfg
# =============================================================================


@dataclass(frozen=True)
class EvalCfg:
    """Metric configuration and expert-study scaffolding.

    The counterfactual evaluation columns use the CF-XAI canonical
    vocabulary (validity, sparsity, proximity, realism). The mapping from
    these to your internal pipeline:
       validity   ← faithfulness (was: did the CF reach the target uplift)
       sparsity   ← #joint groups changed
       proximity  ← pose ℓ2 distance to query
       realism    ← PFC + diffusion-prior likelihood
       completion ← (your contribution) all peeled half-points explained
    """

    # ---- AQA -----------------------------------------------------------
    spearman_bootstrap: int = 1000
    r_l2_score_normaliser: str = "dataset"   # dataset | per_action_type

    # ---- Physical plausibility -----------------------------------------
    # When pfc_metric is "both" in CFCfg, we report both forms; this
    # determines which one drives go/no-go for CF acceptance.
    pfc_primary_for_acceptance: str = "edge"  # edge | lateral_accel

    # ---- Counterfactual metrics ----------------------------------------
    cf_faithfulness_tol: float = 0.10
    cf_minimality_metric: str = "pose_l2"    # pose_l2 | joint_arc_length

    # ---- Attribution ---------------------------------------------------
    attribution_topk: Tuple[int, ...] = (1, 3)
    # When True, attribution F1 is computed with NSAQA-style fault-gating
    # (knee CFs on tuck dives are dropped). Off would inflate F1 on
    # rare-but-meaningless predictions.
    attribution_apply_dive_gating: bool = True

    # ---- Expert study (FineDiving-HF user study, after ExpertAF) -------
    # Likert 1-4, 5 raters per dive, each output rated independently —
    # see ExpertAF (Ashutosh et al. CVPR 2025) §4.
    expert_study_csv: Path = Path("experiments/expert_study/ratings.csv")
    expert_study_raters_per_dive: int = 5
    expert_study_dives_per_rater: int = 50
    expert_study_likert_max: int = 4
    expert_study_seed: int = 0

    def __post_init__(self) -> None:
        if self.pfc_primary_for_acceptance not in {"edge", "lateral_accel"}:
            raise ValueError(
                f"pfc_primary_for_acceptance must be edge|lateral_accel; "
                f"got {self.pfc_primary_for_acceptance}"
            )


# =============================================================================
# TrainCfg
# =============================================================================


@dataclass(frozen=True)
class TrainCfg:
    """Optimiser, schedule, DDP, checkpointing, and the master seed."""

    # ---- AQA optimisation ----------------------------------------------
    # Reference builder.py: optimiser=Adam, weight_decay=0, base_lr=1e-3,
    # lr_factor=0.1 → backbone LR=1e-4, head LR=1e-3.
    aqa_optimizer: str = "adam"
    aqa_lr_backbone: float = 1e-4
    aqa_lr_head: float = 1e-3
    aqa_weight_decay: float = 0.0
    aqa_epochs: int = 200                    # reference YAML: max_epoch=200
    aqa_batch_size_per_gpu: int = 4          # 16 across 4× 3090

    # ---- Diffusion optimisation ---------------------------------------
    diffusion_optimizer: str = "adamw"
    diffusion_lr: float = 4e-4
    diffusion_weight_decay: float = 0.02
    diffusion_warmup_steps: int = 1000
    diffusion_grad_clip: float = 1.0
    diffusion_epochs: int = 2000
    diffusion_batch_size_per_gpu: int = 128  # 512 across 4× 3090
    # EDGE codebase model/diffusion.py:63 uses EMA(0.9999) — *not* 0.999.
    # At 2000 epochs the difference is ~10× in effective averaging window.
    diffusion_ema_decay: float = 0.9999

    # ---- Distributed --------------------------------------------------
    nproc_per_node: int = 4
    ddp_backend: str = "nccl"
    find_unused_parameters: bool = False

    # ---- Mixed precision ----------------------------------------------
    amp_dtype: str = "bf16"                  # 3090 supports bf16 autocast

    # ---- Logging / checkpointing --------------------------------------
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 5
    keep_top_k_checkpoints: int = 3

    # ---- The master seed (used by Cfg.seed_everything) ----------------
    seed: int = 0

    def __post_init__(self) -> None:
        if self.aqa_optimizer not in {"adam", "adamw", "sgd"}:
            raise ValueError(f"unknown aqa_optimizer: {self.aqa_optimizer}")
        if self.diffusion_optimizer not in {"adam", "adamw"}:
            raise ValueError(f"unknown diffusion_optimizer: {self.diffusion_optimizer}")
        if self.amp_dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"amp_dtype must be fp32|fp16|bf16; got {self.amp_dtype}")
        if not 0.0 < self.diffusion_ema_decay < 1.0:
            raise ValueError("diffusion_ema_decay must be in (0, 1)")


# =============================================================================
# Top-level Cfg
# =============================================================================


@dataclass(frozen=True)
class Cfg:
    """The composed run configuration.

    Instantiate, optionally override fields via :func:`dataclasses.replace`,
    then call :meth:`snapshot` at the start of a run to persist config
    + git + CUDA provenance next to the outputs.
    """

    run_name: str = "default"
    data: DataCfg = field(default_factory=DataCfg)
    aqa: AQACfg = field(default_factory=AQACfg)
    diffusion: DiffusionCfg = field(default_factory=DiffusionCfg)
    cf: CFCfg = field(default_factory=CFCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    # ---- Reproducibility ----------------------------------------------

    def seed_everything(self) -> None:
        """Seed every RNG we touch.

        Covers the five places non-determinism leaks in our stack:
        Python ``random``, NumPy, PyTorch CPU, PyTorch CUDA, and cuDNN /
        CUDA workspace. Worker-side seeding is handled by ``data.py``'s
        ``seed_worker`` so that DataLoader workers are also deterministic.
        """
        seed = self.train.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Tight CUDA workspace determinism (small perf hit, large repro win).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            # PyTorch >= 1.11 accepts warn_only; older versions don't.
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)

    def fingerprint(self) -> str:
        """Stable 12-char hash of the resolved config — useful for naming."""
        payload = json.dumps(
            _to_serialisable(asdict(self)), sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    # ---- (De)serialisation -------------------------------------------

    def to_yaml(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(
                _to_serialisable(asdict(self)),
                f,
                sort_keys=False,
                default_flow_style=False,
            )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Cfg":
        with Path(path).open("r") as f:
            data = yaml.safe_load(f)
        return _from_dict(cls, data)  # type: ignore[return-value]

    # ---- Snapshotting -------------------------------------------------

    def snapshot(self, run_dir: Path | str) -> Path:
        """Write config + git + CUDA provenance into ``run_dir``.

        Returns the run directory. After this point, every figure/table
        produced in ``run_dir`` is traceable to a (config, commit) pair.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.to_yaml(run_dir / "config.yaml")
        provenance: Dict[str, Any] = {
            "run_name": self.run_name,
            "fingerprint": self.fingerprint(),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "python_version": _python_version(),
        }
        with (run_dir / "provenance.yaml").open("w") as f:
            yaml.safe_dump(provenance, f, sort_keys=False)
        return run_dir


# =============================================================================
# Helpers
# =============================================================================


def _to_serialisable(obj: Any) -> Any:
    """Recursively coerce Paths and tuples to YAML-friendly types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_to_serialisable(x) for x in obj]
    if isinstance(obj, list):
        return [_to_serialisable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_serialisable(v) for k, v in obj.items()}
    return obj


def _from_dict(cls: Type[T], data: Mapping[str, Any]) -> T:
    """Reconstruct a (possibly nested) dataclass from a dict.

    Resolves ``from __future__ import annotations`` string types via
    ``typing.get_type_hints``, recurses into nested dataclasses, and
    coerces lists back to tuples where the field type demands it.
    """
    if not is_dataclass(cls):
        return data  # type: ignore[return-value]
    type_hints = typing.get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue  # let the dataclass default kick in
        kwargs[f.name] = _coerce(data[f.name], type_hints.get(f.name, f.type))
    return cls(**kwargs)  # type: ignore[call-arg]


def _coerce(v: Any, ftype: Any) -> Any:
    """Coerce a YAML-loaded value to the dataclass field's declared type."""
    if v is None:
        return None
    if is_dataclass(ftype):
        return _from_dict(ftype, v)
    if ftype is Path:
        return Path(v)
    origin = typing.get_origin(ftype)
    if origin is tuple:
        return tuple(v)
    if origin is list:
        return list(v)
    return v


def _git_commit() -> str:
    """Return the current git commit SHA, or ``'unknown'`` if not a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _python_version() -> str:
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


__all__ = [
    "DataCfg",
    "AQACfg",
    "DiffusionCfg",
    "CFCfg",
    "EvalCfg",
    "TrainCfg",
    "Cfg",
]

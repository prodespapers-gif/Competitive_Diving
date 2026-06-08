"""counterfactual.py — Half-point counterfactual search.

This is the paper's central algorithmic contribution. Given a query dive,
a frozen AQA-style scoring function, and the trained motion diffusion
prior, it iteratively peels half-point counterfactuals off the query's
score gap, producing an ordered list of (intervention, counterfactual,
attributed fault) records that constitutes the coachable explanation.

Conceptual contract
-------------------
- The scoring function ``score_fn: (B, T, 151) → (B,)`` is *injected* by
  the caller. ``counterfactual.py`` stays agnostic about what's behind it
  (frame-AQA + SMPL render, distilled pose-AQA, or anything else). The
  caller is also responsible for picking the *units* — judge-points or
  official score — and for ensuring ``target_uplift`` is in the same units.
- The diffusion model provides two editing primitives: gradient-free
  ``inpaint`` (the default) and ``inpaint_with_score_guidance`` (used
  when ``score_fn`` is differentiable through pose tokens).
- The :class:`Intervention` dataclass is owned by this module; both
  :mod:`attribution` (for fault mapping) and :mod:`experiments` (for
  table reporting) import it from here. :mod:`attribution` uses
  ``TYPE_CHECKING`` to avoid a circular import.

Algorithm
---------
For one half-point peel-off (:func:`search_one_half_point`):
    1. Enumerate candidate intervention supports from
       :func:`enumerate_intervention_candidates`.
    2. Build a (T, 151) mask per candidate; stack into a batch.
    3. Run :meth:`MotionDiffusion.inpaint` (or the score-guided variant)
       once for the whole batch — one diffusion pass, not K of them.
       The four diffusion knobs (noise_level_tau, dime_gradient_scaling,
       inpaint_resamples, inpaint_resample_fraction) flow from CFCfg.
    4. Re-score each counterfactual via ``score_fn``; reject candidates
       outside the uplift / plausibility / minimality envelope.
    5. If no candidate survives at the initial CFG weight, retry at the
       next weight in ``cf_cfg.iterative_guidance_weights``. Stronger
       conditioning pulls outputs more aggressively toward the intervention,
       so higher weights catch the difficult cases that the default weight
       missed. Backwards compatible: a single-entry list reverts to the
       previous one-shot behaviour.
    6. Return the minimum-norm survivor (or ``None`` if every retry's
       envelope was empty).

For the outer decomposition (:func:`decompose`):
    Apply :func:`search_one_half_point` repeatedly, each time starting
    from the previous peel's counterfactual, until the residual score
    gap is below ``CFCfg.residual_threshold`` or the peel cap is hit.

Knob threading (camera-ready)
-----------------------------
The CF search exposes four sampler knobs added to :mod:`diffusion`:

  - ``noise_level_tau`` (DiME late-start): start denoising from
    ``q(z_τT | x_known)`` instead of pure noise. With τ<1.0 the CFs
    stay closer to the original dive (tighter, more conservative).
  - ``dime_gradient_scaling``: take the score gradient w.r.t. ``x̂``
    and apply explicit ``1/√ᾱ_t``; more stable when sweeping τ.
  - ``inpaint_resamples`` (R): RePaint Algorithm 1 inner-loop count
    over the final ``inpaint_resample_fraction`` of steps. R=0 keeps
    the previous single-pass behaviour.

These are read from ``cf_cfg`` at the call site so callers don't need
to pass them through; ablations toggle them by varying the cfg only.

References
----------
- The half-point as atomic unit: FINA Diving Rules; FineDiving-HF
  annotation protocol in this paper's dataset section.
- Inpainting primitive: EDGE (Tseng et al., CVPR 2023) Eq. (8).
- Classifier-guided sampling: Dhariwal & Nichol (NeurIPS 2021).
- DiME late-start and 1/√ᾱ_t scaling: Jeanneret et al., CVPR 2022.
- RePaint resampling: Lugmayr et al., CVPR 2022, Algorithm 1.

Changes from previous version
-----------------------------
- Threading of four new diffusion knobs through both inpaint paths
  (Issues 10, 11, 17). No new function signatures — knobs come from
  cf_cfg so callers don't change.
- Iterative guidance ramp: retry on no-valid-CF at successive weights
  from ``cf_cfg.iterative_guidance_weights``. Backwards compatible.
- :class:`Counterfactual` carries four new diagnostic fields recording
  which knobs produced it (for reproducibility and ablation tables).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .configs import CFCfg, DataCfg
from .data import DiveAnnotation
from .diffusion import (
    MotionDiffusion,
    make_joint_subset_mask, make_phase_window_mask,
    TOKEN_DIM,
)

log = logging.getLogger(__name__)


# =============================================================================
# Public type aliases
# =============================================================================

# (B, T, 151) → (B,). May be differentiable (score-guided) or not.
ScoreFn = Callable[[torch.Tensor], torch.Tensor]
# (B, T, 151) → (B,). PFC-style physical plausibility score.
PlausibilityFn = Callable[[torch.Tensor], torch.Tensor]


# =============================================================================
# Intervention — the search's input parameter, owned here
# =============================================================================


@dataclass(frozen=True)
class Intervention:
    """One intervention support: a (modality, step, frame_range, joints) tuple.

    Frozen so it can be used as a dict key for caching and as an element
    of ``Counterfactual.intervention``. Field semantics by modality:

    - ``joint_subset``: free the SMPL joints in ``joint_groups`` over
      frames in ``frame_range``. The joint-group → SMPL-index lookup is
      done in :mod:`attribution` at mask-construction time.
    - ``phase_window``: free **all** joints over the entire ``frame_range``
      (typically the temporal span of a single step).
    - ``transition_timestamp``: free a small window centred on
      ``frame_range``'s midpoint, letting the diffusion re-time the step
      boundary. ``transition_shift_frames`` carries the sign — negative
      meaning the original transition was *late*, positive meaning *early*.
    """

    modality: str                                 # joint_subset | phase_window | transition_timestamp
    step: int                                     # 1-indexed step within the dive
    frame_range: Tuple[int, int]                  # (t_start, t_end) inclusive-exclusive
    joint_groups: Tuple[str, ...] = ()            # populated for joint_subset
    transition_shift_frames: int = 0              # populated for transition_timestamp

    def __post_init__(self) -> None:
        if self.modality not in ("joint_subset", "phase_window", "transition_timestamp"):
            raise ValueError(f"unknown modality: {self.modality}")
        if self.step < 1:
            raise ValueError("step is 1-indexed; got < 1")
        if self.frame_range[0] >= self.frame_range[1]:
            raise ValueError(f"empty frame_range: {self.frame_range}")
        if self.modality == "joint_subset" and not self.joint_groups:
            raise ValueError("joint_subset modality requires non-empty joint_groups")
        if self.modality != "transition_timestamp" and self.transition_shift_frames != 0:
            raise ValueError("transition_shift_frames only valid for transition_timestamp")


@dataclass
class Counterfactual:
    """A generated counterfactual together with its diagnostics.

    Camera-ready: the four trailing ``*_used`` fields record which
    diffusion-sampler knobs produced this CF. They make the search
    reproducible per-record and let ablation tables read off the
    actually-used settings without re-running anything.
    """

    x_cf: torch.Tensor               # (1, T, 151) the counterfactual pose tokens
    intervention: Intervention
    mask: torch.Tensor               # (T, 151) mask used to generate x_cf
    claimed_uplift: float            # target_uplift the search aimed at
    actual_uplift: float             # score_fn(x_cf) - score_fn(x_query_at_peel)
    minimality_norm: float           # L2 norm of (x_cf - x_query) in the masked region
    plausibility: float              # output of plausibility_fn, lower=more plausible
    faithfulness_passed: bool        # |actual - claimed| within tolerance

    # ---- diagnostic: which knobs produced this CF (Issue 10/11/17) ----
    noise_level_tau_used: float = 1.0
    dime_gradient_scaling_used: bool = False
    inpaint_resamples_used: int = 0
    guidance_weight_used: float = 1.0


@dataclass
class DecompositionRecord:
    """One peeled half-point in the decomposition."""

    peel_index: int                  # 0-indexed peel order
    intervention: Intervention
    counterfactual: Counterfactual
    fault: str                       # FINA fault category from attribution.py
    delta_score: float               # actual_uplift at this peel (alias for counterfactual.actual_uplift)


@dataclass
class DecompositionResult:
    """The full attribution result for one dive."""

    dive_id: str
    target_uplift: float             # total score gap we tried to close
    achieved_uplift: float           # sum of delta_score across records
    initial_score: float             # score_fn(query)
    final_score: float               # score_fn(last counterfactual)
    records: List[DecompositionRecord]
    residual_gap: float              # target - achieved (>0 if incomplete)
    completed: bool                  # True if residual_gap < cfg.residual_threshold
    aborted_reason: Optional[str] = None  # set if the decomposition stopped early

    def to_attribution_tuples(self) -> List[Tuple[int, Tuple[str, ...], str, float]]:
        """Return ``(step, joints, fault, delta)`` tuples for comparing against
        the FineDiving-HF expert annotations."""
        out: List[Tuple[int, Tuple[str, ...], str, float]] = []
        for r in self.records:
            out.append((
                r.intervention.step,
                r.intervention.joint_groups,
                r.fault,
                r.delta_score,
            ))
        return out


# =============================================================================
# Candidate enumeration
# =============================================================================


def enumerate_intervention_candidates(
    annotation: DiveAnnotation,
    cf_cfg: CFCfg,
    joint_groups: Sequence[str],
) -> List[Intervention]:
    """Generate the intervention candidate pool for one peel-off step.

    For each step in the dive procedure:
        - one ``joint_subset`` candidate per joint group in ``joint_groups``,
          spanning the step's full frame range,
        - one ``phase_window`` candidate, freeing the whole step,
        - up to two ``transition_timestamp`` candidates (at the boundaries
          between this step and its neighbours, with both shift signs).

    The set is capped at ``cf_cfg.search_max_candidates``.

    Args:
        annotation: the query dive's :class:`DiveAnnotation`. Provides the
            step transition frame indices and step count.
        cf_cfg: provides the enabled modalities and budget.
        joint_groups: the joint-group lexicon from :mod:`attribution`,
            passed in to avoid the circular import.

    Returns:
        A list of :class:`Intervention` objects. Order is deterministic.
    """
    transitions = annotation.step_transitions
    num_steps = len(annotation.sub_action_types)
    # Step k spans [t_{k-1}, t_k); convention: t_0 = 0, t_L = end_of_clip.
    phase_bounds: List[Tuple[int, int]] = []
    prev = 0
    for t in transitions:
        phase_bounds.append((prev, int(t)))
        prev = int(t)
    phase_bounds.append((prev, annotation.clip_num_frames))

    candidates: List[Intervention] = []
    transition_shift_radius = 3   # frames; ~0.1 s @ 30fps

    for step_idx in range(num_steps):
        t_start, t_end = phase_bounds[step_idx]
        if t_start >= t_end:
            # Degenerate step (segmenter collapsed two transitions); skip.
            continue

        # --- joint_subset: one per joint group ---
        if "joint_subset" in cf_cfg.intervention_modalities:
            for group in joint_groups:
                candidates.append(Intervention(
                    modality="joint_subset",
                    step=step_idx + 1,
                    frame_range=(t_start, t_end),
                    joint_groups=(group,),
                ))

        # --- phase_window: free the entire step ---
        if "phase_window" in cf_cfg.intervention_modalities:
            candidates.append(Intervention(
                modality="phase_window",
                step=step_idx + 1,
                frame_range=(t_start, t_end),
            ))

        # --- transition_timestamp: shift the boundary at the END of this step ---
        # The sign of ``shift`` encodes the direction we want the boundary to
        # move; the frame window is asymmetric to match. Without this, both
        # signs would produce *identical* counterfactuals (the diffusion
        # would see the same symmetric window either way) and downstream
        # attribution couldn't tell early-shift from late-shift.
        if (
            "transition_timestamp" in cf_cfg.intervention_modalities
            and step_idx < num_steps - 1
        ):
            t_boundary = int(transitions[step_idx])
            for shift in (-transition_shift_radius, +transition_shift_radius):
                if shift > 0:
                    # Free frames AFTER the boundary — bias toward LATER transition.
                    window = (
                        t_boundary,
                        min(annotation.clip_num_frames, t_boundary + 2 * shift),
                    )
                else:
                    # Free frames BEFORE the boundary — bias toward EARLIER transition.
                    window = (
                        max(0, t_boundary + 2 * shift),
                        t_boundary,
                    )
                if window[0] >= window[1]:
                    continue
                candidates.append(Intervention(
                    modality="transition_timestamp",
                    step=step_idx + 1,
                    frame_range=window,
                    transition_shift_frames=shift,
                ))

    if len(candidates) > cf_cfg.search_max_candidates:
        log.debug(
            "%s: enumerated %d candidates, capping at %d",
            annotation.dive_id, len(candidates), cf_cfg.search_max_candidates,
        )
        candidates = candidates[: cf_cfg.search_max_candidates]
    return candidates


# =============================================================================
# Intervention → mask
# =============================================================================


def intervention_to_mask(
    intervention: Intervention,
    num_frames: int,
    joint_group_to_indices: Dict[str, Sequence[int]],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the (T, 151) mask corresponding to an :class:`Intervention`.

    ``joint_group_to_indices`` is the lexicon-to-SMPL-index lookup from
    :mod:`attribution`, passed in here to keep the dependency one-way.

    Returns a tensor of 1s (preserve query) and 0s (regenerate).
    """
    if intervention.modality == "joint_subset":
        joint_indices: List[int] = []
        for g in intervention.joint_groups:
            if g not in joint_group_to_indices:
                raise ValueError(
                    f"unknown joint group {g!r}; "
                    f"known: {sorted(joint_group_to_indices)}"
                )
            joint_indices.extend(joint_group_to_indices[g])
        return make_joint_subset_mask(
            num_frames=num_frames, joint_indices=joint_indices,
            frame_range=intervention.frame_range,
            device=device, dtype=dtype,
        )

    if intervention.modality == "phase_window":
        return make_phase_window_mask(
            num_frames=num_frames, frame_range=intervention.frame_range,
            include_root=True, include_contacts=True,
            device=device, dtype=dtype,
        )

    # transition_timestamp: the frame_range stored in the Intervention
    # already encodes the asymmetric directional window (see
    # ``enumerate_intervention_candidates``); ``transition_shift_frames``
    # carries the sign as metadata for downstream attribution.
    return make_phase_window_mask(
        num_frames=num_frames, frame_range=intervention.frame_range,
        include_root=True, include_contacts=True,
        device=device, dtype=dtype,
    )


# =============================================================================
# Inner loop: one half-point peel
# =============================================================================


def _resolve_guidance_schedule(cf_cfg: CFCfg) -> Tuple[float, ...]:
    """Pick the CFG-weight schedule for one peel.

    Returns:
        Tuple of weights to try in order. Falls back to
        ``(cf_cfg.search_guidance_weight,)`` (single attempt, the old
        behaviour) when ``iterative_guidance_weights`` is unset, empty,
        or has only one entry.
    """
    weights = getattr(cf_cfg, "iterative_guidance_weights", None)
    if not weights:
        return (cf_cfg.search_guidance_weight,)
    if isinstance(weights, (int, float)):
        return (float(weights),)
    out = tuple(float(w) for w in weights)
    if not out:
        return (cf_cfg.search_guidance_weight,)
    return out


def _run_inpaint_batch(
    diffusion: MotionDiffusion,
    x_query_batched: torch.Tensor,
    masks: torch.Tensor,
    context_batched: torch.Tensor,
    cf_cfg: CFCfg,
    score_fn: ScoreFn,
    *,
    use_score_guidance: bool,
    guidance_w: float,
    seed: Optional[int],
) -> torch.Tensor:
    """Run one inpaint pass (score-guided or not), threading the four
    diffusion knobs from ``cf_cfg``.

    Returns the (K, T, 151) batch of generated counterfactuals.
    """
    # Camera-ready: read the four new knobs from cf_cfg.
    tau = float(getattr(cf_cfg, "noise_level_tau", 1.0))
    R = int(getattr(cf_cfg, "inpaint_resamples", 0))
    R_frac = float(getattr(cf_cfg, "inpaint_resample_fraction", 0.2))
    dime = bool(getattr(cf_cfg, "dime_gradient_scaling", False))

    if use_score_guidance:
        return diffusion.inpaint_with_score_guidance(
            x_known=x_query_batched, mask=masks, context=context_batched,
            score_fn=score_fn,
            target_uplift=cf_cfg.uplift_target,
            grad_scale=cf_cfg.score_grad_scale,
            num_inference_steps=cf_cfg.search_inpaint_steps,
            guidance_w=guidance_w,
            noise_level_tau=tau,
            dime_gradient_scaling=dime,
            inpaint_resamples=R,
            inpaint_resample_fraction=R_frac,
            seed=seed,
        )
    return diffusion.inpaint(
        x_known=x_query_batched, mask=masks, context=context_batched,
        num_inference_steps=cf_cfg.search_inpaint_steps,
        guidance_w=guidance_w,
        noise_level_tau=tau,
        inpaint_resamples=R,
        inpaint_resample_fraction=R_frac,
        seed=seed,
    )


def search_one_half_point(
    x_query: torch.Tensor,                    # (1, T, 151)
    annotation: DiveAnnotation,
    diffusion: MotionDiffusion,
    score_fn: ScoreFn,
    context: torch.Tensor,                    # (1, cond_dim)
    cf_cfg: CFCfg,
    joint_group_to_indices: Dict[str, Sequence[int]],
    *,
    plausibility_fn: Optional[PlausibilityFn] = None,
    use_score_guidance: bool = False,
    seed: Optional[int] = None,
) -> Optional[Counterfactual]:
    """Search for one half-point counterfactual starting from ``x_query``.

    Args:
        x_query: the current pose token tensor (1, T, 151). For the first
            peel this is the original query dive's pose; for subsequent
            peels it's the previous peel's counterfactual.
        annotation: the query dive's annotation. Used for step boundaries
            (intervention enumeration) but NOT for scoring — the scorer
            sees only ``x_query``.
        diffusion: the trained pose-diffusion model.
        score_fn: ``(B, T, 151) → (B,)`` scoring callable. Must be batchable.
        context: ``(1, cond_dim)`` conditioning vector for diffusion.
            Typically the dive's (action_type embedding, difficulty embedding).
        cf_cfg: search hyperparameters (uplift target, budgets, thresholds,
            and the four diffusion knobs threaded through this function).
        joint_group_to_indices: joint-group lexicon from :mod:`attribution`.
        plausibility_fn: optional PFC-style penaliser; rejects implausible CFs.
        use_score_guidance: if True, use the gradient-guided diffusion path
            (requires ``score_fn`` to be differentiable through pose tokens).
        seed: optional generator seed for reproducibility. When the iterative
            guidance ramp retries, retry ``i`` uses ``seed + i`` so each
            attempt sees fresh noise.

    Returns:
        The minimum-norm counterfactual whose actual uplift is in
        ``[uplift_target - uplift_tolerance, uplift_max]`` and that passes
        the plausibility / minimality envelope; or ``None`` if every
        weight in ``cf_cfg.iterative_guidance_weights`` came up empty.
    """
    candidates = enumerate_intervention_candidates(
        annotation, cf_cfg, joint_groups=tuple(joint_group_to_indices.keys())
    )
    if not candidates:
        log.warning("no intervention candidates for %s", annotation.dive_id)
        return None

    device = x_query.device
    T = x_query.shape[1]
    K = len(candidates)

    # Build the stack of (T, 151) masks ONCE — they don't depend on the
    # guidance weight, so we re-use them across retries.
    masks = torch.stack([
        intervention_to_mask(
            c, num_frames=T,
            joint_group_to_indices=joint_group_to_indices,
            device=device, dtype=x_query.dtype,
        )
        for c in candidates
    ], dim=0)                                      # (K, T, 151)

    # Replicate the query and context along the candidate-batch axis.
    x_query_batched = x_query.expand(K, -1, -1).contiguous()
    context_batched = context.expand(K, -1).contiguous()

    # Baseline score: re-evaluated once on the current x_query so uplift
    # is measured relative to "the state we're peeling from", not the
    # original query. This is what makes peel-off iterative.
    with torch.no_grad():
        baseline_score = float(score_fn(x_query).item())

    # ---- Iterative guidance ramp --------------------------------------
    # Loop over CFG weights. Stop at the first one that yields ANY valid
    # CF; pick the minimum-norm survivor among that weight's candidates.
    # Single-weight schedule (the old default) collapses to one attempt.
    weight_schedule = _resolve_guidance_schedule(cf_cfg)
    tau_used = float(getattr(cf_cfg, "noise_level_tau", 1.0))
    R_used = int(getattr(cf_cfg, "inpaint_resamples", 0))
    dime_used = bool(getattr(cf_cfg, "dime_gradient_scaling", False))

    last_uplift_sample: List[float] = []
    for attempt_idx, guidance_w in enumerate(weight_schedule):
        attempt_seed = None if seed is None else seed + attempt_idx

        # ---- Sample counterfactuals (batched) -------------------------
        x_cf_batched = _run_inpaint_batch(
            diffusion, x_query_batched, masks, context_batched, cf_cfg, score_fn,
            use_score_guidance=use_score_guidance,
            guidance_w=guidance_w, seed=attempt_seed,
        )

        # ---- Re-score ------------------------------------------------
        with torch.no_grad():
            cf_scores = score_fn(x_cf_batched).detach().cpu()       # (K,)
        uplifts = cf_scores - baseline_score                         # (K,)
        last_uplift_sample = [float(u) for u in uplifts[:5].tolist()]

        # ---- Minimality (in the regenerated region only) -------------
        # The inpainter guarantees the preserved region is bit-identical to
        # x_query, but we multiply by (1 - mask) anyway so numerical noise
        # in the preserved region can't inflate the norm.
        with torch.no_grad():
            diff = (x_cf_batched - x_query_batched) * (1.0 - masks)
            norms = diff.pow(2).sum(dim=(1, 2)).sqrt().detach().cpu()  # (K,)

        # ---- Plausibility (optional) ---------------------------------
        if plausibility_fn is not None:
            with torch.no_grad():
                plaus = plausibility_fn(x_cf_batched).detach().cpu()   # (K,)
        else:
            plaus = torch.zeros(K)

        # ---- Filter to the uplift / norm / plausibility envelope -----
        valid = (
              (uplifts >= cf_cfg.uplift_target - cf_cfg.uplift_tolerance)
            & (uplifts <= cf_cfg.uplift_max)
            & (norms <= cf_cfg.minimality_norm_cap)
        )
        if plausibility_fn is not None:
            valid = valid & (plaus <= cf_cfg.pfc_threshold)

        if not bool(valid.any().item()):
            if len(weight_schedule) > 1:
                log.debug(
                    "%s: no valid CFs at guidance_w=%.2f (attempt %d/%d); "
                    "uplift sample: %s",
                    annotation.dive_id, guidance_w,
                    attempt_idx + 1, len(weight_schedule),
                    [f"{u:.3f}" for u in last_uplift_sample],
                )
            continue

        # ---- Select the minimum-norm survivor at THIS weight ---------
        valid_idx = valid.nonzero(as_tuple=True)[0]
        norms_valid = norms[valid_idx]
        best = int(valid_idx[norms_valid.argmin()].item())
        actual_uplift = float(uplifts[best].item())
        return Counterfactual(
            x_cf=x_cf_batched[best:best + 1].detach().clone(),
            intervention=candidates[best],
            mask=masks[best].detach().clone(),
            claimed_uplift=cf_cfg.uplift_target,
            actual_uplift=actual_uplift,
            minimality_norm=float(norms[best].item()),
            plausibility=float(plaus[best].item()),
            faithfulness_passed=(
                abs(actual_uplift - cf_cfg.uplift_target) <= cf_cfg.uplift_tolerance
            ),
            # Diagnostics — record what we actually ran.
            noise_level_tau_used=tau_used,
            dime_gradient_scaling_used=dime_used and use_score_guidance,
            inpaint_resamples_used=R_used,
            guidance_weight_used=float(guidance_w),
        )

    # All weights exhausted with no valid CF.
    log.debug(
        "%s: no valid CFs after %d guidance attempt(s) (uplift sample: %s)",
        annotation.dive_id, len(weight_schedule),
        [f"{u:.3f}" for u in last_uplift_sample],
    )
    return None


# =============================================================================
# Outer loop: iterative decomposition
# =============================================================================


def decompose(
    x_query: torch.Tensor,                    # (1, T, 151)
    annotation: DiveAnnotation,
    diffusion: MotionDiffusion,
    score_fn: ScoreFn,
    context: torch.Tensor,                    # (1, cond_dim)
    cf_cfg: CFCfg,
    joint_group_to_indices: Dict[str, Sequence[int]],
    fault_mapper: Callable[[Intervention], str],
    *,
    target_uplift: Optional[float] = None,
    plausibility_fn: Optional[PlausibilityFn] = None,
    use_score_guidance: bool = False,
    seed: int = 0,
) -> DecompositionResult:
    """Iteratively peel half-point counterfactuals off the query's score gap.

    Args:
        x_query: the original query dive's pose tokens.
        annotation: the query's :class:`DiveAnnotation`.
        diffusion: the trained pose-diffusion model.
        score_fn: ``(B, T, 151) → (B,)`` scoring callable.
        context: diffusion conditioning vector.
        cf_cfg: search hyperparameters.
        joint_group_to_indices: joint-group lexicon (from attribution).
        fault_mapper: ``Intervention → fault_name`` from :mod:`attribution`.
        target_uplift: total score gap to close. If ``None``, defaulted to
            ``10 - median(judges_scores)`` — the convention used by the
            FineDiving-HF annotation protocol. The caller is responsible
            for ensuring ``target_uplift`` and ``score_fn``'s output share
            the same units (judge-points or official score).
        plausibility_fn: optional PFC-style plausibility callable.
        use_score_guidance: switch to gradient-guided diffusion sampling.
        seed: master seed. Each peel uses
            ``seed + peel_index * len(iterative_guidance_weights)`` as
            its base seed so the per-peel iterative-guidance retries
            (which increment within the peel) don't collide with later
            peels' seeds.

    Returns:
        :class:`DecompositionResult` carrying the ordered list of records
        plus diagnostic counters.
    """
    if target_uplift is None:
        median_judge = float(np.median(annotation.judges_scores))
        target_uplift = 10.0 - median_judge
    if target_uplift <= 0:
        # Perfect (or close to perfect) dive: nothing to decompose.
        with torch.no_grad():
            base = float(score_fn(x_query).item())
        return DecompositionResult(
            dive_id=annotation.dive_id,
            target_uplift=0.0, achieved_uplift=0.0,
            initial_score=base, final_score=base,
            records=[], residual_gap=0.0, completed=True,
            aborted_reason=None,
        )

    with torch.no_grad():
        initial_score = float(score_fn(x_query).item())

    # Per-peel seed stride: enough slots for every iterative-guidance
    # retry without spilling into the next peel's seed range. Use 1000
    # as a coarse upper bound — far above any realistic schedule length.
    seed_stride = max(1000, len(_resolve_guidance_schedule(cf_cfg)) + 1)

    records: List[DecompositionRecord] = []
    current_x = x_query
    achieved = 0.0
    aborted_reason: Optional[str] = None

    for peel_idx in range(cf_cfg.max_peeled_half_points):
        remaining = target_uplift - achieved
        if remaining < cf_cfg.residual_threshold:
            break

        log.info(
            "[%s] peel %d (achieved %.2f / %.2f, remaining %.2f)",
            annotation.dive_id, peel_idx, achieved, target_uplift, remaining,
        )

        cf = search_one_half_point(
            x_query=current_x, annotation=annotation,
            diffusion=diffusion, score_fn=score_fn, context=context,
            cf_cfg=cf_cfg, joint_group_to_indices=joint_group_to_indices,
            plausibility_fn=plausibility_fn,
            use_score_guidance=use_score_guidance,
            seed=seed + peel_idx * seed_stride,
        )
        if cf is None:
            aborted_reason = f"no_valid_cf_at_peel_{peel_idx}"
            log.warning(
                "[%s] decomposition stopped early at peel %d (no valid CF)",
                annotation.dive_id, peel_idx,
            )
            break

        fault = fault_mapper(cf.intervention)
        records.append(DecompositionRecord(
            peel_index=peel_idx,
            intervention=cf.intervention,
            counterfactual=cf,
            fault=fault,
            delta_score=cf.actual_uplift,
        ))

        achieved += cf.actual_uplift
        current_x = cf.x_cf

    with torch.no_grad():
        final_score = float(score_fn(current_x).item())

    residual = target_uplift - achieved
    completed = residual < cf_cfg.residual_threshold
    if not completed and aborted_reason is None:
        aborted_reason = "peel_cap_reached"

    return DecompositionResult(
        dive_id=annotation.dive_id,
        target_uplift=target_uplift,
        achieved_uplift=achieved,
        initial_score=initial_score,
        final_score=final_score,
        records=records,
        residual_gap=max(0.0, residual),
        completed=completed,
        aborted_reason=aborted_reason,
    )


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    # Types
    "ScoreFn", "PlausibilityFn",
    # Dataclasses
    "Intervention", "Counterfactual",
    "DecompositionRecord", "DecompositionResult",
    # Algorithm
    "enumerate_intervention_candidates", "intervention_to_mask",
    "search_one_half_point", "decompose",
]

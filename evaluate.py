"""evaluate.py — CLI evaluation pipeline.

Subcommands
-----------
- ``aqa``: load an AQA checkpoint, run on the test set with M-exemplar
  voting, compute Spearman + R-ℓ2 with bootstrap CIs, stratify by
  action type / score quintile / height, and write a CSV. The numbers
  produced here are what Table 1 of the paper reports.

- ``cf``: load AQA + diffusion + ContextEncoder + (optional) pose-AQA
  proxy, run :func:`counterfactual.decompose` on every FineDiving-HF
  dive, and dump the results as a single JSON cache. Each cache entry
  carries per-record diagnostics (the four diffusion-sampler knobs that
  produced each CF — Issues 10/11/17) and a per-dive CF-XAI block
  (validity / sparsity / proximity / realism / completion — Issue 12)
  computed while the x_cf tensors are still in memory.

- ``attribution``: load the CF cache, compare against FineDiving-HF
  expert labels, and write a CSV with overall / top-K / per-fault /
  stratified attribution metrics, plus aggregated CF-XAI metrics.
  Table 3 of the paper. Reports both ``top_k_per_dive`` (recommended,
  the headline) and ``top_k_union`` (for cross-comparison).

- ``expert_study``: aggregate per-rater preference CSVs, compute
  pairwise Cohen's κ inter-rater agreement and the majority-preferred
  rate, and write a summary CSV. Table 4.

Design notes
------------
- The :class:`PoseAQAProxy` here is the explicit bridge between
  pose-space (where diffusion lives) and AQA-score-space (which is
  pixel-based in TSA-Net). Trained separately to predict frame-AQA
  scores from pose tokens; without a trained checkpoint, falls back to
  a randomly-initialised proxy and logs a clear warning.

- Evaluation is single-process; DDP isn't worth the complexity here.

- The CF cache uses JSON (not pickled tensors) so it's human-inspectable
  and version-control-friendly. Tensor counterfactuals are too large
  to cache; we therefore compute the realism / sparsity / proximity
  numbers during ``cf`` (when the tensors are in hand) and only store
  the scalar metrics.

Changes from previous version (camera-ready)
---------------------------------------------
- Issue 12 (CF-XAI vocabulary): per-dive ``cfxai_metrics`` block
  embedded in the cache; aggregated via :func:`cf_metrics_cfxai` in
  ``attribution`` to produce the canonical
  validity/sparsity/proximity/realism/completion columns.
- Issue 8 (dual PFC): realism is computed as EDGE-style PFC by
  default (``cfg.cf.pfc_metric``). When ``pfc_metric == "both"`` both
  variants are stored and the headline column tracks
  ``cfg.eval.pfc_primary_for_acceptance``.
- Issue 13 (fault gating): ``cmd_cf`` uses
  :func:`attribution.make_fault_mapper` so the dive's
  ``sub_action_types`` and the ``enforce_fault_dive_gating`` flag are
  baked into the closure. ``cmd_attribution`` calls
  :func:`attribution.filter_attribution_tuples_by_applicability` on
  predicted tuples when ``cfg.eval.attribution_apply_dive_gating`` is
  True (per-fault P/R/F1 then has a fair denominator).
- ``top_k_per_dive`` is the headline metric; ``top_k_union`` is also
  emitted for backwards-compatibility comparison.
- :func:`_decomposition_to_dict` now surfaces the four diagnostic
  fields from :class:`Counterfactual` (noise_level_tau_used,
  dime_gradient_scaling_used, inpaint_resamples_used,
  guidance_weight_used), making the cache reproducibility-complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .configs import Cfg
from .data import (
    DataModule, DiveAnnotation, FaultAnnotations,
    ExemplarSelector, PoseDataset, load_finediving_annotations,
)
from .aqa_model import TSAModel
from .diffusion import (
    MotionDiffusion, tokenize_pose, axis_angle_to_rotmat,
    derive_foot_contacts, detokenize_pose, sixd_to_rotmat,
    SMPL_FOOT_JOINTS, TOKEN_DIM,
)
from .counterfactual import (
    Intervention, Counterfactual, DecompositionResult, decompose,
)
from .attribution import (
    JOINT_GROUP_TO_INDICES, FAULT_TAXONOMY, FAULT_NULL,
    FAULT_APPLICABILITY,
    make_fault_mapper,
    filter_attribution_tuples_by_applicability,
)
from .metrics import (
    evaluate_aqa as metrics_eval_aqa,
    evaluate_attribution_set,
    cf_metrics_cfxai,
    physical_foot_contact_edge, airborne_lateral_accel,
    spearman_correlation, r_l2, MetricWithCI,
    bootstrap_metric,
)
from .train import ContextEncoder, load_cfg, setup_logging

log = logging.getLogger(__name__)


# =============================================================================
# PoseAQAProxy — the pose → AQA-score bridge
# =============================================================================


class PoseAQAProxy(nn.Module):
    """Small transformer-over-pose-tokens trained to predict AQA scores.

    Bridges the gap between diffusion (pose space) and the AQA scorer
    (pixel space). Trained as a separate step to mimic frame-AQA's
    predictions on the same dives — i.e. distilled from the frozen
    TSA-Net oracle.

    Architecture: a thin transformer encoder over 151-dim pose tokens,
    mean-pooled across time, projected to a scalar score. Small enough
    that the CF search's per-step gradient pass through it is cheap.
    """

    def __init__(
        self, token_dim: int = TOKEN_DIM, width: int = 256,
        heads: int = 4, depth: int = 4, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(token_dim, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, 151) → (B,)`` predicted score."""
        h = self.input_proj(x)                # (B, T, width)
        h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


def build_score_fn(
    args: argparse.Namespace, device: torch.device,
):
    """Build the ``score_fn`` callable used by :func:`decompose`.

    If ``--pose-proxy-ckpt`` is provided, load it. Otherwise spin up a
    randomly-initialised proxy and emit a loud warning — useful for
    end-to-end pipeline verification but obviously not for paper numbers.
    """
    proxy = PoseAQAProxy()
    if getattr(args, "pose_proxy_ckpt", None):
        state = torch.load(args.pose_proxy_ckpt, map_location="cpu")
        sd = state.get("state_dict", state)
        proxy.load_state_dict(sd, strict=True)
        log.info("loaded pose-AQA proxy from %s", args.pose_proxy_ckpt)
    else:
        log.warning(
            "NO POSE-AQA PROXY CHECKPOINT GIVEN — using random init. "
            "The CF search will run end-to-end but the numbers are MEANINGLESS."
        )
    proxy.to(device).eval()
    for p in proxy.parameters():
        p.requires_grad_(False)

    def score_fn(x: torch.Tensor) -> torch.Tensor:
        return proxy(x)

    return score_fn


# =============================================================================
# Checkpoint loaders
# =============================================================================


def load_aqa_model(ckpt_path: Path, cfg: Cfg, device: torch.device) -> TSAModel:
    """Load a TSAModel from a train.py-format checkpoint."""
    state = torch.load(str(ckpt_path), map_location="cpu")
    use_stsa = bool(state.get("use_stsa", False))
    model = TSAModel(cfg.aqa, cfg.data, use_stsa=use_stsa)
    model.load_state_dict(state["state_dict"], strict=True)
    if state.get("config_fingerprint") != cfg.fingerprint():
        log.warning(
            "AQA checkpoint config fingerprint mismatch (ckpt=%s vs runtime=%s); "
            "loading anyway but the architecture should match — verify before "
            "trusting numbers.",
            state.get("config_fingerprint"), cfg.fingerprint(),
        )
    model.to(device).eval()
    return model


def load_diffusion_model(
    ckpt_path: Path, cfg: Cfg, device: torch.device,
) -> Tuple[MotionDiffusion, ContextEncoder]:
    """Load EMA diffusion weights + the matching ContextEncoder."""
    state = torch.load(str(ckpt_path), map_location="cpu")
    diffusion = MotionDiffusion(cfg.diffusion, cfg.data)
    # Use the EMA weights for inference (this is what training optimised for).
    if "ema" in state:
        diffusion.load_state_dict(state["ema"], strict=True)
    else:
        log.warning("no EMA weights in checkpoint; falling back to live model")
        diffusion.load_state_dict(state["state_dict"], strict=True)
    diffusion.to(device).eval()
    for p in diffusion.parameters():
        p.requires_grad_(False)

    action_types = state.get("action_types")
    if not action_types:
        raise RuntimeError(
            "diffusion checkpoint is missing 'action_types' — was it produced "
            "by an outdated train.py? Re-train or hand-patch the checkpoint."
        )
    encoder = ContextEncoder(
        action_types=action_types,
        cond_action_dim=cfg.diffusion.cond_action_embedding_dim,
        cond_summary_dim=cfg.diffusion.cond_summary_dim,
    )
    encoder.load_state_dict(state["encoder"], strict=True)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return diffusion, encoder


# =============================================================================
# Subcommand: aqa
# =============================================================================


@torch.no_grad()
def _run_aqa_inference(
    model: TSAModel, dm: DataModule, cfg: Cfg, device: torch.device,
    M: int = 10, limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """One predicted score per test dive (M-exemplar voting), plus metadata.

    Inference path: calls :meth:`TSAModel.frozen` to force
    ``use_symmetric=False`` and disable the curriculum gate, so the
    deployed model behaves exactly as it does at test time regardless
    of training-time settings.
    """
    test_ds = dm.test_dataset()
    train_pool = dm._require("_train_anns")
    selector = ExemplarSelector(
        train_pool=train_pool,
        strategy=cfg.data.exemplar_strategy,
        difficulty_window=cfg.data.exemplar_difficulty_window,
        seed=cfg.train.seed,
    )
    predictions: List[float] = []
    targets: List[float] = []
    metadata: List[Dict[str, Any]] = []
    n_dives = len(test_ds) if limit is None else min(limit, len(test_ds))
    for i in range(n_dives):
        query = test_ds.annotations[i]
        exemplars = selector.sample_voting_set(query, M=M)
        per_exemplar_scores: List[float] = []
        for ex in exemplars:
            pair = test_ds.make_pair(query, ex)
            q = pair.query_frames.unsqueeze(0).to(device)
            e = pair.exemplar_frames.unsqueeze(0).to(device)
            ys = pair.exemplar_score.unsqueeze(0).to(device)
            out = model(q, e, ys)
            per_exemplar_scores.append(float(out.predicted_score.item()))
        predictions.append(float(np.mean(per_exemplar_scores)))
        targets.append(query.final_score)
        metadata.append({
            "dive_id": query.dive_id,
            "action_type": query.action_type,
            "difficulty": query.difficulty,
            "height": query.height,
        })
        if (i + 1) % 20 == 0:
            log.info("  AQA inference: %d / %d", i + 1, n_dives)
    return np.asarray(predictions), np.asarray(targets), metadata


def _stratify_aqa(
    preds: np.ndarray, targets: np.ndarray, metadata: List[Dict[str, Any]],
    cfg: Cfg, min_group: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Per-stratum AQA metrics for the table."""
    stratified: Dict[str, Dict[str, float]] = {}
    score_range = (cfg.data.score_min, cfg.data.score_max)

    def _summary(p: np.ndarray, t: np.ndarray) -> Dict[str, float]:
        return {
            "spearman": float(spearman_correlation(p, t)),
            "r_l2": float(r_l2(p, t, score_range)),
            "n": int(len(p)),
        }

    # By action_type
    action_types = np.array([m["action_type"] for m in metadata])
    for a in sorted(set(action_types)):
        mask = action_types == a
        if int(mask.sum()) >= min_group:
            stratified[f"action::{a}"] = _summary(preds[mask], targets[mask])

    # By score quintile (same bounds as FineDiving-HF stratification)
    for i, (lo, hi) in enumerate(FaultAnnotations.QUINTILE_BOUNDS):
        if i == len(FaultAnnotations.QUINTILE_BOUNDS) - 1:
            mask = targets >= lo
        else:
            mask = (targets >= lo) & (targets < hi)
        if int(mask.sum()) >= min_group:
            stratified[f"quintile::{i}"] = _summary(preds[mask], targets[mask])

    # By height
    heights = np.array([m["height"] for m in metadata if m["height"] is not None])
    all_heights = [m["height"] for m in metadata]
    for h in sorted(set(heights)):
        mask = np.array([x == h for x in all_heights])
        if int(mask.sum()) >= min_group:
            stratified[f"height::{h}"] = _summary(preds[mask], targets[mask])

    return stratified


def _write_aqa_csv(
    overall: Dict[str, MetricWithCI], stratified: Dict[str, Dict[str, float]],
    output: Path,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stratum", "metric", "point", "ci_low", "ci_high", "n"])
        for name, m in overall.items():
            w.writerow(["overall", name, m.point, m.ci_low, m.ci_high, ""])
        for stratum, vals in sorted(stratified.items()):
            n = vals.get("n", "")
            for metric_name, value in vals.items():
                if metric_name == "n":
                    continue
                w.writerow([stratum, metric_name, value, "", "", n])
    log.info("wrote AQA metrics CSV to %s", output)


def cmd_aqa(args: argparse.Namespace, cfg: Cfg, device: torch.device) -> None:
    log.info("loading AQA checkpoint: %s", args.ckpt)
    model = load_aqa_model(Path(args.ckpt), cfg, device)
    dm = DataModule(cfg)
    dm.setup()
    log.info("running test-set inference (M=%d)", cfg.aqa.voting_M_inference)
    with model.frozen():
        preds, targets, metadata = _run_aqa_inference(
            model, dm, cfg, device,
            M=cfg.aqa.voting_M_inference, limit=args.limit,
        )
    action_types = [m["action_type"] for m in metadata]
    overall = metrics_eval_aqa(
        preds, targets,
        score_range=(cfg.data.score_min, cfg.data.score_max),
        action_types=action_types,
        n_bootstrap=cfg.eval.spearman_bootstrap,
        seed=cfg.train.seed,
    )
    log.info("Spearman %s, R-ℓ2 %s", overall["spearman"], overall["r_l2"])
    stratified = _stratify_aqa(preds, targets, metadata, cfg)
    _write_aqa_csv(overall, stratified, Path(args.output))


# =============================================================================
# Subcommand: cf
# =============================================================================


def _tokenize_single_pose(
    theta: torch.Tensor, tau: torch.Tensor, kinematics,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Single-batch version of train.py's tokenizer (no DDP, no autocast)."""
    rotmat = axis_angle_to_rotmat(theta)
    positions = kinematics(rotmat, tau)
    contacts = derive_foot_contacts(
        positions[..., SMPL_FOOT_JOINTS, :], velocity_threshold=velocity_threshold,
    )
    return tokenize_pose(theta, tau, contacts=contacts)


def _x_cf_to_joints_3d(
    x_cf: torch.Tensor, kinematics,
) -> np.ndarray:
    """Convert a (1, T, 151) CF pose-token tensor into (T, J, 3) joints_3d.

    Used to compute realism (PFC) on the generated CF without storing
    the full 151-dim tensor in the cache.
    """
    decoded = detokenize_pose(x_cf)                # contacts/sixd/root
    rotmat = sixd_to_rotmat(decoded["sixd"])       # (1, T, J, 3, 3)
    positions = kinematics(rotmat, decoded["root"])  # (1, T, J, 3)
    return positions.squeeze(0).detach().cpu().numpy()


def _x_cf_to_com_and_contacts(
    x_cf: torch.Tensor, kinematics,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert (1, T, 151) into (com_positions: (T, 3), contacts: (T, 4)).

    Used for the lateral-accel PFC variant.
    """
    decoded = detokenize_pose(x_cf)
    rotmat = sixd_to_rotmat(decoded["sixd"])
    positions = kinematics(rotmat, decoded["root"])     # (1, T, J, 3)
    com = positions.mean(dim=-2).squeeze(0).detach().cpu().numpy()  # (T, 3)
    contacts = decoded["contacts"].squeeze(0).detach().cpu().numpy()  # (T, 4)
    return com, contacts


def _compute_record_pfc(
    x_cf: torch.Tensor, kinematics, cf_cfg,
) -> Dict[str, float]:
    """Compute per-record PFC (one variant or both) on a CF pose tensor.

    Reads ``cf_cfg.pfc_metric`` ∈ {"edge", "lateral_accel", "both"} —
    when "both", both numbers are stored so the CSV can switch between
    them downstream without re-running ``cf``.
    """
    metric = getattr(cf_cfg, "pfc_metric", "edge")
    out: Dict[str, float] = {}
    if metric in ("edge", "both"):
        joints_3d = _x_cf_to_joints_3d(x_cf, kinematics)
        # Per-clip EDGE PFC (population scaling applied in aggregation).
        out["pfc_edge"] = float(physical_foot_contact_edge(joints_3d))
    if metric in ("lateral_accel", "both"):
        com, contacts = _x_cf_to_com_and_contacts(x_cf, kinematics)
        out["pfc_lateral_accel"] = float(
            airborne_lateral_accel(com, contacts)
        )
    return out


def _decomposition_to_dict(
    result: DecompositionResult,
    *,
    kinematics=None,
    cf_cfg=None,
) -> Dict[str, Any]:
    """JSON-serialisable representation of a :class:`DecompositionResult`.

    The four diagnostic fields from :class:`Counterfactual` are surfaced
    so the cache is reproducibility-complete (which τ / R / guidance
    weight / DiME flag produced each record). When ``kinematics`` and
    ``cf_cfg`` are provided, per-record PFC scores are also computed
    and embedded.
    """
    records: List[Dict[str, Any]] = []
    for r in result.records:
        cf = r.counterfactual
        rec: Dict[str, Any] = {
            "peel_index": r.peel_index,
            "step": r.intervention.step,
            "modality": r.intervention.modality,
            "frame_range": list(r.intervention.frame_range),
            "joint_groups": list(r.intervention.joint_groups),
            "transition_shift_frames": r.intervention.transition_shift_frames,
            "fault": r.fault,
            "delta_score": r.delta_score,
            "actual_uplift": cf.actual_uplift,
            "claimed_uplift": cf.claimed_uplift,
            "minimality_norm": cf.minimality_norm,
            "plausibility": cf.plausibility,
            "faithfulness_passed": cf.faithfulness_passed,
            # ---- diagnostics (Issues 10/11/17): which knobs produced this CF
            "noise_level_tau_used":       cf.noise_level_tau_used,
            "dime_gradient_scaling_used": cf.dime_gradient_scaling_used,
            "inpaint_resamples_used":     cf.inpaint_resamples_used,
            "guidance_weight_used":       cf.guidance_weight_used,
        }
        # ---- per-record PFC (Issue 8): EDGE + optionally lateral_accel
        if kinematics is not None and cf_cfg is not None:
            try:
                rec.update(_compute_record_pfc(cf.x_cf, kinematics, cf_cfg))
            except Exception as e:                                # pragma: no cover
                log.debug(
                    "PFC computation failed for peel %d: %s",
                    r.peel_index, e,
                )
        records.append(rec)

    return {
        "dive_id": result.dive_id,
        "target_uplift": result.target_uplift,
        "achieved_uplift": result.achieved_uplift,
        "initial_score": result.initial_score,
        "final_score": result.final_score,
        "residual_gap": result.residual_gap,
        "completed": result.completed,
        "aborted_reason": result.aborted_reason,
        "records": records,
    }


def cmd_cf(args: argparse.Namespace, cfg: Cfg, device: torch.device) -> None:
    log.info("loading AQA from %s", args.aqa_ckpt)
    aqa = load_aqa_model(Path(args.aqa_ckpt), cfg, device)
    log.info("loading diffusion from %s", args.diff_ckpt)
    diffusion, encoder = load_diffusion_model(Path(args.diff_ckpt), cfg, device)
    score_fn = build_score_fn(args, device)

    if args.use_score_guidance:
        log.info("CF mode: score-guided diffusion (autograd through score_fn)")
    else:
        log.info("CF mode: gradient-free inpainting")
    log.info(
        "CF knobs: noise_level_tau=%s, dime_gradient_scaling=%s, "
        "inpaint_resamples=%s, iterative_guidance_weights=%s, "
        "enforce_fault_dive_gating=%s, pfc_metric=%s",
        getattr(cfg.cf, "noise_level_tau", 1.0),
        getattr(cfg.cf, "dime_gradient_scaling", False),
        getattr(cfg.cf, "inpaint_resamples", 0),
        getattr(cfg.cf, "iterative_guidance_weights", None),
        getattr(cfg.cf, "enforce_fault_dive_gating", True),
        getattr(cfg.cf, "pfc_metric", "edge"),
    )

    # Annotations — search across train + val + test since FineDiving-HF
    # annotates dives from all splits (it's stratified by score, not split).
    dm = DataModule(cfg)
    dm.setup()
    all_anns: List[DiveAnnotation] = (
        dm._require("_train_anns") + dm._require("_val_anns") + dm._require("_test_anns")
    )
    anns_by_id = {a.dive_id: a for a in all_anns}

    # FineDiving-HF labels — these are the dives we decompose.
    fa = FaultAnnotations.from_cfg(cfg.data)
    dive_ids = list(fa.dive_ids)
    if args.limit is not None:
        dive_ids = dive_ids[: args.limit]

    # Poses — silently skip missing.
    pose_ds = PoseDataset(dive_ids, cfg.data, require_all=False)
    pose_index = {d: i for i, d in enumerate(pose_ds.dive_ids)}

    log.info("decomposing %d dives", len(dive_ids))
    results: Dict[str, Any] = {}
    kinematics = diffusion.kinematics
    with aqa.frozen():
        for n, dive_id in enumerate(dive_ids, start=1):
            if dive_id not in anns_by_id:
                results[dive_id] = {"error": "no annotation found"}
                continue
            if dive_id not in pose_index:
                results[dive_id] = {"error": "no pose data"}
                continue
            ann = anns_by_id[dive_id]
            pose = pose_ds[pose_index[dive_id]]
            theta = pose["theta"].unsqueeze(0).to(device)
            tau = pose["tau"].unsqueeze(0).to(device)
            x_query = _tokenize_single_pose(theta, tau, diffusion.kinematics)
            difficulty = torch.tensor([ann.difficulty], device=device)
            context = encoder([ann.action_type], difficulty)

            # Build the per-dive fault mapper via the factory (Issue 13).
            # Bakes in sub_action_types and the dive-type gating flag.
            fault_mapper = make_fault_mapper(
                ann.sub_action_types,
                gate_by_dive_type=getattr(cfg.cf, "enforce_fault_dive_gating", True),
            )

            try:
                result = decompose(
                    x_query=x_query, annotation=ann, diffusion=diffusion,
                    score_fn=score_fn, context=context, cf_cfg=cfg.cf,
                    joint_group_to_indices=JOINT_GROUP_TO_INDICES,
                    fault_mapper=fault_mapper,
                    use_score_guidance=args.use_score_guidance,
                    seed=cfg.train.seed + n,
                )
                results[dive_id] = _decomposition_to_dict(
                    result, kinematics=kinematics, cf_cfg=cfg.cf,
                )
            except Exception as e:                            # pragma: no cover
                log.exception("decomposition crashed for %s", dive_id)
                results[dive_id] = {"error": str(e)}
            if n % 5 == 0:
                log.info("  decomposed %d / %d", n, len(dive_ids))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote CF cache to %s (%d entries)", output, len(results))


# =============================================================================
# Subcommand: attribution
# =============================================================================


def _load_cf_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    with Path(path).open("r") as f:
        return json.load(f)


def _build_attribution_inputs(
    cache: Dict[str, Dict[str, Any]],
    faults: FaultAnnotations,
    annotations_by_id: Dict[str, DiveAnnotation],
    *,
    apply_dive_gating: bool = False,
) -> Tuple[
    List[Tuple[List[Tuple], bool]],
    List[List[Tuple]],
    List[str],
    List[Optional[int]],
    int,
]:
    """Pair predicted decompositions with expert labels.

    When ``apply_dive_gating`` is True, predicted tuples whose fault
    doesn't apply to the dive (per :data:`FAULT_APPLICABILITY`) are
    dropped before the comparison — this gives per-fault P/R/F1 a
    fair denominator on dives where the fault is structurally
    impossible (e.g. ``late_twist`` on a no-twist dive).

    Returns:
        ``(decompositions, expert_per_dive, dive_ids, strata, n_gated)``.
        ``n_gated`` counts predicted tuples that were filtered out by
        dive-type gating (zero when ``apply_dive_gating=False``).
    """
    decompositions: List[Tuple[List[Tuple], bool]] = []
    expert_per_dive: List[List[Tuple]] = []
    dive_ids: List[str] = []
    strata: List[Optional[int]] = []
    n_gated = 0

    for dive_id, entry in cache.items():
        if "error" in entry or "records" not in entry:
            continue
        if dive_id not in faults:
            log.debug("%s: no FineDiving-HF label, skipping", dive_id)
            continue
        # Predicted tuples in peel order.
        predicted = [
            (r["step"], list(r["joint_groups"]), r["fault"], float(r["delta_score"]))
            for r in entry["records"]
        ]
        # Optional eval-time gating by dive type (Issue 13).
        if apply_dive_gating and dive_id in annotations_by_id:
            sub_actions = annotations_by_id[dive_id].sub_action_types
            pre_count = len(predicted)
            predicted = list(filter_attribution_tuples_by_applicability(
                predicted, sub_actions,
            ))
            n_gated += pre_count - len(predicted)

        # Expert tuples (order doesn't matter, but we preserve the codebook order).
        expert_records = faults.for_dive(dive_id).records
        expert = [
            (er.step, list(er.joints), er.fault, float(er.delta))
            for er in expert_records
        ]
        decompositions.append((predicted, bool(entry.get("completed", False))))
        expert_per_dive.append(expert)
        dive_ids.append(dive_id)
        strata.append(faults.stratum_for(faults.for_dive(dive_id).median_judge_score * 10.0))

    return decompositions, expert_per_dive, dive_ids, strata, n_gated


def _aggregate_cfxai_metrics(
    cache: Dict[str, Dict[str, Any]],
    pfc_primary: str = "edge",
) -> Dict[str, float]:
    """Aggregate per-dive CF-XAI numbers from the cache.

    Reads the per-record ``actual_uplift / claimed_uplift /
    minimality_norm / pfc_edge or pfc_lateral_accel``, plus the
    dive-level ``completed`` flag, and pipes them through
    :func:`metrics.cf_metrics_cfxai` for the canonical
    validity/sparsity/proximity/realism/completion columns.

    Args:
        pfc_primary: which PFC variant to use as ``realism``.
            ``"edge"`` is the headline (recommended). ``"lateral_accel"``
            is the legacy variant — the CSV reports both when the
            cache carries both.
    """
    actual: List[float] = []
    claimed: List[float] = []
    norms: List[float] = []
    n_joints_changed: List[int] = []
    pfc_scores: List[float] = []
    completed_flags: List[bool] = []

    pfc_key = "pfc_edge" if pfc_primary == "edge" else "pfc_lateral_accel"

    for dive_id, entry in cache.items():
        if "error" in entry or "records" not in entry:
            continue
        # For CF-XAI, the per-dive "validity / sparsity / proximity"
        # are aggregated over the dive's peels: each peel is one
        # generated CF, and the dive-level CF-XAI summary takes the
        # mean/rate over all peels in the cache.
        for r in entry["records"]:
            actual.append(float(r.get("actual_uplift", 0.0)))
            claimed.append(float(r.get("claimed_uplift", 0.0)))
            norms.append(float(r.get("minimality_norm", 0.0)))
            n_joints_changed.append(len(r.get("joint_groups", [])))
            # Realism: use the cached PFC under the chosen variant.
            if pfc_key in r:
                pfc_scores.append(float(r[pfc_key]))
            else:
                # Fall back to the legacy plausibility field (might be 0
                # when no plausibility_fn was passed).
                pfc_scores.append(float(r.get("plausibility", 0.0)))
        completed_flags.append(bool(entry.get("completed", False)))

    return cf_metrics_cfxai(
        actual_uplifts=actual, claimed_uplifts=claimed,
        pose_l2_norms=norms,
        n_joint_groups_changed=n_joints_changed,
        pfc_scores=pfc_scores,
        completed_flags=completed_flags,
    )


def _write_attribution_csv(
    summary: Dict[str, Any],
    stratified: Dict[int, Dict[str, Any]],
    cfxai_primary: Dict[str, float],
    cfxai_legacy: Optional[Dict[str, float]],
    output: Path,
    *,
    pfc_primary: str,
    n_gated: int,
) -> None:
    """Write the camera-ready attribution CSV.

    Columns (one row per metric):
        scope, metric, value, n
    Scopes used:
        - ``overall``:   overall P/R/F1, fully_explained_rate
        - ``top_<k>_per_dive``:  per-dive top-K (headline; Issue 13)
        - ``top_<k>_union``:     union top-K (legacy; for cross-comparison)
        - ``fault::<name>``:     per-fault P/R/F1
        - ``quintile::<i>``:     stratified by FineDiving-HF score quintile
        - ``cfxai_<pfc>``:       canonical CF-XAI metrics (Issue 12)
                                 with realism=<pfc_primary>
        - ``cfxai_<other>``:     other CF-XAI variant (if both stored)
        - ``meta``:              top-line counters (gating, etc.)
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scope", "metric", "value", "n"])

        # ---- Overall (set-level) -----------------------------------
        o = summary["overall"]
        w.writerow(["overall", "precision", o["precision"], o["n_pred"]])
        w.writerow(["overall", "recall", o["recall"], o["n_expert"]])
        w.writerow(["overall", "f1", o["f1"], ""])
        w.writerow([
            "overall", "fully_explained_rate",
            summary["fully_explained_rate"], summary["n_dives"],
        ])

        # ---- Top-K (per-dive: headline; union: legacy) -------------
        for k_name, k_metrics in summary["top_k_per_dive"].items():
            w.writerow([
                f"{k_name}_per_dive", "precision",
                k_metrics["precision"], k_metrics["n_pred"],
            ])
            w.writerow([
                f"{k_name}_per_dive", "recall",
                k_metrics["recall"], k_metrics["n_expert"],
            ])
            w.writerow([f"{k_name}_per_dive", "f1", k_metrics["f1"], ""])
        for k_name, k_metrics in summary["top_k_union"].items():
            w.writerow([
                f"{k_name}_union", "precision",
                k_metrics["precision"], k_metrics["n_pred"],
            ])
            w.writerow([
                f"{k_name}_union", "recall",
                k_metrics["recall"], k_metrics["n_expert"],
            ])
            w.writerow([f"{k_name}_union", "f1", k_metrics["f1"], ""])

        # ---- Per-fault --------------------------------------------
        for fault, m in sorted(summary["per_fault"].items()):
            w.writerow([f"fault::{fault}", "precision", m["precision"], m["n_pred"]])
            w.writerow([f"fault::{fault}", "recall", m["recall"], m["n_expert"]])
            w.writerow([f"fault::{fault}", "f1", m["f1"], ""])

        # ---- Stratified by score quintile -------------------------
        for stratum, s in sorted(stratified.items()):
            o = s["overall"]
            w.writerow([f"quintile::{stratum}", "precision", o["precision"], o["n_pred"]])
            w.writerow([f"quintile::{stratum}", "recall", o["recall"], o["n_expert"]])
            w.writerow([f"quintile::{stratum}", "f1", o["f1"], ""])
            w.writerow([f"quintile::{stratum}", "fully_explained_rate",
                        s["fully_explained_rate"], s["n_dives"]])

        # ---- CF-XAI canonical columns (Issue 12) -------------------
        scope_primary = f"cfxai_{pfc_primary}"
        for k, v in cfxai_primary.items():
            if k == "n":
                continue
            w.writerow([scope_primary, k, v, cfxai_primary.get("n", "")])
        if cfxai_legacy is not None:
            other = "lateral_accel" if pfc_primary == "edge" else "edge"
            scope_other = f"cfxai_{other}"
            for k, v in cfxai_legacy.items():
                if k == "n":
                    continue
                w.writerow([scope_other, k, v, cfxai_legacy.get("n", "")])

        # ---- Meta counters ----------------------------------------
        w.writerow(["meta", "n_gated_by_dive_type", n_gated, ""])
        w.writerow(["meta", "pfc_primary", pfc_primary, ""])

    log.info("wrote attribution metrics CSV to %s", output)


def cmd_attribution(args: argparse.Namespace, cfg: Cfg) -> None:
    cache = _load_cf_cache(Path(args.decompositions))
    faults = FaultAnnotations.from_cfg(cfg.data)

    # Need annotations for the eval-time dive-type gating (Issue 13).
    apply_gating = bool(getattr(cfg.eval, "attribution_apply_dive_gating", True))
    annotations_by_id: Dict[str, DiveAnnotation] = {}
    if apply_gating:
        dm = DataModule(cfg)
        dm.setup()
        all_anns = (
            dm._require("_train_anns") + dm._require("_val_anns") + dm._require("_test_anns")
        )
        annotations_by_id = {a.dive_id: a for a in all_anns}
        log.info(
            "applying dive-type fault gating "
            "(cfg.eval.attribution_apply_dive_gating=True)"
        )

    decompositions, expert_per_dive, dive_ids, strata, n_gated = (
        _build_attribution_inputs(
            cache, faults, annotations_by_id,
            apply_dive_gating=apply_gating,
        )
    )
    if apply_gating:
        log.info("gating dropped %d predicted tuples", n_gated)
    log.info("evaluating attribution on %d dives", len(decompositions))
    if not decompositions:
        raise RuntimeError("no dives with both predictions and expert labels")

    summary = evaluate_attribution_set(
        decompositions, expert_per_dive,
        fault_categories=FAULT_TAXONOMY,
        top_k_values=cfg.eval.attribution_topk,
        match=args.match,
    )

    # Per-stratum summaries.
    stratified: Dict[int, Dict[str, Any]] = {}
    for q in sorted({s for s in strata if s is not None}):
        idxs = [i for i, s in enumerate(strata) if s == q]
        if not idxs:
            continue
        stratified[q] = evaluate_attribution_set(
            [decompositions[i] for i in idxs],
            [expert_per_dive[i] for i in idxs],
            fault_categories=FAULT_TAXONOMY,
            top_k_values=cfg.eval.attribution_topk,
            match=args.match,
        )

    # ---- CF-XAI metrics aggregated from cache ---------------------
    pfc_primary = str(getattr(cfg.eval, "pfc_primary_for_acceptance", "edge"))
    cfxai_primary = _aggregate_cfxai_metrics(cache, pfc_primary=pfc_primary)
    log.info(
        "CF-XAI (pfc=%s): validity=%.3f sparsity=%.3f proximity=%.3f "
        "realism=%.4f completion=%.3f (n=%d)",
        pfc_primary,
        cfxai_primary.get("validity", float("nan")),
        cfxai_primary.get("sparsity", float("nan")),
        cfxai_primary.get("proximity", float("nan")),
        cfxai_primary.get("realism", float("nan")),
        cfxai_primary.get("completion", float("nan")),
        cfxai_primary.get("n", 0),
    )

    # Optionally compute the other PFC variant if the cache carries both.
    cfxai_legacy: Optional[Dict[str, float]] = None
    other_pfc = "lateral_accel" if pfc_primary == "edge" else "edge"
    sample_entry = next(
        (e for e in cache.values()
         if isinstance(e, dict) and e.get("records")), None,
    )
    if sample_entry and sample_entry["records"]:
        sample_rec = sample_entry["records"][0]
        if (f"pfc_{other_pfc}") in sample_rec:
            cfxai_legacy = _aggregate_cfxai_metrics(cache, pfc_primary=other_pfc)
            log.info(
                "CF-XAI (pfc=%s, secondary): realism=%.4f",
                other_pfc, cfxai_legacy.get("realism", float("nan")),
            )

    _write_attribution_csv(
        summary, stratified,
        cfxai_primary, cfxai_legacy,
        Path(args.output),
        pfc_primary=pfc_primary,
        n_gated=n_gated,
    )


# =============================================================================
# Subcommand: expert_study
# =============================================================================


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Cohen's κ for two raters on paired categorical labels.

    Implemented manually so we don't need sklearn here. Returns NaN for
    empty input; returns 1.0 in the degenerate all-agree-on-one-label case.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return float("nan")
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    p_e = 0.0
    for label in labels:
        pa = sum(1 for x in a if x == label) / n
        pb = sum(1 for x in b if x == label) / n
        p_e += pa * pb
    if abs(1.0 - p_e) < 1e-12:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def cmd_expert_study(args: argparse.Namespace) -> None:
    """Aggregate per-rater preference CSVs.

    Expected CSV columns: ``dive_id``, ``cf_id``, ``preferred`` (bool / 0/1).
    One file per rater; the file stem is used as the rater id.
    """
    raters: Dict[str, Dict[Tuple[str, str], bool]] = {}
    for path_str in args.rater_csvs:
        path = Path(path_str)
        rater_id = path.stem
        ratings: Dict[Tuple[str, str], bool] = {}
        with path.open("r") as f:
            for row in csv.DictReader(f):
                key = (row["dive_id"], row["cf_id"])
                ratings[key] = row["preferred"].strip().lower() in ("true", "1", "yes", "t")
        raters[rater_id] = ratings
        log.info("rater %s: %d ratings", rater_id, len(ratings))

    rater_ids = sorted(raters)
    if len(rater_ids) < 2:
        raise ValueError("need at least 2 raters for inter-rater agreement")

    common_keys = set.intersection(*[set(r.keys()) for r in raters.values()])
    log.info("%d ratings shared across all %d raters", len(common_keys), len(rater_ids))
    common_keys_list = sorted(common_keys)

    output: Dict[str, float] = {}

    # Per-rater preference rate
    for rid in rater_ids:
        prefs = [raters[rid][k] for k in common_keys_list]
        output[f"preferred_rate::{rid}"] = float(np.mean(prefs)) if prefs else float("nan")

    # Pairwise Cohen's κ
    for i, r1 in enumerate(rater_ids):
        for r2 in rater_ids[i + 1:]:
            a = [raters[r1][k] for k in common_keys_list]
            b = [raters[r2][k] for k in common_keys_list]
            output[f"kappa::{r1}::{r2}"] = float(cohen_kappa(a, b))

    # Majority-preferred rate
    majority: List[bool] = []
    for k in common_keys_list:
        votes = [raters[rid][k] for rid in rater_ids]
        majority.append(sum(votes) > len(votes) / 2)
    output["majority_preferred_rate"] = float(np.mean(majority)) if majority else float("nan")

    # Per-cf-type breakdown (e.g. "model" vs "random_baseline") so the
    # paper's headline number can be re-derived from the same CSV.
    by_cf_id: Dict[str, List[bool]] = {}
    for k in common_keys_list:
        _, cf_id = k
        votes = [raters[rid][k] for rid in rater_ids]
        is_majority_pref = sum(votes) > len(votes) / 2
        by_cf_id.setdefault(cf_id, []).append(is_majority_pref)
    for cf_id, prefs in sorted(by_cf_id.items()):
        output[f"majority_preferred_rate::{cf_id}"] = float(np.mean(prefs)) if prefs else float("nan")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in sorted(output.items()):
            w.writerow([k, v])
    log.info("wrote expert-study summary to %s", out_path)
    for k, v in sorted(output.items()):
        log.info("  %s = %.4f", k, v)


# =============================================================================
# CLI
# =============================================================================


def _pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="src.evaluate", description=__doc__)
    p.add_argument("--config", type=str, default=None, help="YAML Cfg override")
    p.add_argument("--seed-override", type=int, default=None)
    sub = p.add_subparsers(dest="task", required=True)

    aqa = sub.add_parser("aqa", help="AQA test-set evaluation")
    aqa.add_argument("--ckpt", type=str, required=True, help="AQA checkpoint")
    aqa.add_argument("--output", type=str, required=True, help="output CSV path")
    aqa.add_argument("--limit", type=int, default=None, help="cap test dives (debug)")

    cf = sub.add_parser("cf", help="run CF decomposition on FineDiving-HF dives")
    cf.add_argument("--aqa-ckpt", type=str, required=True)
    cf.add_argument("--diff-ckpt", type=str, required=True)
    cf.add_argument("--pose-proxy-ckpt", type=str, default=None,
                    help="trained PoseAQAProxy; without it the search runs but "
                         "numbers are meaningless (see docstring)")
    cf.add_argument("--output", type=str, required=True, help="CF cache JSON")
    cf.add_argument("--use-score-guidance", action="store_true",
                    help="enable gradient-guided diffusion sampling")
    cf.add_argument("--limit", type=int, default=None, help="cap dives (debug)")

    attr = sub.add_parser("attribution", help="score CF cache against FineDiving-HF")
    attr.add_argument("--decompositions", type=str, required=True, help="CF cache JSON from `cf`")
    attr.add_argument("--output", type=str, required=True, help="output CSV")
    attr.add_argument("--match", type=str, default="strict",
                      choices=("strict", "step_fault", "fault_only"),
                      help="set match criterion (default strict)")

    expert = sub.add_parser("expert_study", help="aggregate rater preference CSVs")
    expert.add_argument("--rater-csvs", type=str, nargs="+", required=True,
                        help="one CSV per rater; file stem used as rater id")
    expert.add_argument("--output", type=str, required=True)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = load_cfg(args)
    cfg.seed_everything()
    # Logging — evaluate.py writes to stderr by default; per-run log file
    # only when the user gives an explicit ``--output`` directory.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    device = _pick_device()
    log.info("evaluate task=%s device=%s fingerprint=%s",
             args.task, device, cfg.fingerprint())
    if args.task == "aqa":
        cmd_aqa(args, cfg, device)
    elif args.task == "cf":
        cmd_cf(args, cfg, device)
    elif args.task == "attribution":
        cmd_attribution(args, cfg)
    elif args.task == "expert_study":
        cmd_expert_study(args)
    else:
        parser.error(f"unknown task {args.task!r}")


if __name__ == "__main__":
    main()


__all__ = [
    "PoseAQAProxy",
    "load_aqa_model", "load_diffusion_model",
    "cohen_kappa",
    "cmd_aqa", "cmd_cf", "cmd_attribution", "cmd_expert_study",
]

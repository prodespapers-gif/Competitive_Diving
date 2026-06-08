"""metrics.py — Pure-function evaluation metrics.

Every metric in this module:
    - takes only NumPy arrays, lists of tuples, or scalars (no dataclasses,
      no model objects), so it stays importable without ``torch``;
    - returns a Python ``float`` or a ``dict`` of floats (and
      :class:`MetricWithCI` for bootstrapped quantities), so the outputs
      are trivially JSON-serialisable for the paper's tables;
    - is stateless and pure: same input always yields the same output.

Sections
--------
1.  AQA score metrics: Spearman ρ, Pearson, R-ℓ2 (per FineDiving / IJCV).
2.  Procedure segmentation: temporal IoU at given thresholds.
3.  Physical plausibility: EDGE PFC (verbatim) + the diving-adapted
    airborne-lateral-accel; a dispatcher reports one or both.
4.  Counterfactual metrics: faithfulness, minimality, plus the canonical
    CF-XAI vocabulary (validity / sparsity / proximity / realism /
    completion) as a thin renaming shim.
5.  Attribution metrics: set-level P/R/F1 vs. FineDiving-HF expert
    labels, per-dive top-K (the reference convention) and union top-K
    (the aggregate convention), per-fault, fully-explained rate.
6.  Stratification helpers (per score quintile / dive type / height).
7.  Bootstrap confidence intervals — generic and paired.
8.  High-level convenience: ``evaluate_aqa``, ``evaluate_attribution_set``,
    ``evaluate_physical_plausibility``.

References
----------
- AQA score metrics: FineDiving (Xu et al., CVPR 2022); MTL-AQA (Parmar &
  Tran Morris, CVPR 2019); CoRe (Yu et al., ICCV 2021).
- Procedure segmentation tIoU: FineDiving §5.1 (Sec. "AQA Performance"),
  threshold values ``{0.5, 0.75}`` reported in their Table 2.
- PFC (EDGE-style): Tseng et al., CVPR 2023, §4.4 "Physical Foot Contact".
  Implementation verified line-for-line against ``EDGE-main/eval/eval_pfc.py``
  (the released code base).
- Bootstrap CI methodology: Efron & Tibshirani, "An Introduction to the
  Bootstrap" (1993).
- CF-XAI vocabulary (validity, sparsity, proximity, realism): Verma et al.,
  "Counterfactual Explanations for Machine Learning: A Review", ACM CSUR
  2024 — adopted in this paper for cross-XAI comparability.

Changes from previous version
-----------------------------
- Issue 3 (paper-blocking): :func:`physical_foot_contact_edge` is now the
  verbatim port of EDGE-main/eval/eval_pfc.py. The old function is
  renamed to :func:`airborne_lateral_accel` (a defensible domain-adapted
  metric, but NOT EDGE PFC). :func:`physical_foot_contact_metric` is a
  dispatcher returning one or both, used by :mod:`evaluate`.
- Issue 12 (camera-ready): :func:`cf_metrics_cfxai` adds the canonical
  validity / sparsity / proximity / realism / completion vocabulary;
  the original :func:`faithfulness_stats` / :func:`minimality_stats`
  are preserved for backwards compatibility.
- Added :func:`segment_tiou` for the procedure-segmentation table
  (replaces the missing FineDiving tIoU column).
- :func:`attribution_top_k_per_dive` computes per-dive top-K F1 and
  averages; :func:`attribution_top_k_union` is the previous behaviour
  renamed for clarity.
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional,
    Sequence, Tuple, Union,
)

import numpy as np

log = logging.getLogger(__name__)

# A tuple is what :meth:`counterfactual.DecompositionResult.to_attribution_tuples`
# returns; declared as a type alias so the metric signatures stay readable.
AttributionTuple = Tuple[int, Sequence[str], str, float]   # (step, joints, fault, delta)


# =============================================================================
# MetricWithCI — a tiny dataclass for bootstrap output
# =============================================================================


@dataclass(frozen=True)
class MetricWithCI:
    """A point estimate together with a bootstrap confidence interval."""
    point: float
    ci_low: float
    ci_high: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}]"

    def as_dict(self) -> Dict[str, float]:
        return {"point": self.point, "ci_low": self.ci_low, "ci_high": self.ci_high}


# =============================================================================
# 1. AQA score metrics
# =============================================================================


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Compute average ranks with proper tie handling.

    For a vector with no ties this gives the usual 1..n ranks; for ties
    it returns the average of the tied positions (the standard
    "fractional" / "average" tie-breaking convention used by
    ``scipy.stats.rankdata`` with ``method='average'``).
    """
    sorter = np.argsort(x, kind="stable")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    # ``obs[i]`` is True iff the i-th sorted value starts a new tie group.
    obs = np.r_[True, x[sorter[1:]] != x[sorter[:-1]]]
    dense = obs.cumsum()[inv]
    # ``count`` accumulates positions where new groups start (plus n at the end).
    count = np.r_[np.nonzero(obs)[0], len(x)]
    # Average of the lower and upper bounds of each tie group's positions.
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman_correlation(
    predictions: Sequence[float], targets: Sequence[float],
) -> float:
    """Spearman rank correlation ρ with average-rank tie-breaking.

    Returns NaN for samples of size < 2 and for the degenerate
    constant-prediction or constant-target case.
    """
    p = np.asarray(predictions, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if len(p) != len(t):
        raise ValueError(f"length mismatch: {len(p)} vs {len(t)}")
    if len(p) < 2:
        return float("nan")
    rp = _average_ranks(p)
    rt = _average_ranks(t)
    rp -= rp.mean()
    rt -= rt.mean()
    num = float((rp * rt).sum())
    den = math.sqrt(float((rp ** 2).sum()) * float((rt ** 2).sum()))
    if den == 0.0:
        return float("nan")
    return num / den


def pearson_correlation(
    predictions: Sequence[float], targets: Sequence[float],
) -> float:
    """Pearson correlation. Reported alongside Spearman by some AQA papers."""
    p = np.asarray(predictions, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if len(p) != len(t):
        raise ValueError(f"length mismatch: {len(p)} vs {len(t)}")
    if len(p) < 2:
        return float("nan")
    pm, tm = p.mean(), t.mean()
    num = float(((p - pm) * (t - tm)).sum())
    den = math.sqrt(float(((p - pm) ** 2).sum()) * float(((t - tm) ** 2).sum()))
    if den == 0.0:
        return float("nan")
    return num / den


def r_l2(
    predictions: Sequence[float], targets: Sequence[float],
    score_range: Tuple[float, float],
) -> float:
    """Relative L2 distance, normalised by the score range.

    Defined as ``sqrt( mean( ((y - ŷ) / (y_max - y_min))^2 ) )``,
    matching the FineDiving / MTL-AQA convention (Eq. 1 of FineDiving;
    reported as R-ℓ2 ×100 in the paper's Tables 1-2). Lower is better.

    The reference uses the DATASET-wide score range as the normaliser,
    NOT the per-batch range. Pass the canonical (score_min, score_max)
    from ``DataCfg`` — not statistics computed on this batch.
    """
    p = np.asarray(predictions, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if len(p) != len(t):
        raise ValueError(f"length mismatch: {len(p)} vs {len(t)}")
    score_min, score_max = score_range
    span = score_max - score_min
    if span <= 0:
        return float("nan")
    return float(np.sqrt(np.mean(((p - t) / span) ** 2)))


def r_l2_per_action(
    predictions: Sequence[float], targets: Sequence[float],
    action_types: Sequence[str],
) -> float:
    """R-ℓ2 computed with per-action-type score ranges, then averaged.

    Some AQA papers prefer this normaliser because a 1-point error on a
    DD-3.5 dive isn't the same as on a DD-1.5 dive. Each action type
    gets its own ``[min, max]`` range derived from the targets for
    that type, then the overall metric averages the per-type R-ℓ2.
    """
    p = np.asarray(predictions, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if len(p) != len(t) or len(t) != len(action_types):
        raise ValueError("length mismatch among predictions / targets / action_types")
    by_action: Dict[str, List[int]] = defaultdict(list)
    for i, a in enumerate(action_types):
        by_action[a].append(i)
    per_action: List[float] = []
    for action, idxs in by_action.items():
        if len(idxs) < 2:
            continue   # need >= 2 to define a non-degenerate range
        idx = np.array(idxs)
        t_a, p_a = t[idx], p[idx]
        span = float(t_a.max() - t_a.min())
        if span <= 0:
            continue
        per_action.append(float(np.sqrt(np.mean(((p_a - t_a) / span) ** 2))))
    if not per_action:
        return float("nan")
    return float(np.mean(per_action))


# =============================================================================
# 2. Procedure segmentation — temporal IoU
# =============================================================================


def segment_tiou(
    pred_boundaries: Sequence[Tuple[int, int]],
    gt_boundaries: Sequence[Tuple[int, int]],
) -> float:
    """Average temporal-IoU over matched (predicted, ground-truth) segments.

    The L+1 procedure steps each define an interval [start, end]. With
    the same number of predicted and GT segments per dive, IoU is
    computed per index and averaged. Returns NaN if any input is empty
    or shapes disagree.

    Matches the reference's ``segment_iou`` aggregation
    (FineDiving-main/tools/helper.py:181-188).
    """
    if len(pred_boundaries) == 0 or len(gt_boundaries) == 0:
        return float("nan")
    if len(pred_boundaries) != len(gt_boundaries):
        raise ValueError(
            f"segment count mismatch: pred={len(pred_boundaries)} "
            f"vs gt={len(gt_boundaries)}"
        )
    ious = []
    for (ps, pe), (gs, ge) in zip(pred_boundaries, gt_boundaries):
        # Convention: closed-closed; +1 for the boundary frame.
        inter = max(0, min(pe, ge) - max(ps, gs) + 1)
        union = max(pe, ge) - min(ps, gs) + 1
        if union <= 0:
            ious.append(0.0)
            continue
        ious.append(inter / union)
    return float(np.mean(ious))


def boundaries_from_transitions(
    transitions: Sequence[int], clip_num_frames: int,
) -> List[Tuple[int, int]]:
    """Build L+1 [start, end] segments from L step-transition frame indices.

    Step 0 spans ``[0, transitions[0]]``; step k spans
    ``[transitions[k-1]+1, transitions[k]]`` for 1 ≤ k < L; the last
    step spans ``[transitions[-1]+1, clip_num_frames-1]``. The reference
    helper.py uses the same convention.
    """
    if clip_num_frames <= 0:
        return []
    boundaries: List[Tuple[int, int]] = []
    prev = 0
    for t in transitions:
        t = int(t)
        boundaries.append((prev, t))
        prev = t + 1
    boundaries.append((prev, clip_num_frames - 1))
    return boundaries


def segment_tiou_at_threshold(
    pred_segments_per_dive: Sequence[Sequence[Tuple[int, int]]],
    gt_segments_per_dive: Sequence[Sequence[Tuple[int, int]]],
    threshold: float,
) -> float:
    """Fraction of dives whose average per-segment IoU exceeds ``threshold``.

    The reference FineDiving Table 2 reports tIoU@0.5 and tIoU@0.75 —
    use this with threshold ∈ {0.5, 0.75} to reproduce those columns.
    """
    if len(pred_segments_per_dive) != len(gt_segments_per_dive):
        raise ValueError("dive count mismatch")
    if len(pred_segments_per_dive) == 0:
        return float("nan")
    hits = 0
    for pred, gt in zip(pred_segments_per_dive, gt_segments_per_dive):
        iou = segment_tiou(pred, gt)
        if not math.isnan(iou) and iou >= threshold:
            hits += 1
    return hits / len(pred_segments_per_dive)


# =============================================================================
# 3. Physical plausibility
# =============================================================================
#
# Two metrics live here:
#
# (a) physical_foot_contact_edge — the EDGE PFC formula, ported verbatim
#     from EDGE-main/eval/eval_pfc.py:10-46. Dimensionless, per-clip
#     scalar (the 10000× scaling is applied at population aggregation,
#     not per-clip). The product of (L-foot slowness, R-foot slowness,
#     normalised root acceleration) penalises motion that accelerates
#     the root without a planted foot.
#
# (b) airborne_lateral_accel — a defensible diving-specific metric: the
#     mean horizontal acceleration magnitude in m/s² across airborne
#     frames (where airborne is taken from the diffusion model's binary
#     contact predictions). In flight, gravity is the only external
#     force; lateral acceleration indicates a physically impossible
#     trajectory. This is what the previous codebase computed under the
#     misleading name ``physical_foot_contact``. Renamed for honesty.
#
# Dispatcher: physical_foot_contact_metric routes to one or both based
# on cfg.cf.pfc_metric.
# -----------------------------------------------------------------------------


# SMPL joint indices used by EDGE's PFC. See EDGE-main/eval/eval_pfc.py:33.
#   7 = left ankle, 10 = left foot/toe
#   8 = right ankle, 11 = right foot/toe
EDGE_FOOT_JOINTS: Tuple[int, int, int, int] = (7, 10, 8, 11)


def physical_foot_contact_edge(
    joints_3d: np.ndarray,
    fps: float = 30.0,
    up_axis: int = 1,
    foot_indices: Tuple[int, int, int, int] = EDGE_FOOT_JOINTS,
    eps: float = 1e-12,
) -> float:
    """EDGE Physical Foot Contact, ported verbatim from ``eval_pfc.py``.

    Implements the per-clip score::

        root_v = diff(joints_3d[:, 0]) / dt
        root_a = diff(root_v) / dt
        root_a[:, up] = max(root_a[:, up], 0)         # clip downward accel
        root_a = ||root_a||                           # (S-2,)
        root_a /= root_a.max()                        # per-clip normalisation
        foot_v = ||diff(joints_3d[:, foot, flat])||   # (S-2, 4)
        foot_min_L = min(foot_v[:, 0], foot_v[:, 1])
        foot_min_R = min(foot_v[:, 2], foot_v[:, 3])
        score = mean( foot_min_L * foot_min_R * root_a )

    The result is dimensionless and per-clip. To reproduce EDGE's Table
    columns, take ``np.mean(scores) * 10000`` over the dataset (see
    :func:`physical_foot_contact_edge_dataset`).

    Args:
        joints_3d: ``(S, J, 3)`` per-frame 3D joint positions. ``J`` must
            be at least ``max(foot_indices) + 1``; root joint is index 0.
        fps: source frame rate (default 30 — the EDGE convention).
        up_axis: axis index of the vertical direction. EDGE uses z=2;
            SMPL pipelines commonly use y=1. The default here is 1 to
            match the rest of this codebase. The other two axes are
            treated as "flat" (horizontal).
        foot_indices: SMPL joint indices for (L-ankle, L-toe, R-ankle,
            R-toe). EDGE values used by default.
        eps: small constant added to the max-acceleration normaliser to
            keep the division stable on near-static clips.

    Returns:
        Per-clip PFC score (float). Lower is more physically consistent.
        Returns 0.0 if the clip is too short (< 3 frames) or has zero
        acceleration anywhere.
    """
    if joints_3d.ndim != 3 or joints_3d.shape[-1] != 3:
        raise ValueError(
            f"joints_3d must be (S, J, 3); got {joints_3d.shape}"
        )
    if up_axis not in (0, 1, 2):
        raise ValueError(f"up_axis must be 0, 1, or 2; got {up_axis}")
    if max(foot_indices) >= joints_3d.shape[1]:
        raise ValueError(
            f"foot_indices {foot_indices} exceed joint count {joints_3d.shape[1]}"
        )
    S = joints_3d.shape[0]
    if S < 3:
        return 0.0

    flat_dirs = [i for i in range(3) if i != up_axis]
    dt = 1.0 / fps

    # ---- root acceleration (joint 0 = pelvis) ----
    root = joints_3d[:, 0, :]                                # (S, 3)
    root_v = (root[1:] - root[:-1]) / dt                     # (S-1, 3)
    root_a = (root_v[1:] - root_v[:-1]) / dt                 # (S-2, 3)
    # Clamp ONLY the up direction to non-negative — gravity-resisted
    # downward acceleration is "expected", so it's nulled out.
    root_a[:, up_axis] = np.maximum(root_a[:, up_axis], 0.0)
    root_a_mag = np.linalg.norm(root_a, axis=-1)             # (S-2,)
    scaling = root_a_mag.max()
    # Reference (eval_pfc.py:30) divides by max unconditionally. We add
    # a minimal == 0 guard so a truly static clip returns 0 instead of
    # NaN; any scaling > 0 (including float-precision noise) follows the
    # reference's behaviour exactly.
    if scaling == 0.0:
        return 0.0
    root_a_mag = root_a_mag / scaling                        # in [0, 1]

    # ---- foot horizontal velocity ----
    feet = joints_3d[:, list(foot_indices), :]               # (S, 4, 3)
    # NB: EDGE uses (feet[2:] - feet[1:-1]) — a forward diff at indices 1..S-2.
    # This aligns with root_a, which lives at indices 1..S-2.
    foot_v = np.linalg.norm(
        feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1,
    )                                                         # (S-2, 4)

    # ---- per-side foot min (slowness proxy for contact) ----
    foot_min_L = np.minimum(foot_v[:, 0], foot_v[:, 1])
    foot_min_R = np.minimum(foot_v[:, 2], foot_v[:, 3])

    # ---- the PFC loss ----
    foot_loss = foot_min_L * foot_min_R * root_a_mag         # (S-2,)
    return float(foot_loss.mean())


def physical_foot_contact_edge_dataset(
    joints_3d_per_clip: Sequence[np.ndarray],
    fps: float = 30.0,
    up_axis: int = 1,
    foot_indices: Tuple[int, int, int, int] = EDGE_FOOT_JOINTS,
    scaling: float = 10000.0,
) -> float:
    """Dataset-level EDGE PFC: ``mean(per-clip-pfc) × 10000``.

    Reproduces EDGE-main/eval/eval_pfc.py's reported value verbatim.
    Pass the per-clip 3D joint sequences; the function aggregates and
    applies the published × 10000 scale.
    """
    scores = [
        physical_foot_contact_edge(
            j, fps=fps, up_axis=up_axis, foot_indices=foot_indices,
        )
        for j in joints_3d_per_clip
    ]
    if not scores:
        return float("nan")
    return float(np.mean(scores) * scaling)


def airborne_lateral_accel(
    com_positions: np.ndarray,
    contacts: np.ndarray,
    fps: float = 30.0,
    up_axis: int = 1,
) -> float:
    """Mean horizontal-acceleration magnitude across airborne frames.

    This is the diving-adapted metric: in flight, gravity is the only
    external force (vertical), so any horizontal (lateral) COM
    acceleration indicates a physically impossible trajectory. Unlike
    EDGE PFC, this metric:
        - uses an explicit binary contact signal (e.g. the diffusion
          model's contact predictions) to decide which frames count as
          airborne, so takeoff and entry are excluded automatically;
        - is in m/s², not dimensionless;
        - does not penalise vertical acceleration (gravity is fine).

    Reported alongside EDGE PFC for domain-specific intuition. The
    canonical metric for cross-paper comparison is EDGE PFC.

    Args:
        com_positions: ``(T, 3)`` COM coordinates. ``up_axis`` is the
            vertical (gravity) axis; the other two axes are horizontal.
        contacts: ``(T, F)`` binary foot contact flags. Any non-zero
            entry in a row marks that frame as grounded.
        fps: source frame rate.
        up_axis: vertical axis index (0/1/2). Default 1 to match the
            rest of this codebase's SMPL convention.

    Returns:
        Mean ‖a_lateral‖ over airborne frames in m/s². Lower is more
        physically consistent. Returns 0.0 if no airborne frames or if
        the clip is too short to compute acceleration.
    """
    if com_positions.ndim != 2 or com_positions.shape[1] != 3:
        raise ValueError(f"com_positions must be (T, 3); got {com_positions.shape}")
    if up_axis not in (0, 1, 2):
        raise ValueError(f"up_axis must be 0, 1, or 2; got {up_axis}")
    T = com_positions.shape[0]
    if T < 3:
        return 0.0
    dt = 1.0 / fps
    flat_dirs = [i for i in range(3) if i != up_axis]
    velocity = np.gradient(com_positions, dt, axis=0)
    accel = np.gradient(velocity, dt, axis=0)
    lateral_mag = np.sqrt(sum(accel[:, d] ** 2 for d in flat_dirs))
    grounded = contacts.sum(axis=-1) > 0
    airborne = ~grounded
    if not airborne.any():
        return 0.0
    return float(lateral_mag[airborne].mean())


def physical_foot_contact_metric(
    metric: str,
    *,
    joints_3d: Optional[np.ndarray] = None,
    com_positions: Optional[np.ndarray] = None,
    contacts: Optional[np.ndarray] = None,
    fps: float = 30.0,
    up_axis: int = 1,
) -> Dict[str, float]:
    """Dispatcher: returns one or both PFC variants per ``metric``.

    Designed for ``cfg.cf.pfc_metric ∈ {"edge", "lateral_accel", "both"}``.

    Args:
        metric: which variant(s) to compute.
        joints_3d: required for the ``edge`` variant; ignored otherwise.
        com_positions, contacts: required for the ``lateral_accel``
            variant; ignored otherwise.
        fps: source frame rate (shared).
        up_axis: vertical axis (shared).

    Returns:
        A dict containing the requested metric(s) under their canonical
        keys: ``pfc_edge`` and/or ``pfc_lateral_accel``.
    """
    if metric not in {"edge", "lateral_accel", "both"}:
        raise ValueError(
            f"unknown pfc_metric: {metric!r}; "
            "expected 'edge', 'lateral_accel', or 'both'"
        )
    out: Dict[str, float] = {}
    if metric in {"edge", "both"}:
        if joints_3d is None:
            raise ValueError(
                f"pfc_metric={metric!r} requires joints_3d (shape (S, J, 3))"
            )
        out["pfc_edge"] = physical_foot_contact_edge(
            joints_3d, fps=fps, up_axis=up_axis,
        )
    if metric in {"lateral_accel", "both"}:
        if com_positions is None or contacts is None:
            raise ValueError(
                f"pfc_metric={metric!r} requires com_positions and contacts"
            )
        out["pfc_lateral_accel"] = airborne_lateral_accel(
            com_positions, contacts, fps=fps, up_axis=up_axis,
        )
    return out


# Backwards-compatible alias. The previous codebase exported the
# diving-adapted metric under the (misleading) name ``physical_foot_contact``.
# Keep the symbol so old callers don't break, but mark it deprecated.
def physical_foot_contact(
    com_positions: np.ndarray,
    contacts: np.ndarray,
    fps: float = 30.0,
    diving_adapt: bool = True,
    up_axis: int = 1,
) -> float:
    """DEPRECATED — alias for :func:`airborne_lateral_accel`.

    The previous codebase exported the diving-adapted metric under this
    name. It was NEVER an implementation of EDGE PFC; the EDGE formula
    lives in :func:`physical_foot_contact_edge`. Use the dispatcher
    :func:`physical_foot_contact_metric` to choose explicitly.
    """
    _ = diving_adapt   # kept for API stability (no-op)
    return airborne_lateral_accel(
        com_positions, contacts, fps=fps, up_axis=up_axis,
    )


# =============================================================================
# 4. Counterfactual metrics
# =============================================================================


def faithfulness_stats(
    actual_uplifts: Sequence[float], claimed_uplifts: Sequence[float],
    tolerance: float = 0.10,
) -> Dict[str, float]:
    """Faithfulness of a set of counterfactuals.

    For each CF, ``|actual_uplift - claimed_uplift|`` is the
    faithfulness error; we report mean, std, max, and the fraction
    whose error is within ``tolerance``. The within-tolerance rate is
    what the paper's "faithfulness rate" column reports.
    """
    a = np.asarray(actual_uplifts, dtype=np.float64)
    c = np.asarray(claimed_uplifts, dtype=np.float64)
    if len(a) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "max": float("nan"), "within_tol_rate": float("nan"),
                "n": 0}
    err = np.abs(a - c)
    return {
        "mean": float(err.mean()),
        "std": float(err.std()),
        "max": float(err.max()),
        "within_tol_rate": float((err <= tolerance).mean()),
        "n": int(len(err)),
    }


def minimality_stats(norms: Sequence[float]) -> Dict[str, float]:
    """Aggregate statistics over per-CF minimality norms.

    The :class:`Counterfactual` dataclass carries ``minimality_norm``
    per CF; passing them all here gives a population summary for the
    paper's "minimality" column.
    """
    n = np.asarray(norms, dtype=np.float64)
    if len(n) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "n": 0}
    return {
        "mean": float(n.mean()),
        "std": float(n.std()),
        "median": float(np.median(n)),
        "p25": float(np.percentile(n, 25)),
        "p75": float(np.percentile(n, 75)),
        "n": int(len(n)),
    }


def cf_metrics_cfxai(
    actual_uplifts: Sequence[float],
    claimed_uplifts: Sequence[float],
    pose_l2_norms: Sequence[float],
    n_joint_groups_changed: Sequence[int],
    pfc_scores: Sequence[float],
    completed_flags: Sequence[bool],
    faithfulness_tolerance: float = 0.10,
) -> Dict[str, float]:
    """The five CF-XAI canonical columns + our paper's "completion" column.

    Verma et al. (ACM CSUR 2024) standardise the CF evaluation
    vocabulary as ``{validity, sparsity, proximity, realism}``. We
    align with that vocabulary in the paper's tables for
    cross-XAI comparability, and add a fifth column ``completion``
    that is specific to this paper's iterative half-point decomposition.

    Mapping from internal pipeline → CF-XAI vocabulary:
        validity   ← did the CF reach the target uplift within tolerance?
                     (= faithfulness within-tol rate, but the name in the
                     CF-XAI literature is "validity")
        sparsity   ← #joint-groups changed (lower is sparser)
        proximity  ← pose ℓ2 to the query (lower is closer)
        realism    ← physical plausibility score (PFC; lower is more real)
        completion ← (this paper) did the iterative decomposition close
                     the entire score gap?

    Returns:
        Flat dict with the five canonical metrics (population means /
        rates). Each row of the paper's counterfactual table is one
        call to this function.
    """
    fs = faithfulness_stats(
        actual_uplifts, claimed_uplifts, tolerance=faithfulness_tolerance,
    )
    sparsity = np.asarray(n_joint_groups_changed, dtype=np.float64)
    proximity = np.asarray(pose_l2_norms, dtype=np.float64)
    realism = np.asarray(pfc_scores, dtype=np.float64)
    return {
        "validity": fs["within_tol_rate"],
        "sparsity": float(sparsity.mean()) if len(sparsity) else float("nan"),
        "proximity": float(proximity.mean()) if len(proximity) else float("nan"),
        "realism": float(realism.mean()) if len(realism) else float("nan"),
        "completion": fully_explained_rate(completed_flags),
        "n": int(len(actual_uplifts)),
    }


# =============================================================================
# 5. Attribution metrics — set-level vs FineDiving-HF expert labels
# =============================================================================


def _canon_attribution_tuple(t: AttributionTuple, match: str) -> Tuple:
    """Canonicalise an attribution tuple under a given match criterion.

    Three criteria, in increasing strictness:
        - ``fault_only``:  (fault,)
        - ``step_fault``:  (step, fault)
        - ``strict``:      (step, joints_as_frozenset, fault)
    The ``delta`` field is dropped — it's always 0.5 by construction.
    """
    step, joints, fault, _delta = int(t[0]), t[1], str(t[2]), float(t[3])
    if match == "fault_only":
        return (fault,)
    if match == "step_fault":
        return (step, fault)
    if match == "strict":
        return (step, frozenset(str(j) for j in joints), fault)
    raise ValueError(f"unknown match criterion: {match!r}")


def attribution_set_metrics(
    predicted: Sequence[AttributionTuple],
    expert: Sequence[AttributionTuple],
    match: str = "strict",
) -> Dict[str, float]:
    """Set-level precision / recall / F1 over attribution tuples.

    Sets are unordered, so the ordering of half-points in the
    decomposition is ignored; ordering-aware metrics live in the
    ``attribution_top_k_*`` functions below.

    Args:
        predicted: tuples from
            :meth:`counterfactual.DecompositionResult.to_attribution_tuples`.
        expert: tuples from the FineDiving-HF annotation.
        match: ``strict`` | ``step_fault`` | ``fault_only`` — see
            :func:`_canon_attribution_tuple`.

    Returns:
        dict with ``precision``, ``recall``, ``f1``, ``tp``, ``fp``,
        ``fn``, ``n_pred``, ``n_expert``.
    """
    pred_set = {_canon_attribution_tuple(t, match) for t in predicted}
    expert_set = {_canon_attribution_tuple(t, match) for t in expert}
    tp = len(pred_set & expert_set)
    fp = len(pred_set - expert_set)
    fn = len(expert_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred": len(pred_set), "n_expert": len(expert_set),
    }


def attribution_top_k_per_dive(
    predicted_per_dive: Sequence[Sequence[AttributionTuple]],
    expert_per_dive: Sequence[Sequence[AttributionTuple]],
    k: int, match: str = "strict",
) -> Dict[str, float]:
    """Per-dive top-K F1, averaged over dives.

    For each dive: take the first K predicted tuples; compute set
    metrics against THAT dive's expert tuples; then average the
    per-dive precision / recall / F1. This is the standard top-K
    convention used in attribution-evaluation papers.

    Use this for the paper's top-K column; the alternative
    :func:`attribution_top_k_union` is a coarser aggregate that's
    occasionally useful but doesn't isolate per-dive accuracy.
    """
    if len(predicted_per_dive) != len(expert_per_dive):
        raise ValueError("dive counts must match")
    if not predicted_per_dive:
        return {"precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "n_dives": 0}
    precisions, recalls, f1s = [], [], []
    for pred, exp in zip(predicted_per_dive, expert_per_dive):
        head = list(pred)[:k]
        m = attribution_set_metrics(head, exp, match=match)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
        f1s.append(m["f1"])
    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "n_dives": len(predicted_per_dive),
    }


def attribution_top_k_union(
    predicted_per_dive: Sequence[Sequence[AttributionTuple]],
    expert_per_dive: Sequence[Sequence[AttributionTuple]],
    k: int, match: str = "strict",
) -> Dict[str, float]:
    """Top-K computed over the UNION of (first-K predictions, all experts).

    For each dive: take the first K predicted tuples; pool ALL of them
    into one big set; pool ALL expert tuples into another big set;
    compute set P/R/F1. This dilutes per-dive accuracy with global
    set-membership but is sometimes useful for reporting one number.

    Prefer :func:`attribution_top_k_per_dive` for the headline table.
    """
    if len(predicted_per_dive) != len(expert_per_dive):
        raise ValueError("dive counts must match")
    all_pred: List[AttributionTuple] = []
    all_expert: List[AttributionTuple] = []
    for pred, exp in zip(predicted_per_dive, expert_per_dive):
        all_pred.extend(list(pred)[:k])
        all_expert.extend(exp)
    return attribution_set_metrics(all_pred, all_expert, match=match)


# Backwards-compatible alias for the previous function name.
def attribution_top_k(
    predicted_ordered: Sequence[AttributionTuple],
    expert: Sequence[AttributionTuple],
    k: int, match: str = "strict",
) -> Dict[str, float]:
    """DEPRECATED — use :func:`attribution_top_k_per_dive` or
    :func:`attribution_top_k_union` to be explicit about aggregation.

    The previous semantics: take the first K of a single ordered list
    and compare to a single expert list. Equivalent to
    :func:`attribution_top_k_per_dive` with a single dive.
    """
    head = list(predicted_ordered)[:k]
    return attribution_set_metrics(head, expert, match=match)


def per_fault_metrics(
    predicted: Sequence[AttributionTuple],
    expert: Sequence[AttributionTuple],
    fault_categories: Sequence[str],
    match: str = "strict",
) -> Dict[str, Dict[str, float]]:
    """Per-fault precision/recall/F1.

    Useful to identify which fault categories the attribution pipeline
    handles well vs. poorly. Empty for fault categories with zero
    predictions *and* zero expert labels in this batch.
    """
    out: Dict[str, Dict[str, float]] = {}
    for fault in fault_categories:
        p = [t for t in predicted if t[2] == fault]
        e = [t for t in expert if t[2] == fault]
        if not p and not e:
            continue
        out[fault] = attribution_set_metrics(p, e, match=match)
    return out


def fully_explained_rate(completed_flags: Sequence[bool]) -> float:
    """Fraction of dives whose decomposition closed the entire score gap.

    Pass in :attr:`counterfactual.DecompositionResult.completed` for
    each dive. The complement is the partial-decomposition rate.
    """
    if not completed_flags:
        return float("nan")
    return float(np.mean([bool(c) for c in completed_flags]))


# =============================================================================
# 6. Stratification helpers
# =============================================================================


def stratify(
    values: Sequence[Any], strata: Sequence[Any],
) -> Dict[Any, List[Any]]:
    """Group ``values`` by ``strata`` and return ``{stratum: [values]}``."""
    if len(values) != len(strata):
        raise ValueError(f"length mismatch: {len(values)} vs {len(strata)}")
    out: Dict[Any, List[Any]] = defaultdict(list)
    for v, s in zip(values, strata):
        out[s].append(v)
    return dict(out)


def stratified_summary(
    values: Sequence[float], strata: Sequence[Any],
    summary_fn: Callable[[np.ndarray], float] = np.mean,
) -> Dict[Any, float]:
    """Apply ``summary_fn`` to each stratum's values."""
    grouped = stratify(values, strata)
    return {
        s: float(summary_fn(np.asarray(g, dtype=np.float64)))
        for s, g in grouped.items()
    }


# =============================================================================
# 7. Bootstrap confidence intervals
# =============================================================================


def bootstrap_metric(
    metric_fn: Callable[..., float], *arrays: Sequence[Any],
    n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 0,
) -> MetricWithCI:
    """Generic non-parametric bootstrap CI for a metric over paired arrays.

    Resampling preserves pairing: the same set of indices is applied to
    every array, so paired-sample relationships (predictions ↔ targets,
    actual ↔ claimed, etc.) are preserved per bootstrap iterate.

    NaN samples — produced by degenerate resamples, e.g. all-tied
    Spearman — are filtered out and logged. If too few valid samples
    remain (< 10), the CI is reported as NaN.

    Args:
        metric_fn: callable mapping ``*arrays → float``.
        *arrays: paired data arrays, all of the same length.
        n_bootstrap: number of resamples. ``EvalCfg.spearman_bootstrap``
            defaults to 1000.
        confidence: e.g. 0.95 for a 95% CI.
        seed: RNG seed so bootstraps are reproducible.

    Returns:
        :class:`MetricWithCI` with the point estimate (computed on the
        original arrays) and percentile-method CI bounds.
    """
    if not arrays:
        raise ValueError("at least one array required")
    arrays_np: Tuple[np.ndarray, ...] = tuple(np.asarray(a) for a in arrays)
    n = len(arrays_np[0])
    if any(len(a) != n for a in arrays_np):
        raise ValueError("all arrays must have the same length")
    rng = np.random.default_rng(seed)
    point = float(metric_fn(*arrays_np))
    valid: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resampled = tuple(a[idx] for a in arrays_np)
        try:
            v = float(metric_fn(*resampled))
            if not math.isnan(v) and not math.isinf(v):
                valid.append(v)
        except (ValueError, ZeroDivisionError):
            continue
    if len(valid) < 10:
        log.warning(
            "bootstrap_metric: only %d valid samples (out of %d); CI unreliable",
            len(valid), n_bootstrap,
        )
        return MetricWithCI(point=point, ci_low=float("nan"), ci_high=float("nan"))
    samples = np.asarray(valid)
    alpha = 1.0 - confidence
    return MetricWithCI(
        point=point,
        ci_low=float(np.percentile(samples, 100 * alpha / 2)),
        ci_high=float(np.percentile(samples, 100 * (1 - alpha / 2))),
    )


def paired_bootstrap_difference(
    method_a: Sequence[float], method_b: Sequence[float],
    metric_fn: Callable[..., float] = np.mean,
    n_bootstrap: int = 1000, confidence: float = 0.95, seed: int = 0,
) -> MetricWithCI:
    """Paired bootstrap CI on the difference ``metric_fn(B) - metric_fn(A)``.

    Used for paired comparisons (e.g. TSA vs STSA on the same dives,
    counterfactual-faithfulness with vs. without score guidance).
    Resampling pairs together so the comparison is paired (not
    independent two-sample).
    """
    a = np.asarray(method_a, dtype=np.float64)
    b = np.asarray(method_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        return MetricWithCI(
            point=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
        )
    rng = np.random.default_rng(seed)
    point = float(metric_fn(b) - metric_fn(a))
    valid: List[float] = []
    n = len(a)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            v = float(metric_fn(b[idx]) - metric_fn(a[idx]))
            if not math.isnan(v) and not math.isinf(v):
                valid.append(v)
        except (ValueError, ZeroDivisionError):
            continue
    if len(valid) < 10:
        return MetricWithCI(
            point=point, ci_low=float("nan"), ci_high=float("nan"),
        )
    samples = np.asarray(valid)
    alpha = 1.0 - confidence
    return MetricWithCI(
        point=point,
        ci_low=float(np.percentile(samples, 100 * alpha / 2)),
        ci_high=float(np.percentile(samples, 100 * (1 - alpha / 2))),
    )


# =============================================================================
# 8. High-level convenience — multi-metric summaries
# =============================================================================


def evaluate_aqa(
    predictions: Sequence[float], targets: Sequence[float],
    score_range: Tuple[float, float],
    action_types: Optional[Sequence[str]] = None,
    n_bootstrap: int = 1000, seed: int = 0,
) -> Dict[str, Any]:
    """The full AQA-metric pack for one set of (prediction, target) pairs.

    Returns a flat dict with bootstrapped Spearman, Pearson, R-ℓ2
    (dataset-normalised) and optionally per-action R-ℓ2.
    """
    out: Dict[str, Any] = {}
    out["spearman"] = bootstrap_metric(
        spearman_correlation, predictions, targets,
        n_bootstrap=n_bootstrap, seed=seed,
    )
    out["pearson"] = bootstrap_metric(
        pearson_correlation, predictions, targets,
        n_bootstrap=n_bootstrap, seed=seed + 1,
    )
    out["r_l2"] = bootstrap_metric(
        lambda p, t: r_l2(p, t, score_range), predictions, targets,
        n_bootstrap=n_bootstrap, seed=seed + 2,
    )
    if action_types is not None:
        out["r_l2_per_action"] = bootstrap_metric(
            lambda p, t, a: r_l2_per_action(p, t, list(a)),
            predictions, targets, action_types,
            n_bootstrap=n_bootstrap, seed=seed + 3,
        )
    return out


def evaluate_segmentation(
    pred_segments_per_dive: Sequence[Sequence[Tuple[int, int]]],
    gt_segments_per_dive: Sequence[Sequence[Tuple[int, int]]],
    thresholds: Sequence[float] = (0.5, 0.75),
) -> Dict[str, float]:
    """Procedure-segmentation evaluation pack.

    Returns:
        - ``mean_tiou``: average per-segment IoU over all dives
        - ``tiou_at_<threshold>``: fraction of dives with avg-IoU above
          each threshold (FineDiving Table 2 convention)
    """
    if len(pred_segments_per_dive) != len(gt_segments_per_dive):
        raise ValueError("dive counts must match")
    if not pred_segments_per_dive:
        return {"mean_tiou": float("nan")}
    per_dive_iou = [
        segment_tiou(p, g)
        for p, g in zip(pred_segments_per_dive, gt_segments_per_dive)
    ]
    per_dive_iou_np = np.asarray(per_dive_iou, dtype=np.float64)
    out: Dict[str, float] = {
        "mean_tiou": float(np.nanmean(per_dive_iou_np)),
        "n_dives": len(pred_segments_per_dive),
    }
    for thr in thresholds:
        key = f"tiou_at_{thr:g}"
        out[key] = segment_tiou_at_threshold(
            pred_segments_per_dive, gt_segments_per_dive, threshold=thr,
        )
    return out


def evaluate_physical_plausibility(
    joints_3d_per_clip: Optional[Sequence[np.ndarray]] = None,
    com_positions_per_clip: Optional[Sequence[np.ndarray]] = None,
    contacts_per_clip: Optional[Sequence[np.ndarray]] = None,
    *,
    metric: str = "both",
    fps: float = 30.0,
    up_axis: int = 1,
    n_bootstrap: int = 1000, seed: int = 0,
    edge_scaling: float = 10000.0,
) -> Dict[str, Any]:
    """Population PFC pack — both variants with bootstrap CIs.

    Args:
        joints_3d_per_clip: required for the ``edge`` variant.
        com_positions_per_clip, contacts_per_clip: required for the
            ``lateral_accel`` variant. Paired by index.
        metric: ``"edge"`` | ``"lateral_accel"`` | ``"both"`` — see
            :func:`physical_foot_contact_metric`.
        fps, up_axis: shared geometry kwargs.
        n_bootstrap, seed: bootstrap controls.
        edge_scaling: the EDGE convention multiplies the population
            mean by 10000 — applied only to the population point
            estimate, NOT to the bootstrap quantiles (which already
            scale linearly with the mean, so quantile×10000 gives the
            scaled CI exactly).

    Returns:
        dict with up to two :class:`MetricWithCI` entries:
        ``pfc_edge`` (×10000 scaled, dimensionless) and
        ``pfc_lateral_accel`` (m/s²).
    """
    if metric not in {"edge", "lateral_accel", "both"}:
        raise ValueError(f"unknown pfc_metric: {metric!r}")
    out: Dict[str, Any] = {}
    if metric in {"edge", "both"}:
        if joints_3d_per_clip is None:
            raise ValueError("EDGE PFC requires joints_3d_per_clip")
        scores = [
            physical_foot_contact_edge(
                j, fps=fps, up_axis=up_axis,
            )
            for j in joints_3d_per_clip
        ]
        scaled = np.asarray(scores, dtype=np.float64) * edge_scaling
        out["pfc_edge"] = bootstrap_metric(
            lambda s: float(np.mean(s)), scaled,
            n_bootstrap=n_bootstrap, seed=seed,
        )
    if metric in {"lateral_accel", "both"}:
        if com_positions_per_clip is None or contacts_per_clip is None:
            raise ValueError(
                "lateral_accel requires com_positions_per_clip and contacts_per_clip"
            )
        if len(com_positions_per_clip) != len(contacts_per_clip):
            raise ValueError("com_positions and contacts must align")
        scores = [
            airborne_lateral_accel(
                c, k, fps=fps, up_axis=up_axis,
            )
            for c, k in zip(com_positions_per_clip, contacts_per_clip)
        ]
        out["pfc_lateral_accel"] = bootstrap_metric(
            lambda s: float(np.mean(s)),
            np.asarray(scores, dtype=np.float64),
            n_bootstrap=n_bootstrap, seed=seed + 1,
        )
    return out


def evaluate_attribution_set(
    decompositions: Sequence[Tuple[Sequence[AttributionTuple], bool]],
    expert_per_dive: Sequence[Sequence[AttributionTuple]],
    fault_categories: Sequence[str],
    top_k_values: Sequence[int] = (1, 3),
    match: str = "strict",
) -> Dict[str, Any]:
    """The full attribution-metric pack over many dives.

    Args:
        decompositions: per-dive ``(predicted_tuples, completed_flag)``,
            in the same order as ``expert_per_dive``.
        expert_per_dive: per-dive expert tuples in the same order.
        fault_categories: the 9-category taxonomy from :mod:`attribution`.
        top_k_values: K values for top-K reporting. Both per-dive and
            union aggregations are reported per K (per-dive is the
            recommended headline number).
        match: match criterion (see :func:`attribution_set_metrics`).

    Returns:
        dict with overall set-level P/R/F1, per-fault P/R/F1, both
        top-K aggregations, and fully-explained rate.
    """
    if len(decompositions) != len(expert_per_dive):
        raise ValueError("decompositions and expert_per_dive must align")
    # Flatten across dives for set-level metrics; per-dive lookups
    # are recovered via stratification.
    all_pred: List[AttributionTuple] = []
    all_expert: List[AttributionTuple] = []
    completed: List[bool] = []
    predicted_per_dive: List[Sequence[AttributionTuple]] = []
    for (pred, done), exp in zip(decompositions, expert_per_dive):
        all_pred.extend(pred)
        all_expert.extend(exp)
        completed.append(done)
        predicted_per_dive.append(pred)

    out: Dict[str, Any] = {
        "overall": attribution_set_metrics(all_pred, all_expert, match=match),
        "per_fault": per_fault_metrics(
            all_pred, all_expert, fault_categories, match=match,
        ),
        "fully_explained_rate": fully_explained_rate(completed),
        "n_dives": len(decompositions),
        "n_predicted_total": len(all_pred),
        "n_expert_total": len(all_expert),
    }
    out["top_k_per_dive"] = {
        f"top_{k}": attribution_top_k_per_dive(
            predicted_per_dive, expert_per_dive, k=k, match=match,
        )
        for k in top_k_values
    }
    out["top_k_union"] = {
        f"top_{k}": attribution_top_k_union(
            predicted_per_dive, expert_per_dive, k=k, match=match,
        )
        for k in top_k_values
    }
    return out


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    # Types
    "AttributionTuple", "MetricWithCI",
    # AQA
    "spearman_correlation", "pearson_correlation",
    "r_l2", "r_l2_per_action",
    # Procedure segmentation
    "segment_tiou", "boundaries_from_transitions", "segment_tiou_at_threshold",
    # Physical plausibility
    "physical_foot_contact_edge", "physical_foot_contact_edge_dataset",
    "airborne_lateral_accel", "physical_foot_contact_metric",
    "physical_foot_contact",   # deprecated alias
    "EDGE_FOOT_JOINTS",
    # Counterfactual
    "faithfulness_stats", "minimality_stats", "cf_metrics_cfxai",
    # Attribution
    "attribution_set_metrics",
    "attribution_top_k_per_dive", "attribution_top_k_union",
    "attribution_top_k",         # deprecated alias
    "per_fault_metrics", "fully_explained_rate",
    # Stratification
    "stratify", "stratified_summary",
    # Bootstrap
    "bootstrap_metric", "paired_bootstrap_difference",
    # High-level
    "evaluate_aqa", "evaluate_segmentation",
    "evaluate_physical_plausibility", "evaluate_attribution_set",
]

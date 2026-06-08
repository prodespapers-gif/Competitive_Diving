"""attribution.py — FINA fault taxonomy and intervention → fault mapping.

This module encodes three things the rest of the project treats as fixed
"vocabulary":

1. The eight-group **joint lexicon** used by FineDiving-HF annotators, plus
   its mapping into SMPL joint indices (so :mod:`counterfactual` can turn
   a fault-style joint name into a concrete pose mask).
2. The nine-category **FINA fault taxonomy** from this paper's dataset
   section, plus a small interpretable rule set that maps an
   :class:`~counterfactual.Intervention` to the most likely fault.
3. A per-fault **applicability matrix** that gates each fault on the
   dive types where it can structurally occur (e.g. ``late_twist``
   requires a twist phase). Used at attribution time (to drop
   structurally-impossible predictions) and at evaluation time (so
   per-fault P/R/F1 is computed against the right denominator).

Why a rule set and not a learned classifier?
--------------------------------------------
The FineDiving-HF expert annotation codebook is itself a rule set
(annotators apply written guidelines). For attribution, the choice
between matching the codebook with rules vs. learning a classifier from
200 labelled dives is straightforward: 200 dives is too few to train a
nine-way classifier reliably, the rules generalise to unseen dive types
trivially, and reviewers can audit the mapping line-by-line. The rule
set below is exactly the one we register with the annotators during
codebook calibration (Sec. 3.3 of the paper). Each rule is also tagged
with a short name (``"R1"``, ``"R2"``, ...) so an attribution audit can
report which rule fired per record.

Applicability gating (camera-ready)
-----------------------------------
Per Nekoui & Cheng (NSAQA, 2023), per-fault evaluation should account
for the fact that not every fault applies to every dive type. The
:data:`FAULT_APPLICABILITY` matrix encodes the prerequisites; the
helper :func:`fault_applies_to_dive` evaluates them for a concrete dive.
Two consumers:

  - :func:`map_intervention_to_fault` accepts ``gate_by_dive_type``
    (default True). When True, faults that don't apply to this dive are
    replaced with :data:`FAULT_NULL` before return. The ablation toggle
    ``cfg.cf.enforce_fault_dive_gating`` controls this from the search.
  - :func:`filter_attribution_tuples_by_applicability` is the
    evaluation-side counterpart, exposed for :mod:`evaluate` to filter
    tuples before computing per-fault metrics. The toggle is
    ``cfg.eval.attribution_apply_dive_gating``.

Circular-import note
--------------------
:mod:`counterfactual` imports the :class:`Intervention` dataclass and
this module's mapping function. To avoid a cycle, the type reference to
``Intervention`` here is guarded by ``typing.TYPE_CHECKING``; at runtime
the rule set only duck-types on attribute access.

Changes from previous version
-----------------------------
- Issue 13: new :data:`FAULT_APPLICABILITY` matrix + the gating helpers
  :func:`fault_applies_to_dive`,
  :func:`filter_attribution_tuples_by_applicability`. The
  ``gate_by_dive_type`` parameter in :func:`map_intervention_to_fault`
  defaults to True so callers see gated output unless they explicitly
  opt out (e.g. the ablation row).
- New factory :func:`make_fault_mapper` builds the closure that
  :mod:`counterfactual` expects, baking in ``sub_action_types`` and the
  gating flag.
- Optional ``return_rule`` mode: ``map_intervention_to_fault(...,
  return_rule=True)`` returns ``(fault, rule_name)``. Lets the paper's
  attribution-audit table report which rule fired.
- Cleaned up the somersault knee branches (two near-identical cases
  collapsed into one with a clearer comment).
"""
from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Sequence,
    Tuple, Union, overload,
)

if TYPE_CHECKING:
    from .counterfactual import Intervention   # noqa: F401  (TYPE_CHECKING-only)

log = logging.getLogger(__name__)


# =============================================================================
# Joint group lexicon — the eight groups used by FineDiving-HF annotators
# =============================================================================
#
# Each group maps to one or more SMPL joint indices. The mapping is the
# annotator's mental model: "ankles" really means "the lower-leg tip the
# annotator can see in the video", which the diffusion edits via the
# ankle joint plus the foot joint downstream of it. Similarly "shoulders"
# includes the collar joints because the collar's rotation determines the
# resting shoulder position in SMPL.
#
# The 24-joint SMPL convention is documented in :mod:`diffusion`:
#   0 pelvis  1 l_hip   2 r_hip   3 spine1  4 l_knee  5 r_knee
#   6 spine2  7 l_ankle 8 r_ankle 9 spine3  10 l_foot 11 r_foot
#   12 neck   13 l_collar 14 r_collar 15 head
#   16 l_shoulder 17 r_shoulder 18 l_elbow 19 r_elbow
#   20 l_wrist 21 r_wrist 22 l_hand 23 r_hand
# -----------------------------------------------------------------------------


JOINT_GROUP_TO_INDICES: Dict[str, Tuple[int, ...]] = {
    "ankles":    (7, 8, 10, 11),       # ankles + feet
    "knees":     (4, 5),
    "hips":      (1, 2),
    "shoulders": (13, 14, 16, 17),     # collars + shoulders
    "elbows":    (18, 19),
    "wrists":    (20, 21, 22, 23),     # wrists + hands
    "head":      (12, 15),             # neck + head
    "torso":     (0, 3, 6, 9),         # pelvis + spine chain
}

# Reverse lookup (for occasional debugging / introspection).
SMPL_INDEX_TO_GROUP: Dict[int, str] = {
    idx: group
    for group, indices in JOINT_GROUP_TO_INDICES.items()
    for idx in indices
}


def is_valid_joint_group(group: str) -> bool:
    return group in JOINT_GROUP_TO_INDICES


# =============================================================================
# FINA fault taxonomy — nine categories from FineDiving-HF
# =============================================================================


# Listed in the order they appear in the paper's dataset section.
FAULT_TAXONOMY: Tuple[str, ...] = (
    "bent_knees",         # Pike/Straight: knees not extended; Tuck: excessive bend
    "separated_feet",     # Feet apart by >10 cm during flight or entry
    "poor_body_line",     # General alignment / shape issue (the codebook's catch-all)
    "over_rotation",      # Athlete rotates past vertical, feet enter first
    "under_rotation",     # Athlete enters past vertical with head/back first
    "late_twist",         # Twist initiated too late in the somersault
    "crooked_entry",      # Entry not perpendicular to the water surface
    "large_splash",       # Splash >7.5 cm at entry
    "unstable_entry",     # Instability immediately after submersion
)

#: Sentinel returned when a rule fires no specific mapping. The decomposition
#: keeps the record (with this label) so the search isn't silently lossy.
FAULT_NULL: str = "unattributed"


def is_valid_fault(fault: str) -> bool:
    return fault in FAULT_TAXONOMY or fault == FAULT_NULL


# =============================================================================
# Fault applicability matrix — which faults can apply to which dive types
# =============================================================================
#
# Issue 13 (camera-ready): not every FINA fault can occur on every dive.
# A non-twisting dive cannot logically have a ``late_twist`` fault; a
# back-takeoff dive without rotations cannot have ``over_rotation``.
# The applicability matrix encodes the structural prerequisites: a fault
# is applicable to a dive iff every phase listed in
# ``FAULT_APPLICABILITY[fault]`` appears in the dive's sub-action chain.
#
# The encoding choice (a set of required *phases*, not dive-type codes)
# is deliberate. FineDiving sub-action labels factor cleanly into
# {takeoff, somersault, twist, entry} via :func:`classify_sub_action`,
# and this gives us per-dive applicability without needing a per-dive-code
# lookup table that would need updating whenever a new dive code appears.
#
# The matrix matches the NSAQA convention (Nekoui & Cheng, 2023): faults
# without structural prerequisites apply to all dives; rotation faults
# require a somersault phase; the twist fault requires a twist phase.
# -----------------------------------------------------------------------------


FAULT_APPLICABILITY: Dict[str, Tuple[str, ...]] = {
    # Faults applicable to every dive (no structural prerequisite).
    # Every dive has an entry, so ``entry``-keyed faults are universal too.
    "bent_knees":      (),
    "separated_feet":  (),
    "poor_body_line":  (),
    "crooked_entry":   (),     # every dive ends with entry
    "large_splash":    (),
    "unstable_entry":  (),
    # Rotation faults require at least one somersault phase.
    "over_rotation":   ("somersault",),
    "under_rotation":  ("somersault",),
    # Twist fault requires at least one twist phase.
    "late_twist":      ("twist",),
}


def _dive_phase_set(sub_action_types: Sequence[str]) -> FrozenSet[str]:
    """Return the set of distinct phases present in ``sub_action_types``."""
    return frozenset(classify_sub_action(s) for s in sub_action_types)


def fault_applies_to_dive(
    fault: str, sub_action_types: Sequence[str],
) -> bool:
    """Decide whether ``fault`` can structurally occur in the given dive.

    A fault applies iff every phase listed in :data:`FAULT_APPLICABILITY`
    for that fault appears in the dive's sub-action chain.

    Args:
        fault: a fault name. ``FAULT_NULL`` always applies (placeholder).
            Unknown fault names return False with a debug log; we don't
            raise so an annotator typo in the expert labels doesn't crash
            the pipeline.
        sub_action_types: the dive's per-step sub-action labels (from
            :class:`~data.DiveAnnotation.sub_action_types`).

    Returns:
        True iff the fault is applicable to this dive.
    """
    if fault == FAULT_NULL:
        return True
    if fault not in FAULT_APPLICABILITY:
        log.debug("unknown fault for applicability: %r", fault)
        return False
    required = FAULT_APPLICABILITY[fault]
    if not required:
        return True
    phases = _dive_phase_set(sub_action_types)
    return all(p in phases for p in required)


def filter_attribution_tuples_by_applicability(
    tuples: Sequence[Tuple[int, Sequence[str], str, float]],
    sub_action_types: Sequence[str],
) -> List[Tuple[int, Sequence[str], str, float]]:
    """Drop attribution tuples whose fault doesn't apply to this dive.

    Used by :mod:`evaluate` to ensure per-fault metrics are computed on a
    fair denominator. Pass in the four-tuple format that
    :meth:`counterfactual.DecompositionResult.to_attribution_tuples`
    produces; tuples whose third element (the fault) fails
    :func:`fault_applies_to_dive` are removed.

    Args:
        tuples: sequence of ``(step, joints, fault, delta)`` tuples.
        sub_action_types: the dive's per-step sub-action labels.

    Returns:
        The subset of tuples that pass the applicability check, in the
        original order. Identity-preserving on the underlying objects.
    """
    return [
        t for t in tuples
        if fault_applies_to_dive(t[2], sub_action_types)
    ]


# =============================================================================
# Sub-action classification — figuring out which phase a step is in
# =============================================================================
#
# FineDiving sub-action labels combine three orthogonal facts:
#   - takeoff direction         e.g. "Forward", "Back", "Reverse", "Inward",
#                                    "Arm.Forward", "Arm.Back", "Arm.Reverse"
#   - flight content           e.g. "2.5 Soms.Pike", "1 Twists", "3 Twists"
#   - entry                    "Entry"
# The classifier below collapses these into one of {takeoff, somersault,
# twist, entry, unknown}. The body position is extracted separately for
# bent-knees disambiguation.
# -----------------------------------------------------------------------------


_TAKEOFF_LABELS: FrozenSet[str] = frozenset({
    "Forward", "Back", "Reverse", "Inward",
    "Arm.Forward", "Arm.Back", "Arm.Reverse",
    # Verbose variants that have appeared in some release pickles.
    "Armstand Forward", "Armstand Back", "Armstand Reverse",
})

_ENTRY_LABELS: FrozenSet[str] = frozenset({"Entry"})

#: Body positions that demand straight knees (per FINA rules). Tuck is
#: explicitly *not* in this set — knees should be bent in tuck position.
_TIGHT_KNEE_POSITIONS: FrozenSet[str] = frozenset({"Pike", "Straight"})


def classify_sub_action(label: str) -> str:
    """Return one of: ``takeoff``, ``somersault``, ``twist``, ``entry``, ``unknown``.

    The classification is purely string-based and is robust to whitespace
    differences in the FineDiving release pickles. ``unknown`` is returned
    rather than guessed so the attribution logic can decide what to do
    (typically: fall back to ``FAULT_NULL``).
    """
    s = label.strip()
    if s in _TAKEOFF_LABELS:
        return "takeoff"
    if s in _ENTRY_LABELS:
        return "entry"
    # Twist labels look like "1.5 Twists", "2 Twi", "0.5 Twists" etc.
    if "Twist" in s or "Twi." in s:
        return "twist"
    # Somersault labels look like "2.5 Soms.Pike", "1 Soms.Tuck", etc.
    if "Soms" in s or "Somersault" in s:
        return "somersault"
    return "unknown"


def extract_body_position(label: str) -> Optional[str]:
    """Return ``"Pike"``, ``"Tuck"``, ``"Straight"``, or ``None``.

    Used by the bent-knees rule to distinguish the position where bent
    knees are a fault (pike, straight) from the position where they're
    expected (tuck).
    """
    s = label.strip()
    if "Pike" in s:
        return "Pike"
    if "Tuck" in s:
        return "Tuck"
    if "Straight" in s:
        return "Straight"
    return None


# =============================================================================
# The rule set — intervention → fault mapping
# =============================================================================
#
# Each rule has a short name (``R<n>``) that the codebook calibration
# document references. ``map_intervention_to_fault`` can return the
# rule name alongside the fault when called with ``return_rule=True``.
# -----------------------------------------------------------------------------


@overload
def map_intervention_to_fault(
    intervention: "Intervention", sub_action_types: Sequence[str],
    *, gate_by_dive_type: bool = True, return_rule: bool = False,
) -> str: ...
@overload
def map_intervention_to_fault(
    intervention: "Intervention", sub_action_types: Sequence[str],
    *, gate_by_dive_type: bool = True, return_rule: bool = True,
) -> Tuple[str, str]: ...

def map_intervention_to_fault(
    intervention: "Intervention",
    sub_action_types: Sequence[str],
    *,
    gate_by_dive_type: bool = True,
    return_rule: bool = False,
) -> Union[str, Tuple[str, str]]:
    """Map an :class:`~counterfactual.Intervention` to a FINA fault category.

    Args:
        intervention: the search's output. Only attribute access is used
            at runtime (``modality``, ``step``, ``joint_groups``,
            ``transition_shift_frames``), so no import-time dependency
            on the :class:`Intervention` class is needed.
        sub_action_types: the dive's per-step sub-action labels, taken
            from :class:`~data.DiveAnnotation.sub_action_types`. The
            intervention's ``step`` field is 1-indexed into this tuple.
        gate_by_dive_type: when True (default), faults that don't apply
            to this dive (per :data:`FAULT_APPLICABILITY`) are replaced
            with :data:`FAULT_NULL` before return. Off means "raw rule
            output" — useful for the ablation row in the paper that
            shows the effect of gating.
        return_rule: when True, return ``(fault, rule_name)`` so the
            audit table can show which rule fired.

    Returns:
        A fault name from :data:`FAULT_TAXONOMY` if any rule applies, or
        :data:`FAULT_NULL` if no rule fires (preserves the half-point in
        the decomposition while flagging it as unattributed). When
        ``return_rule=True``, also returns the rule identifier.

    Algorithm:
        Dispatches on ``intervention.modality`` to one of three rule
        helpers below. Each helper takes the phase derived from the step's
        sub-action label and returns ``(fault, rule_name)``. Gating is
        applied last.
    """
    step_idx = intervention.step - 1
    if not 0 <= step_idx < len(sub_action_types):
        log.debug("step %d out of range for %d sub-actions",
                  intervention.step, len(sub_action_types))
        return (FAULT_NULL, "R0_step_out_of_range") if return_rule else FAULT_NULL

    sub_action = sub_action_types[step_idx]
    phase = classify_sub_action(sub_action)
    next_phase: Optional[str] = None
    if step_idx + 1 < len(sub_action_types):
        next_phase = classify_sub_action(sub_action_types[step_idx + 1])

    if intervention.modality == "joint_subset":
        fault, rule = _map_joint_subset(intervention, phase, sub_action)
    elif intervention.modality == "phase_window":
        fault, rule = _map_phase_window(intervention, phase)
    elif intervention.modality == "transition_timestamp":
        fault, rule = _map_transition_timestamp(intervention, phase, next_phase)
    else:
        log.warning("unrecognised modality: %s", intervention.modality)
        fault, rule = FAULT_NULL, "R0_unknown_modality"

    # ---- applicability gating (Issue 13) ----
    # If the rule fired a real fault that doesn't apply to this dive (e.g.
    # ``late_twist`` on a non-twist dive), substitute ``FAULT_NULL`` and
    # annotate the rule trace so the audit table reports the gating event.
    if gate_by_dive_type and fault != FAULT_NULL:
        if not fault_applies_to_dive(fault, sub_action_types):
            log.debug(
                "gated fault %r → %s (does not apply to this dive)",
                fault, FAULT_NULL,
            )
            fault, rule = FAULT_NULL, f"{rule}_gated"

    return (fault, rule) if return_rule else fault


# ---- modality-specific rules ------------------------------------------------


def _map_joint_subset(
    intervention: "Intervention", phase: str, sub_action: str,
) -> Tuple[str, str]:
    """Joint-subset rules — by phase and by which joint group is freed.

    Rule precedence is deterministic: more specific rules first (knee in a
    tight-knee position → ``bent_knees``), then less specific (any body
    group during flight → ``poor_body_line``). The codebook tie-break
    from the dataset section is implemented here: if and only if the
    intervention localises strictly on the knee joints, it's
    ``bent_knees``; otherwise the body-line catch-all fires.

    Returns:
        ``(fault, rule_name)``.
    """
    groups: FrozenSet[str] = frozenset(intervention.joint_groups)

    # --- Entry-phase rules ----------------------------------------------
    if phase == "entry":
        if "ankles" in groups and len(groups) == 1:
            # R1: strictly ankle-localised intervention at entry is
            # post-submersion instability (the legs wobble after
            # touching the water).
            return "unstable_entry", "R1_entry_ankles_unstable"
        if "torso" in groups and len(groups) == 1:
            # R2: strictly torso-localised intervention at entry is a
            # splash issue (the torso position determines the splash
            # volume more than any other joint group).
            return "large_splash", "R2_entry_torso_splash"
        if groups & {"head", "shoulders", "hips"}:
            return "crooked_entry", "R3_entry_axis_crooked"
        # Anything else at entry — generic entry fault.
        return "crooked_entry", "R3b_entry_other_crooked"

    # --- Flight (somersault) rules --------------------------------------
    if phase == "somersault":
        # Codebook tie-break (Sec. 3.3): strictly knee-localised in any
        # position is ``bent_knees`` — in pike/straight the knees should
        # be extended, in tuck they should be held tight against the
        # chest, and the codebook treats *excessive* bend as a fault in
        # all three positions.
        if groups == {"knees"}:
            return "bent_knees", "R4_soms_knees_bent"
        if groups == {"ankles"}:
            return "separated_feet", "R5_soms_ankles_separated"
        # Multi-group or non-knee single group: body-line catch-all.
        return "poor_body_line", "R6_soms_body_line"

    # --- Flight (twist) rules ------------------------------------------
    if phase == "twist":
        if groups == {"knees"}:
            return "bent_knees", "R7_twist_knees_bent"
        if groups == {"ankles"}:
            return "separated_feet", "R8_twist_ankles_separated"
        return "poor_body_line", "R9_twist_body_line"

    # --- Takeoff: nothing in the taxonomy is takeoff-specific. ---------
    if phase == "takeoff":
        return "poor_body_line", "R10_takeoff_body_line"

    return FAULT_NULL, "R0_unknown_phase"


def _map_phase_window(
    intervention: "Intervention", phase: str,
) -> Tuple[str, str]:
    """Phase-window rules — entire step is freed.

    Phase-windowed interventions don't carry joint-localisation
    information, so we map to the catch-all of that phase: entry-wide
    interventions tend to be splash-related; flight-wide interventions
    are body-line issues.
    """
    if phase == "entry":
        return "large_splash", "R11_phase_entry_splash"
    if phase in ("somersault", "twist", "takeoff"):
        return "poor_body_line", f"R12_phase_{phase}_body_line"
    return FAULT_NULL, "R0_phase_unknown"


def _map_transition_timestamp(
    intervention: "Intervention", phase: str, next_phase: Optional[str],
) -> Tuple[str, str]:
    """Transition-timestamp rules — boundary between two phases.

    Sign convention for ``transition_shift_frames`` (set in
    :func:`counterfactual.enumerate_intervention_candidates`):
        - ``shift < 0``: the counterfactual prefers an *earlier* transition,
          i.e. the original transition was *late*.
        - ``shift > 0``: the counterfactual prefers a *later* transition,
          i.e. the original transition was *early*.

    These map onto FINA's rotation faults:
        - somersault → entry boundary, original late → ``over_rotation``
          (too many rotations occurred before entry).
        - somersault → entry boundary, original early → ``under_rotation``
          (athlete entered before completing rotations).
        - any boundary involving a twist phase, original late twist
          → ``late_twist``.

    Boundaries that don't fit any of these specific patterns fall back to
    ``poor_body_line`` (anything mistimed reads as poor body shape in
    practice) or ``FAULT_NULL`` (totally unrecognised structure).
    """
    shift = int(intervention.transition_shift_frames)
    if shift == 0:
        return FAULT_NULL, "R0_transition_zero_shift"

    # --- Twist-related boundary ----------------------------------------
    # The boundary either enters or exits a twist phase. A negative shift
    # means the twist was initiated too late; the converse has no
    # FineDiving-HF category (early twist isn't a recognised fault).
    if phase == "twist" or next_phase == "twist":
        if shift < 0:
            return "late_twist", "R13_transition_twist_late"
        return "poor_body_line", "R14_transition_twist_early"  # codebook fallback

    # --- Somersault → entry boundary -----------------------------------
    if phase == "somersault" and next_phase == "entry":
        if shift < 0:
            return "over_rotation", "R15_transition_over_rotation"
        return "under_rotation", "R16_transition_under_rotation"

    # --- Any other boundary --------------------------------------------
    return "poor_body_line", "R17_transition_other"


# =============================================================================
# Factory — build the Callable[[Intervention], str] closure
# =============================================================================


def make_fault_mapper(
    sub_action_types: Sequence[str],
    *,
    gate_by_dive_type: bool = True,
) -> Callable[["Intervention"], str]:
    """Return a closure that :mod:`counterfactual` can pass as ``fault_mapper``.

    :func:`counterfactual.decompose` expects ``fault_mapper: (Intervention) → str``
    that knows nothing about the dive's sub-action chain. This factory
    bakes ``sub_action_types`` and the gating flag into the closure so
    callers don't have to thread the chain through `decompose` themselves.

    Args:
        sub_action_types: the dive's per-step sub-action labels.
        gate_by_dive_type: forwarded to :func:`map_intervention_to_fault`.
            Set to ``cfg.cf.enforce_fault_dive_gating`` at the call site.

    Returns:
        A picklable function ``(Intervention) → str``.
    """
    sub_action_tuple = tuple(sub_action_types)   # immutable closure capture

    def _mapper(intervention: "Intervention") -> str:
        return map_intervention_to_fault(
            intervention, sub_action_tuple,
            gate_by_dive_type=gate_by_dive_type,
        )

    return _mapper


# =============================================================================
# Inverse mapping — typical intervention for each fault (figures only)
# =============================================================================


def fault_to_typical_intervention(fault: str) -> Dict[str, Any]:
    """Return a *typical* intervention specification for a given fault.

    This is used only for qualitative figures and for a round-trip
    consistency check (``fault → typical intervention → fault`` should
    return the same fault). It does **not** produce a complete
    :class:`Intervention` — frame ranges depend on the specific dive's
    transitions, which the caller fills in.

    Returns a dict with at least ``modality``; for joint-subset faults
    it also carries ``joint_groups`` and a target ``phase``; for
    transition faults it carries ``shift_sign`` and the ``boundary``
    type (``"somersault_to_entry"`` or ``"twist"``).
    """
    table: Dict[str, Dict[str, Any]] = {
        "bent_knees": {
            "modality": "joint_subset",
            "joint_groups": ("knees",),
            "phase": "somersault",
        },
        "separated_feet": {
            "modality": "joint_subset",
            "joint_groups": ("ankles",),
            "phase": "somersault",
        },
        "poor_body_line": {
            "modality": "phase_window",
            "phase": "somersault",
        },
        "over_rotation": {
            "modality": "transition_timestamp",
            "shift_sign": -1,
            "boundary": "somersault_to_entry",
        },
        "under_rotation": {
            "modality": "transition_timestamp",
            "shift_sign": +1,
            "boundary": "somersault_to_entry",
        },
        "late_twist": {
            "modality": "transition_timestamp",
            "shift_sign": -1,
            "boundary": "twist",
        },
        "crooked_entry": {
            "modality": "joint_subset",
            "joint_groups": ("head", "shoulders"),
            "phase": "entry",
        },
        "large_splash": {
            "modality": "phase_window",
            "phase": "entry",
        },
        "unstable_entry": {
            "modality": "joint_subset",
            "joint_groups": ("ankles",),
            "phase": "entry",
        },
    }
    if fault not in table:
        raise KeyError(f"unknown fault: {fault!r}")
    return table[fault]


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    # Lexicons
    "JOINT_GROUP_TO_INDICES", "SMPL_INDEX_TO_GROUP",
    "FAULT_TAXONOMY", "FAULT_NULL",
    "FAULT_APPLICABILITY",
    # Sub-action classification
    "classify_sub_action", "extract_body_position",
    # Mapping
    "map_intervention_to_fault",
    "make_fault_mapper",
    "fault_to_typical_intervention",
    # Applicability gating (Issue 13)
    "fault_applies_to_dive",
    "filter_attribution_tuples_by_applicability",
    # Validation
    "is_valid_joint_group", "is_valid_fault",
]

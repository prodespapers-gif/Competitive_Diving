"""data.py — Datasets, exemplar selection, fault annotations, DataModule.

Implements
----------
- :func:`load_finediving_annotations`: auto-detects the released FineDiving
  pickle format (tuple-indexed entries) and converts to typed
  :class:`DiveAnnotation` records. Also handles the dict-with-named-fields
  format used by FineDiving+ and curated re-releases.
- :class:`FineDivingDataset`: pairs (query, exemplar) for AQA training and
  inference. Returns a typed :class:`AQABatch` whose field order matches
  the original ``FineDiving_Pair.py`` tuple, so existing AQA baselines
  drop in via ``next(iter(loader)).as_legacy_tuple()``.
- :class:`PoseDataset`: loads pre-extracted 3D SMPL pose sequences from
  ``data/poses/*.npz``. Poses are pre-extracted offline because PHALP's
  build chain is incompatible with our deployment env (CentOS 7,
  GCC 4.8.5); see README for the offline pipeline.
- :class:`FaultAnnotations`: loads FineDiving-HF, the half-point fault
  annotation layer introduced in this paper (200 dives, stratified across
  score quintiles, expert-labelled with κ = 0.71).
- :class:`ExemplarSelector`: action-type-conditioned with configurable
  difficulty-window or random fallback for cross-height transfer where
  some action types are absent from the train split.
- :class:`DataModule`: factory wiring train / val / test splits with
  deterministic, DDP-aware samplers and stratified-by-action-type val
  holdout.

Verified against the reference repository
-----------------------------------------
The FineDiving release at https://github.com/xujinglin/FineDiving stores
annotations as a Python pickle whose entries are positional tuples:
    data_anno[(folder, dive_id)] = [
        action_type / dive_number (str, e.g. "5152B"),  # [0]
        final_score (float),                            # [1]
        difficulty (float),                             # [2]
        <position 3 — varies; we treat it as optional>, # [3]
        per-frame sub-action labels (list[int|str]),    # [4]
    ]
Step transitions are derived at runtime from the per-frame label sequence
in :func:`_derive_transitions_from_labels` — wherever the label changes,
a new step starts. This mirrors ``FineDiving_Pair.py:load_video``
lines 70-78 of the reference implementation.

Frame filenames are 5-digit zero-padded (``00001.jpg``) per
``data_preparation/data_process.py`` of the reference. Configurable via
``DataCfg.frame_filename_pattern``.

The reference image transform is ``Resize((200,112)) → 112×112 crop`` —
NOT the more familiar 224×224 ImageNet recipe. See ``DataCfg.image_size``
and ``DataCfg.image_resize_hw`` for the matching defaults.

Design notes
------------
- The fault/joint string lexicon is *not* validated here. Validation is
  ``attribution.py``'s job; ``data.py`` treats those strings as opaque
  to keep the import graph one-way.
- The held-out validation set is a deterministic 10% of the FineDiving
  train split, seeded by ``DataCfg.val_split_seed`` and stratified by
  action_type so rare dive types are not entirely absent from val.
- Multi-exemplar voting (M=10) at inference is handled by repeated
  sampling at the DataModule level, not by stacking exemplars per item.
"""
from __future__ import annotations

import json
import logging
import pickle
import random
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, FrozenSet, Iterable, Iterator, List,
    Mapping, NamedTuple, Optional, Sequence, Tuple, Union
)

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import (
    DataLoader, Dataset, DistributedSampler, Sampler, Subset
)
from torchvision import transforms

from .configs import Cfg, DataCfg

log = logging.getLogger(__name__)

# Canonical dive identifier. The join key between AQA annotations, pose
# .npz files, and FineDiving-HF fault annotations. Format:
#   "<competition>_<event>_<round>_<athlete>"  (string, never parsed downstream)
DiveID = str


# =============================================================================
# Schema
# =============================================================================


@dataclass(frozen=True)
class DiveAnnotation:
    """One dive's worth of AQA annotation, normalised across datasets.

    Produced by :func:`load_finediving_annotations` (and its FineDiving+ /
    MTL-AQA siblings). Immutable so it can be shared across DDP workers.
    """

    dive_id: DiveID
    video_id: str                       # the (folder, dive_idx) join key flattened
    competition: str
    # Trimmed-clip frame indices in the source video — used by frame-loader
    # to pick the first/last on-disk JPEG. May be ``(0, clip_num_frames-1)``
    # when the FineDiving pickle does not pre-trim (which is the common case).
    start_frame: int
    end_frame: int
    clip_num_frames: int
    # Dive metadata
    action_type: str                    # e.g. "5152B"
    sub_action_types: Tuple[str, ...]   # length == num_steps
    difficulty: float
    # Step-level boundaries: frame indices (in trimmed-clip coordinates) of
    # the L step transitions. Length == num_step_transitions == num_steps - 1.
    # Sorted ascending; each is in [0, clip_num_frames).
    step_transitions: Tuple[int, ...]
    # Scores
    judges_scores: Tuple[float, ...]    # empty if the source pickle didn't carry them
    final_score: float                  # DD × sum-of-middle-five (judges)
    # Discipline (FineDiving+ adds these; FineDiving leaves them None)
    height: Optional[str] = None        # "10m" | "3m"
    is_synchronised: bool = False

    def __post_init__(self) -> None:
        if self.start_frame > self.end_frame:
            raise ValueError(f"{self.dive_id}: start>end")
        if self.clip_num_frames <= 0:
            raise ValueError(f"{self.dive_id}: clip_num_frames must be > 0")
        if len(self.sub_action_types) - 1 != len(self.step_transitions):
            raise ValueError(
                f"{self.dive_id}: {len(self.sub_action_types)} sub-actions imply "
                f"{len(self.sub_action_types) - 1} transitions, "
                f"got {len(self.step_transitions)}"
            )
        # Transitions must be sorted and bounded — protects the model code
        # downstream which assumes ordering (Eq. 3 of FineDiving paper).
        prev = -1
        for t in self.step_transitions:
            if t < 0 or t >= self.clip_num_frames:
                raise ValueError(
                    f"{self.dive_id}: transition {t} out of range "
                    f"[0, {self.clip_num_frames})"
                )
            if t < prev:
                raise ValueError(
                    f"{self.dive_id}: step_transitions {self.step_transitions} "
                    f"are not monotonically increasing"
                )
            prev = t
        if not 1.0 <= self.difficulty <= 5.0:
            raise ValueError(
                f"{self.dive_id}: difficulty {self.difficulty} out of range [1, 5]"
            )
        if self.judges_scores and any(
            s < 0 or s > 10.0 for s in self.judges_scores
        ):
            raise ValueError(f"{self.dive_id}: judge score outside [0, 10]")

    def median_judge_score(self) -> float:
        """Median of the per-judge execution scores; 0.0 if unavailable."""
        if not self.judges_scores:
            return 0.0
        return float(np.median(self.judges_scores))

    def __repr__(self) -> str:  # compact, for logs
        h = self.height or "?"
        return (
            f"DiveAnnotation({self.dive_id} | {self.action_type} | "
            f"DD={self.difficulty:.1f} | score={self.final_score:.2f} | "
            f"{h} | T={len(self.step_transitions)})"
        )


@dataclass(frozen=True)
class FaultRecord:
    """One atomic half-point deduction record.

    Each record encodes exactly 0.5 points off a perfect ten; deductions
    larger than 0.5 are expanded into multiple records sharing the same
    (k, J, f). This is the natural unit of counterfactual attribution
    since 0.5 is the atomic award in FINA execution scoring.
    """

    step: int                       # k ∈ {1, ..., num_steps}
    joints: FrozenSet[str]          # subset of the joint-group lexicon
    fault: str                      # FINA fault category (validated elsewhere)
    delta: float = 0.5              # always 0.5 in HF

    def __post_init__(self) -> None:
        if not 0.4 < self.delta < 0.6:
            raise ValueError(f"FaultRecord.delta must be 0.5, got {self.delta}")
        if len(self.joints) == 0:
            raise ValueError("FaultRecord.joints must be non-empty")
        if self.step < 1:
            raise ValueError("FaultRecord.step is 1-indexed; got < 1")


@dataclass(frozen=True)
class DiveFaults:
    """All half-point records for one dive."""

    dive_id: DiveID
    action_type: str
    judges_scores: Tuple[float, ...]
    median_judge_score: float
    deduction_total: float           # 10 - median_judge_score
    records: Tuple[FaultRecord, ...] # ordered, sum(record.delta) == deduction_total

    def __post_init__(self) -> None:
        s = sum(r.delta for r in self.records)
        if abs(s - self.deduction_total) > 1e-6:
            raise ValueError(
                f"{self.dive_id}: records sum to {s:.3f}, "
                f"expected deduction_total={self.deduction_total:.3f}"
            )


# =============================================================================
# Typed batch — also acts as the legacy-tuple compatibility shim
# =============================================================================


class AQABatch(NamedTuple):
    """The dataset's per-item payload.

    NamedTuple gives us free tuple-unpacking for compatibility with the
    original ``FineDiving_Pair.py`` field order, plus attribute access for
    readable downstream code. ``as_legacy_tuple()`` reproduces the exact
    ordering used by existing AQA baselines.
    """

    # Frames: (C, T, H, W) per sample; stacked over batch by the collate fn.
    query_frames: torch.Tensor
    exemplar_frames: torch.Tensor
    # Scalars (kept as tensors for clean batching)
    query_score: torch.Tensor              # ()
    exemplar_score: torch.Tensor           # ()
    query_difficulty: torch.Tensor         # ()
    exemplar_difficulty: torch.Tensor      # ()
    # Step-level supervision for the segmentation block (Eq. 2-3)
    query_step_transitions: torch.Tensor   # (L,) int64
    exemplar_step_transitions: torch.Tensor
    # Action / sub-action labels (strings; left as strings in batches)
    query_action_type: str
    exemplar_action_type: str
    query_sub_action_types: Tuple[str, ...]
    exemplar_sub_action_types: Tuple[str, ...]
    # Individual judge scores (used by the half-point CF outer loop;
    # may be a length-0 tensor if the source pickle didn't carry them)
    query_judges_scores: torch.Tensor      # (num_judges,) or (0,)
    exemplar_judges_scores: torch.Tensor
    # Identifiers (used for joining poses and faults)
    query_dive_id: DiveID
    exemplar_dive_id: DiveID

    def as_legacy_tuple(self):
        """Return the tuple ordering used by ``FineDiving_Pair.py``."""
        return (
            self.query_frames, self.exemplar_frames,
            self.query_score, self.exemplar_score,
            self.query_difficulty, self.exemplar_difficulty,
            self.query_step_transitions, self.exemplar_step_transitions,
        )


# =============================================================================
# Annotation loaders
# =============================================================================


def _derive_transitions_from_labels(
    frame_labels: Sequence[Any],
) -> Tuple[Tuple[Any, ...], Tuple[int, ...]]:
    """Walk per-frame sub-action labels and return (sub_action_types, transitions).

    Mirrors the reference implementation in ``FineDiving_Pair.py:load_video``
    lines 75-78:
        frames_catogeries = list(set(frames_labels))
        frames_catogeries.sort(key=frames_labels.index)
        transitions = [frames_labels.index(c) for c in frames_catogeries]

    Returns:
        sub_action_types: ordered tuple of unique labels in the order they
            first appear; length == num_steps.
        step_transitions: starting frame index of each step EXCEPT the
            first (since step 1 always starts at frame 0). Length ==
            num_steps - 1 == num_step_transitions.

    Reference helper.py:84-86 takes
        transitions = [transitions[1]-1, transitions[-1]-1]
    i.e. it offsets by -1 and skips the first transition (= 0 by construction).
    We replicate that here.
    """
    if len(frame_labels) == 0:
        raise ValueError("frame_labels is empty; cannot derive transitions")
    # Preserve order of first appearance (handles repeated label patterns).
    seen: List[Any] = []
    seen_set: set = set()
    for lbl in frame_labels:
        if lbl not in seen_set:
            seen.append(lbl)
            seen_set.add(lbl)
    sub_action_types = tuple(seen)
    # Start-frame of each unique label in order.
    starts: List[int] = []
    for c in sub_action_types:
        for i, lbl in enumerate(frame_labels):
            if lbl == c:
                starts.append(i)
                break
    # Reference convention: drop the first (= 0) and subtract 1. The
    # reference's BCE label `label_12_pad[bs, transition_idx, k] = 1` is
    # set at frame indices ``starts[1]-1`` and ``starts[-1]-1``.
    transitions = tuple(s - 1 for s in starts[1:])
    return sub_action_types, transitions


def _adapt_finediving_tuple_entry(
    key: Any, value: Sequence[Any],
    num_step_transitions: int,
) -> Optional[DiveAnnotation]:
    """Convert one reference-format pickle entry to a DiveAnnotation.

    The reference schema for ``fine-grained_annotation_aqa.pkl``:
        key   = (folder_str, dive_id_int)
        value = [action_type, final_score, difficulty, <something>, frame_labels]
    Position [3] varies across releases; we tolerate any type there and
    only consume positions [0], [1], [2], and [4]. If the entry is
    malformed (too short, frame_labels empty), returns None and logs a
    warning — the caller filters None out.
    """
    if not isinstance(value, (list, tuple)) or len(value) < 5:
        log.warning(
            "FineDiving entry %r is not a length-5+ sequence; skipping.", key
        )
        return None
    if isinstance(key, tuple) and len(key) >= 2:
        folder, dive_idx = key[0], key[1]
        dive_id = f"{folder}_{dive_idx}"
        video_id = f"{folder}/{dive_idx}"
        competition = str(folder)
    else:
        dive_id = str(key)
        video_id = str(key)
        competition = "unknown"

    try:
        action_type = str(value[0])
        final_score = float(value[1])
        difficulty = float(value[2])
        frame_labels = list(value[4])
    except (TypeError, ValueError, IndexError) as exc:
        log.warning(
            "FineDiving entry %r failed to parse (positions 0/1/2/4): %s; skipping.",
            key, exc,
        )
        return None

    if not frame_labels:
        log.warning("FineDiving entry %r has empty frame_labels; skipping.", key)
        return None

    sub_action_types, transitions = _derive_transitions_from_labels(frame_labels)
    if len(transitions) != num_step_transitions:
        log.warning(
            "%s: derived %d transitions (expected %d); skipping.",
            dive_id, len(transitions), num_step_transitions,
        )
        return None

    clip_num_frames = len(frame_labels)
    # Stringify labels for downstream consistency (sub-action codes are
    # ints in the reference pickle but strings in our type system).
    sub_action_types_str = tuple(str(s) for s in sub_action_types)

    return DiveAnnotation(
        dive_id=dive_id,
        video_id=video_id,
        competition=competition,
        start_frame=0,
        end_frame=clip_num_frames - 1,
        clip_num_frames=clip_num_frames,
        action_type=action_type,
        sub_action_types=sub_action_types_str,
        difficulty=difficulty,
        step_transitions=transitions,
        judges_scores=tuple(),       # not in the FineDiving pickle
        final_score=final_score,
    )


def _adapt_finediving_dict_entry(
    key: Any, entry: Mapping[str, Any],
    num_step_transitions: int, num_judges: int, require_judges: bool,
) -> Optional[DiveAnnotation]:
    """Convert one dict-format annotation entry to a DiveAnnotation.

    Used for FineDiving+ and any future curated dict-based releases. The
    expected fields are documented in the body; missing fields raise
    KeyError (caller decides whether to skip).
    """
    if isinstance(key, tuple):
        dive_id = "_".join(str(k) for k in key)
        competition = str(key[0])
    else:
        dive_id = str(key)
        competition = str(entry.get("competition", "unknown"))

    try:
        action_type = str(entry["action_type"])
        difficulty = float(entry["difficulty"])
        final_score = float(entry["final_score"])
        sub_action_types = tuple(str(s) for s in entry["sub_action_types"])
        step_transitions = tuple(int(t) for t in entry["step_transitions"])
        start_frame = int(entry.get("start_frame", 0))
        end_frame = int(entry.get(
            "end_frame", start_frame + int(entry.get("num_frames", 1)) - 1
        ))
        clip_num_frames = int(entry.get(
            "num_frames", end_frame - start_frame + 1
        ))
        judges_scores = tuple(
            float(s) for s in entry.get("judges_scores", ())
        )
    except KeyError as exc:
        log.warning(
            "%s: missing required field %s; skipping.", dive_id, exc,
        )
        return None
    except (TypeError, ValueError) as exc:
        log.warning("%s: failed to parse — %s; skipping.", dive_id, exc)
        return None

    if len(step_transitions) != num_step_transitions:
        log.warning(
            "%s: expected %d step transitions, got %d; skipping.",
            dive_id, num_step_transitions, len(step_transitions),
        )
        return None
    if require_judges and len(judges_scores) != num_judges:
        log.warning(
            "%s: expected %d judges, got %d; skipping.",
            dive_id, num_judges, len(judges_scores),
        )
        return None

    return DiveAnnotation(
        dive_id=dive_id,
        video_id=str(entry.get("video_id", dive_id)),
        competition=competition,
        start_frame=start_frame,
        end_frame=end_frame,
        clip_num_frames=clip_num_frames,
        action_type=action_type,
        sub_action_types=sub_action_types,
        difficulty=difficulty,
        step_transitions=step_transitions,
        judges_scores=judges_scores,
        final_score=final_score,
        height=entry.get("height"),
        is_synchronised=bool(entry.get("is_synchronised", False)),
    )


def _detect_format(raw: Mapping[Any, Any]) -> str:
    """Return ``"tuple"`` for the reference FineDiving format, ``"dict"`` otherwise.

    Decision rule: peek at the first entry — if it's a list/tuple of length
    ≥ 5, treat the whole pickle as tuple-format. If it's a Mapping (dict),
    treat as dict-format. Mixed pickles will be flagged with a warning
    and routed by per-entry type-checking inside the loader.
    """
    if not raw:
        raise ValueError("empty annotation pickle")
    first_key = next(iter(raw))
    first_val = raw[first_key]
    if isinstance(first_val, Mapping):
        return "dict"
    if isinstance(first_val, (list, tuple)):
        return "tuple"
    raise ValueError(
        f"unrecognised annotation value type: {type(first_val).__name__}"
    )


def load_finediving_annotations(
    pkl_path: Path,
    num_step_transitions: int,
    num_judges: int = 7,
    require_judges_scores: bool = False,
    split_list_path: Optional[Path] = None,
) -> List[DiveAnnotation]:
    """Load FineDiving / FineDiving+ annotations from a pickle file.

    Auto-detects the released FineDiving format (tuple-indexed entries)
    vs the dict-with-named-fields format used by FineDiving+ and curated
    releases. Both yield typed :class:`DiveAnnotation` records.

    Args:
        pkl_path: path to the annotation pickle. For the reference
            FineDiving release this is
            ``Annotations/fine-grained_annotation_aqa.pkl``.
        num_step_transitions: expected number of step transitions per
            dive (= L in the paper; 2 by default).
        num_judges: expected number of individual judge scores; only
            enforced when ``require_judges_scores=True``.
        require_judges_scores: if True, drop entries with the wrong
            number of judge scores. The reference FineDiving pickle
            does NOT carry judges' scores, so this defaults to False.
        split_list_path: optional path to a pickle containing the list
            of dive keys to keep (``train_split.pkl`` /
            ``test_split.pkl`` in the reference release).

    Returns:
        A list of :class:`DiveAnnotation`. Entries that fail to parse
        are skipped with a warning; loading raises only if zero entries
        survive.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"FineDiving annotation pickle not found: {pkl_path}")
    with pkl_path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"{pkl_path}: top-level object is {type(raw).__name__}, expected Mapping"
        )

    # Optional split filter (reference release ships train_split.pkl as a
    # list of keys).
    keep_keys: Optional[set] = None
    if split_list_path is not None:
        with Path(split_list_path).open("rb") as f:
            split_keys = pickle.load(f)
        keep_keys = set(_freeze(k) for k in split_keys)

    fmt = _detect_format(raw)
    log.info("Detected FineDiving annotation format: %s", fmt)

    annotations: List[DiveAnnotation] = []
    for key, entry in raw.items():
        if keep_keys is not None and _freeze(key) not in keep_keys:
            continue
        # Per-entry routing tolerates the rare mixed pickle.
        if isinstance(entry, Mapping):
            ann = _adapt_finediving_dict_entry(
                key, entry,
                num_step_transitions=num_step_transitions,
                num_judges=num_judges,
                require_judges=require_judges_scores,
            )
        elif isinstance(entry, (list, tuple)):
            ann = _adapt_finediving_tuple_entry(
                key, entry, num_step_transitions=num_step_transitions,
            )
        else:
            log.warning(
                "FineDiving entry %r has unsupported value type %s; skipping.",
                key, type(entry).__name__,
            )
            continue
        if ann is not None:
            annotations.append(ann)

    if not annotations:
        raise RuntimeError(
            f"No annotations loaded from {pkl_path} "
            f"(format detected: {fmt}; check the pickle structure)."
        )
    log.info("Loaded %d FineDiving annotations from %s", len(annotations), pkl_path)
    return annotations


def _freeze(key: Any) -> Any:
    """Convert nested lists/tuples in a key to a hashable form for set ops."""
    if isinstance(key, list):
        return tuple(_freeze(k) for k in key)
    if isinstance(key, tuple):
        return tuple(_freeze(k) for k in key)
    return key


def load_fault_annotations(json_path: Path) -> Dict[DiveID, DiveFaults]:
    """Load the FineDiving-HF half-point fault annotations.

    Schema (JSON):
        {
          "version": "1.0",
          "taxonomy": [<9 FINA fault categories>],
          "joint_lexicon": [<8 joint groups>],
          "annotations": {
              "<dive_id>": {
                  "action_type": str,
                  "judges_scores": [float, ...],
                  "median_judge_score": float,
                  "deduction_total": float,
                  "records": [
                      {"step": int, "joints": [str, ...],
                       "fault": str, "delta": 0.5},
                      ...
                  ]
              },
              ...
          }
        }
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"FineDiving-HF annotations not found: {json_path}")
    with json_path.open("r") as f:
        data = json.load(f)

    if "annotations" not in data:
        raise ValueError(f"{json_path}: missing top-level 'annotations'")

    faults: Dict[DiveID, DiveFaults] = {}
    for dive_id, entry in data["annotations"].items():
        records = tuple(
            FaultRecord(
                step=int(r["step"]),
                joints=frozenset(str(j) for j in r["joints"]),
                fault=str(r["fault"]),
                delta=float(r.get("delta", 0.5)),
            )
            for r in entry["records"]
        )
        faults[dive_id] = DiveFaults(
            dive_id=dive_id,
            action_type=str(entry["action_type"]),
            judges_scores=tuple(float(s) for s in entry["judges_scores"]),
            median_judge_score=float(entry["median_judge_score"]),
            deduction_total=float(entry["deduction_total"]),
            records=records,
        )

    log.info("Loaded %d FineDiving-HF fault annotations from %s",
             len(faults), json_path)
    return faults


def load_failed_pose_ids(path: Path) -> FrozenSet[DiveID]:
    """Load dive ids whose pose extraction failed (water-spray occlusion)."""
    path = Path(path)
    if not path.exists():
        return frozenset()
    with path.open("r") as f:
        ids = {line.strip() for line in f if line.strip() and not line.startswith("#")}
    return frozenset(ids)


# =============================================================================
# Frame loading
# =============================================================================


def build_frame_transform(cfg: DataCfg, train: bool) -> Callable:
    """Build the per-frame transform matching the reference recipe.

    Reference (FineDiving-main/tools/builder.py):
        train: Resize((200, 112)) → RandomHorizontalFlip → RandomCrop(112)
               → ToTensor → Normalize(ImageNet stats)
        eval:  Resize((200, 112)) → CenterCrop(112)
               → ToTensor → Normalize(ImageNet stats)

    The Resize uses an explicit ``(height, width)`` tuple — torchvision
    interprets a tuple as (H, W), not preserving aspect ratio. This is
    deliberate: diving frames are widescreen and the AQA recipe wants a
    deterministic 200×112 input regardless of source resolution.
    """
    norm = transforms.Normalize(mean=cfg.image_mean, std=cfg.image_std)
    resize = transforms.Resize(cfg.image_resize_hw)   # (H, W)
    if train:
        return transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(cfg.image_size),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        resize,
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        norm,
    ])


def _sample_frame_indices(
    start: int, end: int, num_frames: int, train: bool,
    jitter: int = 0,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Sample ``num_frames`` indices from ``[start, end]``.

    Mirrors the reference's deterministic ``np.linspace`` sampling
    (FineDiving_Pair.py:load_video line 71). Set ``jitter > 0`` to add
    a per-index uniform jitter at training time — useful for ablation,
    but disabled by default to match the published recipe.
    """
    span = end - start + 1
    if span <= 0:
        raise ValueError(f"empty clip: [{start}, {end}]")
    if span <= num_frames:
        # Pad by repeating the last frame.
        return list(range(start, end + 1)) + [end] * (num_frames - span)
    base = np.linspace(start, end, num_frames, dtype=np.int64)
    if train and jitter > 0:
        r = rng or random
        jit = np.array(
            [r.randint(-jitter, jitter) for _ in range(num_frames)],
            dtype=np.int64,
        )
        idx = np.clip(base + jit, start, end)
        return idx.tolist()
    return base.tolist()


def _load_clip(
    frames_dir: Path, indices: Sequence[int],
    transform: Callable, filename_pattern: str,
) -> torch.Tensor:
    """Load and transform a list of frames into a (C, T, H, W) tensor.

    Frame filenames follow the FineDiving convention (5-digit zero-padded
    by default). Missing frames raise ``FileNotFoundError`` with the full
    path so the operator can spot extraction gaps quickly.
    """
    frames = []
    for i in indices:
        path = frames_dir / filename_pattern.format(i)
        if not path.exists():
            raise FileNotFoundError(f"missing frame: {path}")
        with Image.open(path) as img:
            frames.append(transform(img.convert("RGB")))
    # (T, C, H, W) -> (C, T, H, W) to match I3D's expected layout.
    return torch.stack(frames, dim=1)


# =============================================================================
# Exemplar selection
# =============================================================================


class ExemplarSelector:
    """Selects exemplars under a chosen strategy, with deterministic seeding
    and a graceful fallback for cross-distribution transfer.

    The selector is constructed from the *training* pool (the only legal
    source of exemplars), and is queried by dive_id for both training and
    inference. At training time, one exemplar is drawn per query; at
    inference time, :meth:`sample_voting_set` draws M exemplars for the
    multi-exemplar voting strategy of Yu et al. (CoRe).

    When the primary strategy returns an empty pool — the canonical
    failure mode in cross-height transfer where some action types only
    appear on platform — we fall back to a configurable secondary
    strategy (``"difficulty"`` window or ``"random"``). The fallback
    activations are logged so coverage can be reported alongside numbers.
    """

    def __init__(
        self,
        train_pool: Sequence[DiveAnnotation],
        strategy: str,
        difficulty_window: float = 0.2,
        fallback: str = "difficulty",
        seed: int = 0,
    ) -> None:
        if strategy not in {"action_type", "difficulty", "random"}:
            raise ValueError(f"unknown strategy: {strategy}")
        if fallback not in {"difficulty", "random", "raise"}:
            raise ValueError(f"unknown fallback: {fallback}")
        self.strategy = strategy
        self.difficulty_window = difficulty_window
        self.fallback = fallback
        self._seed = seed
        self._pool: List[DiveAnnotation] = list(train_pool)
        self._by_id: Dict[DiveID, DiveAnnotation] = {
            a.dive_id: a for a in self._pool
        }
        self._by_action: Dict[str, List[DiveAnnotation]] = defaultdict(list)
        for a in self._pool:
            self._by_action[a.action_type].append(a)
        # Coverage tracking — reportable alongside cross-dataset numbers.
        self._fallback_hits: Counter = Counter()
        self._strategy_hits: int = 0

    # ----- queries -----------------------------------------------------

    def candidates_for(self, query: DiveAnnotation) -> List[DiveAnnotation]:
        """All legal exemplars for ``query`` under the current strategy.

        If the primary pool is empty and a fallback is configured, the
        fallback pool is returned instead and the event is logged for
        coverage reporting. If ``fallback="raise"``, raises RuntimeError.
        """
        primary = self._primary_pool(query)
        # Never use the query itself as its own exemplar.
        primary = [a for a in primary if a.dive_id != query.dive_id]
        if primary:
            self._strategy_hits += 1
            return primary
        # Empty primary pool — apply fallback.
        if self.fallback == "raise":
            raise RuntimeError(
                f"No exemplar candidates for {query.dive_id} under "
                f"strategy={self.strategy} and fallback=raise."
            )
        secondary = self._fallback_pool(query)
        secondary = [a for a in secondary if a.dive_id != query.dive_id]
        if not secondary:
            # Hard failure: nothing in the train pool matches.
            raise RuntimeError(
                f"No exemplar candidates for {query.dive_id} under "
                f"strategy={self.strategy}, fallback={self.fallback}. "
                f"Check the training pool coverage."
            )
        self._fallback_hits[query.action_type] += 1
        log.debug(
            "ExemplarSelector: falling back to %s for query %s (action_type %s)",
            self.fallback, query.dive_id, query.action_type,
        )
        return secondary

    def _primary_pool(self, query: DiveAnnotation) -> List[DiveAnnotation]:
        if self.strategy == "action_type":
            return list(self._by_action.get(query.action_type, []))
        if self.strategy == "difficulty":
            return [
                a for a in self._pool
                if abs(a.difficulty - query.difficulty) <= self.difficulty_window
            ]
        return list(self._pool)   # random

    def _fallback_pool(self, query: DiveAnnotation) -> List[DiveAnnotation]:
        if self.fallback == "difficulty":
            return [
                a for a in self._pool
                if abs(a.difficulty - query.difficulty) <= self.difficulty_window
            ]
        return list(self._pool)   # random

    def sample_one(self, query: DiveAnnotation, epoch: int) -> DiveAnnotation:
        """Sample one exemplar for training (epoch-seeded determinism)."""
        candidates = self.candidates_for(query)
        # Per-query, per-epoch seed: same (query, epoch) ⇒ same exemplar.
        key = (self._seed, epoch, query.dive_id)
        rng = random.Random(hash(key) & 0xFFFFFFFF)
        return rng.choice(candidates)

    def sample_voting_set(
        self, query: DiveAnnotation, M: int
    ) -> List[DiveAnnotation]:
        """Sample M exemplars for inference voting (CoRe strategy).

        Sampling is *with* replacement if the candidate pool has fewer
        than M entries — multi-exemplar voting variance is then
        artificially low and we log a warning so the issue is visible.
        """
        candidates = self.candidates_for(query)
        rng = random.Random(hash((self._seed, "vote", query.dive_id)) & 0xFFFFFFFF)
        if len(candidates) >= M:
            return rng.sample(candidates, M)
        log.warning(
            "Only %d candidates for %s under strategy=%s+fallback=%s; "
            "voting set will repeat (M=%d).",
            len(candidates), query.dive_id, self.strategy, self.fallback, M,
        )
        return [rng.choice(candidates) for _ in range(M)]

    # ----- coverage ----------------------------------------------------

    def coverage_report(self) -> Dict[str, Any]:
        """Return a dict suitable for logging alongside cross-dataset numbers.

        Reports how often the primary strategy succeeded vs how often it
        had to fall back, broken down by query action_type.
        """
        return {
            "strategy": self.strategy,
            "fallback": self.fallback,
            "primary_hits": int(self._strategy_hits),
            "fallback_hits_total": sum(self._fallback_hits.values()),
            "fallback_hits_by_action_type": dict(self._fallback_hits),
        }


# =============================================================================
# Datasets
# =============================================================================


class FineDivingDataset(Dataset):
    """Paired (query, exemplar) AQA dataset.

    Mirrors the field layout of the original ``FineDiving_Pair.py`` so
    existing AQA baselines (TSA-Net, CoRe, MUSDL) drop in via
    ``AQABatch.as_legacy_tuple()``. At training time the exemplar is
    drawn from the training pool; at evaluation time the dataset itself
    returns *one* exemplar — the DataModule re-samples M times for
    multi-exemplar voting (rather than baking M-stacking into __getitem__,
    which would explode memory for no functional gain).
    """

    def __init__(
        self,
        annotations: Sequence[DiveAnnotation],
        train_pool: Sequence[DiveAnnotation],
        cfg: DataCfg,
        train: bool,
        epoch_ref: Optional[Callable[[], int]] = None,
    ) -> None:
        self.cfg = cfg
        self.train = train
        self.annotations: List[DiveAnnotation] = list(annotations)
        self._by_id: Dict[DiveID, DiveAnnotation] = {
            a.dive_id: a for a in self.annotations
        }
        self._selector = ExemplarSelector(
            train_pool=train_pool,
            strategy=cfg.exemplar_strategy,
            difficulty_window=cfg.exemplar_difficulty_window,
            fallback=cfg.exemplar_fallback,
            seed=cfg.val_split_seed,   # any deterministic seed works here
        )
        self._transform = build_frame_transform(cfg, train=train)
        # Caller can hand us a closure that returns the current epoch, so
        # exemplar choice is reproducible per (query, epoch).
        self._epoch_ref = epoch_ref or (lambda: 0)

    # ---- mandatory Dataset surface -----------------------------------

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> AQABatch:
        query = self.annotations[idx]
        epoch = int(self._epoch_ref())
        exemplar = self._selector.sample_one(query, epoch=epoch)
        return self._build_pair(query, exemplar)

    # ---- public helpers ----------------------------------------------

    def get_by_id(self, dive_id: DiveID) -> DiveAnnotation:
        return self._by_id[dive_id]

    def make_pair(
        self, query: DiveAnnotation, exemplar: DiveAnnotation
    ) -> AQABatch:
        """Public path for inference / counterfactual code to materialise
        a specific (query, exemplar) pair without going through __getitem__."""
        return self._build_pair(query, exemplar)

    def selector_coverage(self) -> Dict[str, Any]:
        """Expose exemplar-selector coverage for the cross-dataset table."""
        return self._selector.coverage_report()

    # ---- internals ----------------------------------------------------

    def _build_pair(
        self, query: DiveAnnotation, exemplar: DiveAnnotation
    ) -> AQABatch:
        q_frames = self._load(query)
        e_frames = self._load(exemplar)
        # judges_scores may be empty on the FineDiving release; emit a
        # length-0 tensor in that case rather than padding silently.
        q_judges = torch.tensor(query.judges_scores, dtype=torch.float32)
        e_judges = torch.tensor(exemplar.judges_scores, dtype=torch.float32)
        return AQABatch(
            query_frames=q_frames,
            exemplar_frames=e_frames,
            query_score=torch.tensor(query.final_score, dtype=torch.float32),
            exemplar_score=torch.tensor(exemplar.final_score, dtype=torch.float32),
            query_difficulty=torch.tensor(query.difficulty, dtype=torch.float32),
            exemplar_difficulty=torch.tensor(
                exemplar.difficulty, dtype=torch.float32
            ),
            query_step_transitions=torch.tensor(
                query.step_transitions, dtype=torch.long
            ),
            exemplar_step_transitions=torch.tensor(
                exemplar.step_transitions, dtype=torch.long
            ),
            query_action_type=query.action_type,
            exemplar_action_type=exemplar.action_type,
            query_sub_action_types=query.sub_action_types,
            exemplar_sub_action_types=exemplar.sub_action_types,
            query_judges_scores=q_judges,
            exemplar_judges_scores=e_judges,
            query_dive_id=query.dive_id,
            exemplar_dive_id=exemplar.dive_id,
        )

    def _load(self, ann: DiveAnnotation) -> torch.Tensor:
        frames_dir = self.cfg.path(self.cfg.finediving_frames_root) / ann.video_id
        # Deterministic per-dive jitter source at training time.
        rng = random.Random(
            hash((ann.dive_id, int(self._epoch_ref()))) & 0xFFFFFFFF
        ) if self.train else None
        indices = _sample_frame_indices(
            ann.start_frame, ann.end_frame,
            num_frames=self.cfg.num_frames,
            train=self.train, jitter=self.cfg.frame_index_jitter,
            rng=rng,
        )
        return _load_clip(
            frames_dir, indices, self._transform,
            filename_pattern=self.cfg.frame_filename_pattern,
        )


class PoseDataset(Dataset):
    """Pre-extracted 3D SMPL pose sequences for the diffusion / CF stack.

    Each ``data/poses/<dive_id>.npz`` contains:
        - ``theta``: (T, 24, 3) float32, per-frame axis-angle joint rotations
        - ``tau``:   (T, 3) float32, per-frame root translation
        - ``fps``:   int, source frame rate (always 30 in our pipeline)

    Failed extractions listed in ``data/poses/failed.txt`` are silently
    excluded; downstream code can query :attr:`failed_ids` to report on
    coverage gaps.
    """

    def __init__(
        self,
        dive_ids: Sequence[DiveID],
        cfg: DataCfg,
        require_all: bool = False,
    ) -> None:
        self.cfg = cfg
        self.failed_ids: FrozenSet[DiveID] = load_failed_pose_ids(
            cfg.path(cfg.pose_failed_list)
        )
        kept: List[DiveID] = []
        for did in dive_ids:
            if did in self.failed_ids:
                continue
            p = self._npz_path(did)
            if not p.exists():
                if require_all:
                    raise FileNotFoundError(f"missing pose .npz: {p}")
                log.debug("pose .npz missing for %s; skipping", did)
                continue
            kept.append(did)
        self.dive_ids: List[DiveID] = kept

    def _npz_path(self, dive_id: DiveID) -> Path:
        return self.cfg.path(self.cfg.pose_root) / f"{dive_id}.npz"

    def __len__(self) -> int:
        return len(self.dive_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dive_id = self.dive_ids[idx]
        with np.load(self._npz_path(dive_id)) as z:
            theta = torch.from_numpy(z["theta"].astype(np.float32))  # (T, 24, 3)
            tau = torch.from_numpy(z["tau"].astype(np.float32))      # (T, 3)
            fps = int(z["fps"]) if "fps" in z.files else self.cfg.pose_fps
        if theta.shape[-2:] != (self.cfg.pose_joint_count, self.cfg.pose_rotation_dim):
            raise ValueError(
                f"{dive_id}: theta shape {tuple(theta.shape)} mismatches "
                f"({self.cfg.pose_joint_count}, {self.cfg.pose_rotation_dim})"
            )
        return {
            "dive_id": dive_id,
            "theta": theta,
            "tau": tau,
            "fps": fps,
            "num_frames": theta.shape[0],
        }


class FaultAnnotations:
    """In-memory lookup over the FineDiving-HF half-point records.

    Loaded once at start-up; the 200 annotated dives fit comfortably in
    memory. Use :meth:`for_dive` to retrieve all half-point records for
    a given dive, and :meth:`stratum_for` to query the score quintile a
    dive belongs to (needed for stratified attribution-evaluation breakdowns).
    """

    # Score-quintile boundaries used by the stratified sampling protocol.
    QUINTILE_BOUNDS: Tuple[Tuple[float, float], ...] = (
        (0.0, 60.0), (60.0, 75.0), (75.0, 85.0), (85.0, 95.0), (95.0, 105.0),
    )

    def __init__(self, faults: Mapping[DiveID, DiveFaults]) -> None:
        self._faults: Dict[DiveID, DiveFaults] = dict(faults)
        self._record_counts = Counter(len(f.records) for f in self._faults.values())

    @classmethod
    def from_cfg(cls, cfg: DataCfg) -> "FaultAnnotations":
        path = cfg.path(cfg.faults_root) / cfg.faults_file
        return cls(load_fault_annotations(path))

    # ---- lookup ------------------------------------------------------

    def for_dive(self, dive_id: DiveID) -> Optional[DiveFaults]:
        return self._faults.get(dive_id)

    def __contains__(self, dive_id: DiveID) -> bool:
        return dive_id in self._faults

    def __len__(self) -> int:
        return len(self._faults)

    @property
    def dive_ids(self) -> List[DiveID]:
        return list(self._faults.keys())

    # ---- stratification ---------------------------------------------

    def stratum_for(self, score: float) -> int:
        """Return the score-quintile index (0..4) for the given final score."""
        for i, (lo, hi) in enumerate(self.QUINTILE_BOUNDS):
            if lo <= score < hi:
                return i
        # Highest bucket is inclusive on the right.
        return len(self.QUINTILE_BOUNDS) - 1

    def summary(self) -> Dict[str, Any]:
        """Aggregate statistics used in the dataset section of the paper."""
        records = [r for f in self._faults.values() for r in f.records]
        return {
            "num_dives": len(self._faults),
            "num_records": len(records),
            "records_per_dive_mean": (
                float(np.mean([len(f.records) for f in self._faults.values()]))
                if self._faults else 0.0
            ),
            "records_per_dive_std": (
                float(np.std([len(f.records) for f in self._faults.values()]))
                if self._faults else 0.0
            ),
            "record_count_histogram": dict(self._record_counts),
        }


# =============================================================================
# Splits
# =============================================================================


def _stratified_val_holdout(
    train_anns: Sequence[DiveAnnotation],
    val_fraction: float,
    seed: int,
    stratify_by_action_type: bool,
) -> Tuple[List[DiveAnnotation], List[DiveAnnotation]]:
    """Split ``train_anns`` into (train_kept, val) deterministically.

    Sorts by dive_id first so the split is invariant to filesystem order.
    If ``stratify_by_action_type``, holds out ``val_fraction`` of EACH
    action type separately, with at least one held-out dive per action
    type (when the action type has ≥ 1 sample); rare action types are
    therefore guaranteed to appear in val.

    If not stratifying, falls back to a single uniform random shuffle.
    Same (annotations, fraction, seed) always yields the same holdout —
    across re-runs and re-clones.
    """
    sorted_anns = sorted(train_anns, key=lambda a: a.dive_id)
    rng = np.random.default_rng(seed)

    if not stratify_by_action_type:
        n_val = max(1, int(round(len(sorted_anns) * val_fraction)))
        perm = rng.permutation(len(sorted_anns))
        val_idx = set(perm[:n_val].tolist())
        train_kept = [a for i, a in enumerate(sorted_anns) if i not in val_idx]
        val = [a for i, a in enumerate(sorted_anns) if i in val_idx]
        return train_kept, val

    # Stratified by action_type.
    by_action: Dict[str, List[DiveAnnotation]] = defaultdict(list)
    for a in sorted_anns:
        by_action[a.action_type].append(a)
    train_kept: List[DiveAnnotation] = []
    val: List[DiveAnnotation] = []
    for action_type in sorted(by_action.keys()):
        group = by_action[action_type]
        n = len(group)
        # At least 1 in val if the group has 2+ samples; otherwise keep
        # the single sample in train to avoid stripping the action type
        # entirely from training.
        n_val = max(1, int(round(n * val_fraction))) if n >= 2 else 0
        if n_val >= n:
            n_val = n - 1   # never empty the train side of an action type
        perm = rng.permutation(n)
        val_idx = set(perm[:n_val].tolist())
        for i, a in enumerate(group):
            if i in val_idx:
                val.append(a)
            else:
                train_kept.append(a)
    return train_kept, val


# =============================================================================
# DataModule — the factory that wires it all together
# =============================================================================


class _EpochCounter:
    """Mutable counter shared between the DataModule and its datasets.

    Lets ``ExemplarSelector`` deterministically vary exemplars per epoch
    without coupling the dataset to a training-loop callback.
    """

    def __init__(self) -> None:
        self.value: int = 0

    def __call__(self) -> int:
        return self.value


def _seed_worker(worker_id: int) -> None:
    """Per-worker seed so DataLoader workers are deterministic.

    Called by the DataLoader after fork. Matches the PyTorch deterministic
    DataLoader recipe.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DataModule:
    """Thin factory that owns annotations, datasets, samplers, and loaders.

    Lifecycle:
        dm = DataModule(cfg)
        dm.setup()                       # loads annotations, builds splits
        loader = dm.train_loader()       # DDP-aware, deterministic
        for batch in loader: ...
        dm.set_epoch(e)                  # before each new epoch (DDP shuffle + exemplars)

    The FineDiving-HF fault annotations are loaded lazily — only
    counterfactual / attribution evaluation needs them.
    """

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self._epoch = _EpochCounter()
        # Populated by setup().
        self._train_anns: Optional[List[DiveAnnotation]] = None
        self._val_anns: Optional[List[DiveAnnotation]] = None
        self._test_anns: Optional[List[DiveAnnotation]] = None
        self._faults: Optional[FaultAnnotations] = None

    # ---- one-time setup ----------------------------------------------

    def setup(self) -> None:
        d = self.cfg.data
        anno_dir = d.path(d.finediving_anno_root)
        # The reference release stores one annotation pickle plus two split
        # pickles. We support either:
        #   (a) train.pkl / test.pkl (curated dict format), or
        #   (b) fine-grained_annotation_aqa.pkl + train_split.pkl + test_split.pkl
        #       (reference release).
        aqa_pkl = anno_dir / "fine-grained_annotation_aqa.pkl"
        train_pkl = anno_dir / "train.pkl"
        test_pkl = anno_dir / "test.pkl"
        train_split = anno_dir / "train_split.pkl"
        test_split = anno_dir / "test_split.pkl"
        if aqa_pkl.exists() and train_split.exists():
            log.info("Loading FineDiving via reference release format")
            train_full = load_finediving_annotations(
                aqa_pkl, num_step_transitions=d.num_step_transitions,
                num_judges=d.num_judges,
                require_judges_scores=d.require_judges_scores,
                split_list_path=train_split,
            )
            self._test_anns = load_finediving_annotations(
                aqa_pkl, num_step_transitions=d.num_step_transitions,
                num_judges=d.num_judges,
                require_judges_scores=d.require_judges_scores,
                split_list_path=test_split,
            )
        elif train_pkl.exists() and test_pkl.exists():
            log.info("Loading FineDiving via curated train.pkl/test.pkl format")
            train_full = load_finediving_annotations(
                train_pkl, num_step_transitions=d.num_step_transitions,
                num_judges=d.num_judges,
                require_judges_scores=d.require_judges_scores,
            )
            self._test_anns = load_finediving_annotations(
                test_pkl, num_step_transitions=d.num_step_transitions,
                num_judges=d.num_judges,
                require_judges_scores=d.require_judges_scores,
            )
        else:
            raise FileNotFoundError(
                f"No FineDiving annotation pickles found under {anno_dir}. "
                "Expected either (train.pkl, test.pkl) or "
                "(fine-grained_annotation_aqa.pkl, train_split.pkl, test_split.pkl)."
            )

        self._train_anns, self._val_anns = _stratified_val_holdout(
            train_full,
            val_fraction=d.val_fraction_of_train,
            seed=d.val_split_seed,
            stratify_by_action_type=d.val_stratify_by_action_type,
        )
        log.info(
            "Splits — train: %d, val: %d, test: %d (stratified=%s)",
            len(self._train_anns), len(self._val_anns), len(self._test_anns),
            d.val_stratify_by_action_type,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    # ---- summary ------------------------------------------------------

    def split_summary(self) -> Dict[str, Any]:
        """Compact summary for logs / the paper's dataset section."""
        self._require("_train_anns")
        out: Dict[str, Any] = {}
        for name, anns in (
            ("train", self._train_anns), ("val", self._val_anns),
            ("test", self._test_anns),
        ):
            if anns is None:
                continue
            scores = [a.final_score for a in anns]
            action_counter = Counter(a.action_type for a in anns)
            out[name] = {
                "n": len(anns),
                "n_action_types": len(action_counter),
                "score_min": float(min(scores)) if scores else 0.0,
                "score_max": float(max(scores)) if scores else 0.0,
                "score_mean": float(np.mean(scores)) if scores else 0.0,
            }
        return out

    # ---- dataset constructors ----------------------------------------

    def _make_dataset(
        self, anns: Sequence[DiveAnnotation], train: bool
    ) -> FineDivingDataset:
        assert self._train_anns is not None, "call DataModule.setup() first"
        return FineDivingDataset(
            annotations=anns,
            train_pool=self._train_anns,   # exemplars only ever come from train
            cfg=self.cfg.data,
            train=train,
            epoch_ref=self._epoch,
        )

    def train_dataset(self) -> FineDivingDataset:
        return self._make_dataset(self._require("_train_anns"), train=True)

    def val_dataset(self) -> FineDivingDataset:
        return self._make_dataset(self._require("_val_anns"), train=False)

    def test_dataset(self) -> FineDivingDataset:
        return self._make_dataset(self._require("_test_anns"), train=False)

    # ---- loaders ------------------------------------------------------

    def train_loader(self) -> DataLoader:
        return self._make_loader(
            self.train_dataset(), shuffle=True, drop_last=True,
            batch_size=self.cfg.train.aqa_batch_size_per_gpu,
        )

    def val_loader(self) -> DataLoader:
        return self._make_loader(
            self.val_dataset(), shuffle=False, drop_last=False,
            batch_size=self.cfg.train.aqa_batch_size_per_gpu,
        )

    def test_loader(self) -> DataLoader:
        return self._make_loader(
            self.test_dataset(), shuffle=False, drop_last=False,
            batch_size=self.cfg.train.aqa_batch_size_per_gpu,
        )

    # ---- fault annotations (lazy) ------------------------------------

    def faults(self) -> FaultAnnotations:
        if self._faults is None:
            self._faults = FaultAnnotations.from_cfg(self.cfg.data)
        return self._faults

    # ---- helpers ------------------------------------------------------

    def _require(self, name: str) -> List[DiveAnnotation]:
        val = getattr(self, name)
        if val is None:
            raise RuntimeError("call DataModule.setup() before requesting splits")
        return val

    def _make_loader(
        self, dataset: FineDivingDataset,
        shuffle: bool, drop_last: bool, batch_size: int,
    ) -> DataLoader:
        d = self.cfg.data
        sampler: Optional[Sampler] = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                dataset, shuffle=shuffle, drop_last=drop_last,
                seed=self.cfg.train.seed,
            )
            shuffle = False  # the sampler does the shuffling
        generator = torch.Generator()
        generator.manual_seed(self.cfg.train.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=d.num_workers,
            pin_memory=d.pin_memory,
            prefetch_factor=d.prefetch_factor if d.num_workers > 0 else None,
            persistent_workers=d.persistent_workers and d.num_workers > 0,
            drop_last=drop_last,
            worker_init_fn=_seed_worker,
            generator=generator,
            collate_fn=_collate_aqa_batch,
        )


def _collate_aqa_batch(items: List[AQABatch]) -> Dict[str, Any]:
    """Custom collate that stacks tensors and keeps string fields as tuples.

    DataLoader's default collate would try to torch.stack() the action-type
    strings, which throws. We hand-stack each field instead. Judge-score
    tensors may be length 0 (when the source pickle didn't carry judges)
    and are stacked along the batch dim regardless.
    """
    keys_tensor = (
        "query_frames", "exemplar_frames",
        "query_score", "exemplar_score",
        "query_difficulty", "exemplar_difficulty",
        "query_step_transitions", "exemplar_step_transitions",
        "query_judges_scores", "exemplar_judges_scores",
    )
    keys_passthrough = (
        "query_action_type", "exemplar_action_type",
        "query_sub_action_types", "exemplar_sub_action_types",
        "query_dive_id", "exemplar_dive_id",
    )
    batch: Dict[str, Any] = {}
    for k in keys_tensor:
        batch[k] = torch.stack([getattr(it, k) for it in items], dim=0)
    for k in keys_passthrough:
        batch[k] = tuple(getattr(it, k) for it in items)
    return batch


# =============================================================================
# Public surface
# =============================================================================

__all__ = [
    "DiveID",
    "DiveAnnotation",
    "FaultRecord",
    "DiveFaults",
    "AQABatch",
    "load_finediving_annotations",
    "load_fault_annotations",
    "load_failed_pose_ids",
    "build_frame_transform",
    "ExemplarSelector",
    "FineDivingDataset",
    "PoseDataset",
    "FaultAnnotations",
    "DataModule",
]

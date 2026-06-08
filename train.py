"""train.py — CLI training script for AQA and diffusion.

Usage
-----
Single-GPU debug:
    python -m src.train aqa --run-name aqa_dev
    python -m src.train diffusion --run-name diff_dev

Multi-GPU DDP (4× 3090, the paper's default):
    torchrun --nproc_per_node=4 -m src.train aqa --run-name aqa_baseline
    torchrun --nproc_per_node=4 -m src.train diffusion --run-name diff_v1

Outputs
-------
Everything goes under ``experiments/<run_name>/``:
    config.yaml         The resolved Cfg snapshotted at start (Cfg.snapshot).
    provenance.yaml     git SHA, torch / CUDA / cuDNN versions, fingerprint.
    ckpt_epoch####.pt   Per-validation-cycle checkpoint; CheckpointKeeper
                        prunes to top-K by validation metric.
    best.pt             Symlink to the best checkpoint.
    train.log           Rank-0 logging output.

Design notes
------------
- DDP launching is via ``torchrun``; the script reads ``RANK`` /
  ``WORLD_SIZE`` / ``LOCAL_RANK`` from env. Works as a single process
  without ``torchrun`` for debugging.
- :class:`ContextEncoder` lives here (not in :mod:`diffusion`) because
  it's training scaffolding — the (action_type, difficulty) → context
  mapping is *learned* jointly with the diffusion model and saved in
  the checkpoint. Inference / CF code imports it from here.
- AMP via ``torch.autocast``; default dtype is bf16 (3090 supports it,
  sidesteps fp16 underflow). The GradScaler is only used for fp16.
- :class:`EMA` maintains a decoupled shadow model with decay 0.9999 per
  EDGE (model/diffusion.py:63 — verified). Validation runs on the EMA
  shadow, not the trained model.
- :class:`CheckpointKeeper` keeps top-K by *validation metric* (not
  mtime), so a transient training spike can't lose the actual best.

Changes from previous version
-----------------------------
- Issue 2: AQA training calls :func:`TSAModel.forward` with
  ``use_symmetric=cfg.aqa.use_symmetric_training``, threading
  ``query_score`` through so the symmetric MSE term (FineDiving
  helper.py:163 — mse_qe + mse_eq) can fire. Validation explicitly
  passes ``use_symmetric=False`` for inference behaviour.
- Issue 5: ``current_epoch`` is now threaded into
  :func:`train_one_epoch_aqa` so the model's internal curriculum gate
  (FineDiving helper.py:145-164) can decide between GT and predicted
  step transitions. ``gt_query_transitions`` and
  ``gt_exemplar_transitions`` from the batch flow into the model.
- Issue 4: ``EMA`` default decay corrected to 0.9999 (was 0.999); the
  cfg already passes ``cfg.train.diffusion_ema_decay=0.9999`` so this
  is a no-op for runs that go through the cfg, but the default and
  docstrings now match for code-review clarity.
- Curriculum diagnostic ``used_gt_transitions`` is logged alongside
  the loss components so reviewers can see when the curriculum gate
  flips during training.
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .configs import Cfg
from .data import DataModule, PoseDataset, load_finediving_annotations
from .aqa_model import (
    TSAModel, compute_aqa_loss, load_i3d_kinetics_weights,
)
from .diffusion import (
    MotionDiffusion, compute_diffusion_loss, tokenize_pose,
    axis_angle_to_rotmat, derive_foot_contacts,
    filter_top_quartile_by_action, load_smpl_rest_positions,
    SMPL_FOOT_JOINTS,
)
from .metrics import spearman_correlation, r_l2

log = logging.getLogger(__name__)


# =============================================================================
# DDP + rank utilities
# =============================================================================


def setup_ddp(backend: str = "nccl") -> Tuple[int, int, int, torch.device]:
    """Initialise the DDP process group and return (rank, world_size, local_rank, device).

    Works in two modes:
        - Launched via ``torchrun``: env carries RANK / WORLD_SIZE / LOCAL_RANK.
        - Single process: env unset, falls back to world_size=1 and CPU/cuda:0.
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return rank, world_size, local_rank, device


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def all_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    """Concatenate a same-shape tensor across all ranks (DistributedSampler
    pads each rank to the same per-rank count, so sizes match)."""
    if not (dist.is_available() and dist.is_initialized()):
        return t
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t.contiguous())
    return torch.cat(gathered, dim=0)


def reduce_mean_scalar(value: float, world_size: int, device: torch.device) -> float:
    """Mean-reduce a Python scalar across DDP ranks."""
    if world_size <= 1 or not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(state: Dict, path: Path) -> None:
    """Save a checkpoint dict, but only on the main process."""
    if not is_main_process():
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def _strip_ddp(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class CheckpointKeeper:
    """Top-K checkpoint manager, ordered by validation metric value.

    The keeper holds (metric, path) entries; on every ``add``, the list is
    re-sorted and entries beyond ``top_k`` are deleted from disk. Use
    ``higher_is_better=False`` for losses (lower is better).
    """

    def __init__(self, run_dir: Path, top_k: int, higher_is_better: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.top_k = max(1, int(top_k))
        self.higher_is_better = higher_is_better
        self.entries: List[Tuple[float, Path]] = []

    def add(self, metric: float, path: Path) -> bool:
        """Register a checkpoint; return True if this is the new best."""
        path = Path(path)
        self.entries.append((float(metric), path))
        self.entries.sort(key=lambda e: e[0], reverse=self.higher_is_better)
        # Prune tail. A pruned entry's file is always unlinked — including
        # the just-added one, since being pruned means a strictly better
        # checkpoint already exists in the keeper.
        while len(self.entries) > self.top_k:
            _, old_path = self.entries.pop()
            if old_path.exists():
                try:
                    old_path.unlink()
                except OSError as e:
                    log.warning("could not prune %s: %s", old_path, e)
        return bool(self.entries) and self.entries[0][1] == path

    def best_path(self) -> Optional[Path]:
        return self.entries[0][1] if self.entries else None

    def maybe_symlink_best(self, link: Path) -> None:
        """Point ``link`` (e.g. ``run_dir/best.pt``) at the current best."""
        if not is_main_process() or not self.entries:
            return
        link = Path(link)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(self.entries[0][1].name)


def load_checkpoint_into(
    model: nn.Module, path: Path, *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str = "cpu",
) -> Dict:
    """Load a checkpoint; supports both DDP-wrapped and bare models."""
    state = torch.load(str(path), map_location=map_location)
    _strip_ddp(model).load_state_dict(state["state_dict"], strict=True)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    log.info("resumed from %s (epoch=%s)", path, state.get("epoch"))
    return state


# =============================================================================
# EMA — for the diffusion model
# =============================================================================


class EMA:
    """Exponential moving average of model parameters.

    EDGE uses decay 0.9999 (model/diffusion.py:63 — verified against
    the released codebase). The shadow model is a deep copy that's
    updated *after* every optimiser step. We sample / validate from the
    shadow, not the live model — this is what's saved at checkpoint
    time.

    Note: the cfg default ``cfg.train.diffusion_ema_decay`` is the
    authoritative source for the decay; the class default below is set
    to the EDGE value so a caller that forgets to pass the cfg still
    gets paper-faithful behaviour.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(_strip_ddp(model))
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        src = _strip_ddp(model)
        for ema_p, p in zip(self.shadow.parameters(), src.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        # Buffers (e.g. BN running stats) get copied without averaging.
        for ema_b, b in zip(self.shadow.buffers(), src.buffers()):
            ema_b.data.copy_(b.data)

    def state_dict(self) -> Dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        self.shadow.load_state_dict(state_dict)


# =============================================================================
# ContextEncoder — (action_type, difficulty) → diffusion context
# =============================================================================


class ContextEncoder(nn.Module):
    """Learnable encoder from ``(action_type, difficulty)`` to the diffusion
    model's context vector.

    Constructed once at training start from the list of action types seen
    in the training pool; an unknown action type at inference time logs a
    warning and defaults to action index 0. Saved jointly with the
    diffusion model in the checkpoint so inference and counterfactual
    code can reproduce the exact context vectors used during training.

    Output shape: ``(B, cond_action_embedding_dim + cond_summary_dim)``.
    """

    def __init__(
        self,
        action_types: List[str],
        cond_action_dim: int,
        cond_summary_dim: int,
    ) -> None:
        super().__init__()
        # Sort + dedupe so the index assignment is reproducible.
        self.action_types: Tuple[str, ...] = tuple(sorted(set(action_types)))
        self._index: Dict[str, int] = {t: i for i, t in enumerate(self.action_types)}
        self.action_embedding = nn.Embedding(len(self.action_types), cond_action_dim)
        self.difficulty_proj = nn.Sequential(
            nn.Linear(1, cond_summary_dim),
            nn.SiLU(),
            nn.Linear(cond_summary_dim, cond_summary_dim),
        )
        nn.init.normal_(self.action_embedding.weight, std=0.02)

    def indices_for(self, action_type_strings: List[str], device: torch.device) -> torch.Tensor:
        ids: List[int] = []
        for t in action_type_strings:
            if t in self._index:
                ids.append(self._index[t])
            else:
                log.warning("unseen action type at inference: %s", t)
                ids.append(0)
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(self, action_types: List[str], difficulty: torch.Tensor) -> torch.Tensor:
        idx = self.indices_for(action_types, device=difficulty.device)
        a = self.action_embedding(idx)                                 # (B, action_dim)
        d = self.difficulty_proj(difficulty.unsqueeze(-1).float())      # (B, summary_dim)
        return torch.cat([a, d], dim=-1)


# =============================================================================
# Config loading + run directory setup
# =============================================================================


def load_cfg(args: argparse.Namespace) -> Cfg:
    """Load the base Cfg, apply CLI overrides, and validate."""
    cfg = Cfg.from_yaml(args.config) if args.config else Cfg()
    overrides: Dict = {}
    if getattr(args, "run_name", None):
        overrides["run_name"] = args.run_name
    if getattr(args, "seed_override", None) is not None:
        overrides["train"] = replace(cfg.train, seed=args.seed_override)
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def prepare_run_dir(cfg: Cfg) -> Path:
    """Create the run directory and write provenance (rank-0 only)."""
    run_dir = cfg.data.repo_root / "experiments" / cfg.run_name
    if is_main_process():
        cfg.snapshot(run_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return run_dir


def setup_logging(run_dir: Path) -> None:
    """Configure root logger; rank-0 also writes to a file under run_dir."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(run_dir / "train.log"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


# =============================================================================
# AQA training
# =============================================================================


def _build_aqa_optimizer(model: nn.Module, cfg: Cfg) -> torch.optim.Optimizer:
    """Two parameter groups: I3D backbone (low LR) + everything else (high LR)."""
    target = _strip_ddp(model)
    backbone_params = list(target.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    head_params = [p for p in target.parameters() if id(p) not in backbone_ids]
    return torch.optim.Adam(
        [
            {"params": backbone_params, "lr": cfg.train.aqa_lr_backbone},
            {"params": head_params, "lr": cfg.train.aqa_lr_head},
        ],
        weight_decay=cfg.train.aqa_weight_decay,
    )


def _move_aqa_batch(batch: Dict, device: torch.device) -> Dict:
    """Move tensor fields of the collated AQA batch to ``device``."""
    out: Dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _aqa_model_call(
    model: nn.Module, batch: Dict, cfg: Cfg,
    *,
    current_epoch: Optional[int],
    use_symmetric: bool,
):
    """Centralised call site for :meth:`TSAModel.forward`.

    Reads the curriculum and symmetric-training kwargs from the batch
    and the config. Keeping this in one place means
    :func:`train_one_epoch_aqa` and :func:`validate_aqa` can't drift
    apart on which arguments they pass.

    The ``query_score`` / GT transition fields are surfaced from the
    batch with ``.get`` so an older data loader that doesn't provide
    them gracefully falls back to inference-style behaviour (no
    curriculum, no symmetric path).
    """
    return model(
        batch["query_frames"], batch["exemplar_frames"], batch["exemplar_score"],
        # Symmetric training (Issue 2): needs the query's GT score to
        # compute the exemplar-as-query MSE term (mse_eq, helper.py:163).
        query_score=batch.get("query_score") if use_symmetric else None,
        # Step-transition curriculum (Issue 5; helper.py:145-164). The
        # model's internal ``_should_use_gt`` gate decides whether to
        # actually use these — early epochs typically do; late epochs
        # switch to predicted transitions to match inference.
        gt_query_transitions=batch.get("query_step_transitions"),
        gt_exemplar_transitions=batch.get("exemplar_step_transitions"),
        current_epoch=current_epoch,
        use_symmetric=use_symmetric,
    )


def train_one_epoch_aqa(
    model: nn.Module, loader, optimizer, scaler, autocast_dtype,
    device: torch.device, cfg: Cfg,
    *,
    current_epoch: int,
) -> float:
    """One epoch of AQA training; returns the mean training loss.

    Args:
        model: the (possibly DDP-wrapped) TSAModel.
        loader: training DataLoader.
        optimizer, scaler, autocast_dtype: AMP machinery.
        device: GPU device.
        cfg: the global Cfg (reads cfg.aqa for symmetric / curriculum flags).
        current_epoch: 0-indexed epoch counter. Threaded into
            :meth:`TSAModel.forward` so the curriculum gate
            (``_should_use_gt``) can decide between GT and predicted
            step transitions at this point in training.
    """
    model.train()
    total = 0.0
    n_batches = 0
    n_used_gt = 0
    use_amp = autocast_dtype != torch.float32

    for batch_idx, batch in enumerate(loader):
        batch = _move_aqa_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            out = _aqa_model_call(
                model, batch, cfg,
                current_epoch=current_epoch,
                use_symmetric=cfg.aqa.use_symmetric_training,
            )
            losses = compute_aqa_loss(out, batch, cfg.aqa, cfg.data)
        loss = losses["total"]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        n_batches += 1
        # Diagnostic: count batches where the curriculum gate used GT
        # transitions. Helps reviewers see when the gate flips.
        used_gt = bool(getattr(out, "used_gt_transitions", False))
        n_used_gt += int(used_gt)
        if is_main_process() and batch_idx % cfg.train.log_every_n_steps == 0:
            log.info(
                "  step %d/%d loss=%.4f bce=%.4f mse=%.4f used_gt_trans=%s",
                batch_idx, len(loader),
                float(loss.item()),
                float(losses["bce"].item()), float(losses["mse"].item()),
                used_gt,
            )

    if is_main_process():
        log.info(
            "  epoch %d: %d/%d batches used GT step transitions (curriculum gate)",
            current_epoch, n_used_gt, n_batches,
        )
    return total / max(1, n_batches)


@torch.no_grad()
def validate_aqa(model: nn.Module, loader, device: torch.device, cfg: Cfg) -> Dict[str, float]:
    """Compute Spearman / R-ℓ2 over the validation split.

    Validation runs in inference mode: ``use_symmetric=False`` and no
    curriculum (``current_epoch=None``), so the model uses its predicted
    step transitions regardless of training-time settings. This matches
    how the deployed model is queried by :mod:`evaluate` and
    :mod:`counterfactual`.
    """
    model.eval()
    preds_local: List[torch.Tensor] = []
    targets_local: List[torch.Tensor] = []
    for batch in loader:
        batch = _move_aqa_batch(batch, device)
        out = _aqa_model_call(
            model, batch, cfg,
            current_epoch=None,
            use_symmetric=False,
        )
        preds_local.append(out.predicted_score)
        targets_local.append(batch["query_score"])
    preds = torch.cat(preds_local, dim=0)
    targets = torch.cat(targets_local, dim=0)
    # Gather across DDP.
    preds = all_gather_tensor(preds).cpu().numpy()
    targets = all_gather_tensor(targets).cpu().numpy()
    return {
        "spearman": float(spearman_correlation(preds, targets)),
        "r_l2": float(r_l2(preds, targets, (cfg.data.score_min, cfg.data.score_max))),
    }


def train_aqa(cfg: Cfg, args: argparse.Namespace, world_size: int, device: torch.device) -> None:
    run_dir = prepare_run_dir(cfg)
    setup_logging(run_dir)
    log.info("AQA training run %s (rank %d / %d, device=%s)",
             cfg.run_name, int(os.environ.get("RANK", 0)), world_size, device)
    if is_main_process():
        log.info(
            "AQA settings: use_symmetric=%s, curriculum_threshold_epochs=%d",
            cfg.aqa.use_symmetric_training,
            cfg.aqa.curriculum_threshold_epochs,
        )

    # Model.
    model = TSAModel(cfg.aqa, cfg.data, use_stsa=args.use_stsa).to(device)
    i3d_path = cfg.data.path(cfg.aqa.i3d_pretrained)
    if i3d_path.exists():
        load_i3d_kinetics_weights(model, i3d_path, map_location="cpu")
    elif is_main_process():
        log.warning("I3D pretrained weights not found at %s; training from scratch", i3d_path)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=cfg.train.find_unused_parameters)

    # Optimiser + AMP.
    optimizer = _build_aqa_optimizer(model, cfg)
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.train.amp_dtype]
    scaler = torch.amp.GradScaler() if cfg.train.amp_dtype == "fp16" else None

    # Data.
    dm = DataModule(cfg)
    dm.setup()

    # Resume?
    start_epoch = 0
    if args.resume:
        state = load_checkpoint_into(model, args.resume, optimizer=optimizer, map_location="cpu")
        start_epoch = int(state.get("epoch", -1)) + 1

    keeper = CheckpointKeeper(run_dir, cfg.train.keep_top_k_checkpoints, higher_is_better=True)

    for epoch in range(start_epoch, cfg.train.aqa_epochs):
        dm.set_epoch(epoch)
        if is_main_process():
            log.info("== epoch %d / %d ==", epoch, cfg.train.aqa_epochs)
        train_loss = train_one_epoch_aqa(
            model, dm.train_loader(), optimizer, scaler, autocast_dtype, device, cfg,
            current_epoch=epoch,
        )
        if (epoch + 1) % cfg.train.val_every_n_epochs == 0:
            metrics = validate_aqa(model, dm.val_loader(), device, cfg)
            if is_main_process():
                log.info("epoch %d val: ρ=%.4f R-ℓ2=%.4f train_loss=%.4f",
                         epoch, metrics["spearman"], metrics["r_l2"], train_loss)
                state = {
                    "state_dict": _strip_ddp(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "config_fingerprint": cfg.fingerprint(),
                    "metrics": metrics,
                    "use_stsa": args.use_stsa,
                }
                ckpt_path = run_dir / f"ckpt_epoch{epoch:04d}.pt"
                save_checkpoint(state, ckpt_path)
                is_best = keeper.add(metrics["spearman"], ckpt_path)
                if is_best:
                    keeper.maybe_symlink_best(run_dir / "best.pt")
            if world_size > 1:
                dist.barrier()


# =============================================================================
# Diffusion training
# =============================================================================


def _tokenize_pose_batch(
    theta: torch.Tensor, tau: torch.Tensor,
    kinematics, velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Convert ``(B, T, 24, 3)`` axis-angle + ``(B, T, 3)`` root into ``(B, T, 151)``
    diffusion tokens, with contacts derived from foot-joint velocity."""
    with torch.no_grad():
        rotmat = axis_angle_to_rotmat(theta)
        positions = kinematics(rotmat, tau)
        contacts = derive_foot_contacts(
            positions[..., SMPL_FOOT_JOINTS, :],
            velocity_threshold=velocity_threshold,
        )
    return tokenize_pose(theta, tau, contacts=contacts)


def _build_diffusion_dataset(cfg: Cfg) -> Tuple[PoseDataset, PoseDataset, List[str]]:
    """High-score-filtered train/val pose datasets and the action-type list."""
    train_pkl = cfg.data.path(cfg.data.finediving_anno_root) / "train.pkl"
    train_anns = load_finediving_annotations(
        train_pkl, num_step_transitions=cfg.data.num_step_transitions,
        num_judges=cfg.data.num_judges,
    )
    high_score = filter_top_quartile_by_action(
        train_anns, quartile=cfg.diffusion.high_score_quartile,
    )
    # Sort dive_ids for deterministic splitting (10% held out for val).
    sorted_ids = sorted([a.dive_id for a in high_score])
    n_val = max(1, int(0.10 * len(sorted_ids)))
    rng = np.random.default_rng(cfg.data.val_split_seed)
    perm = rng.permutation(len(sorted_ids))
    val_set = {sorted_ids[i] for i in perm[:n_val]}
    val_ids = [d for d in sorted_ids if d in val_set]
    train_ids = [d for d in sorted_ids if d not in val_set]
    train_ds = PoseDataset(train_ids, cfg.data)
    val_ds = PoseDataset(val_ids, cfg.data)
    action_types = sorted({a.action_type for a in high_score})
    return train_ds, val_ds, action_types


def _collate_pose(items: List[Dict]) -> Dict:
    """Collate function for ``PoseDataset``; stacks theta/tau, keeps ids."""
    return {
        "dive_ids": [it["dive_id"] for it in items],
        "theta": torch.stack([it["theta"] for it in items], dim=0),
        "tau": torch.stack([it["tau"] for it in items], dim=0),
    }


def _build_diffusion_loaders(cfg: Cfg, train_ds, val_ds, world_size: int):
    """DDP-aware DataLoaders for the pose datasets."""
    from torch.utils.data import DataLoader, DistributedSampler
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=cfg.train.seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    common = dict(
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=_collate_pose,
        persistent_workers=cfg.data.persistent_workers and cfg.data.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.diffusion_batch_size_per_gpu,
        shuffle=(train_sampler is None), sampler=train_sampler,
        drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.diffusion_batch_size_per_gpu,
        shuffle=False, sampler=val_sampler, drop_last=False, **common,
    )
    return train_loader, val_loader, train_sampler


def _build_diffusion_optimizer_and_scheduler(
    params, cfg: Cfg,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW + linear warmup over ``diffusion_warmup_steps`` steps."""
    optimizer = torch.optim.AdamW(
        params, lr=cfg.train.diffusion_lr,
        weight_decay=cfg.train.diffusion_weight_decay,
    )
    warmup = max(1, cfg.train.diffusion_warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup),
    )
    return optimizer, scheduler


def train_one_epoch_diffusion(
    model: nn.Module, encoder: ContextEncoder, ema: EMA, annotations_by_id: Dict,
    loader, optimizer, scheduler, scaler, autocast_dtype,
    device: torch.device, cfg: Cfg, global_step: int,
) -> Tuple[float, int]:
    """One epoch; returns ``(mean_total_loss, updated_global_step)``."""
    model.train()
    encoder.train()
    total = 0.0
    n_batches = 0
    use_amp = autocast_dtype != torch.float32
    kinematics = _strip_ddp(model).kinematics
    for batch in loader:
        theta = batch["theta"].to(device, non_blocking=True)
        tau = batch["tau"].to(device, non_blocking=True)
        # Per-sample annotations (for action type + difficulty).
        action_types = [annotations_by_id[d].action_type for d in batch["dive_ids"]]
        difficulties = torch.tensor(
            [annotations_by_id[d].difficulty for d in batch["dive_ids"]],
            dtype=torch.float32, device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            x = _tokenize_pose_batch(theta, tau, kinematics)
            context = encoder(action_types, difficulties)
            losses = compute_diffusion_loss(_strip_ddp(model), x, context)
        loss = losses["total"]
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(encoder.parameters()),
                cfg.train.diffusion_grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(encoder.parameters()),
                cfg.train.diffusion_grad_clip,
            )
            optimizer.step()
        scheduler.step()
        ema.update(model)

        total += float(loss.item())
        n_batches += 1
        global_step += 1
        if is_main_process() and global_step % cfg.train.log_every_n_steps == 0:
            log.info("  step %d total=%.4f simple=%.4f pos=%.4f vel=%.4f contact=%.4f lr=%.2e",
                     global_step,
                     float(loss.item()),
                     float(losses["simple"].item()),
                     float(losses["pos"].item()),
                     float(losses["vel"].item()),
                     float(losses["contact"].item()),
                     float(scheduler.get_last_lr()[0]))
    return total / max(1, n_batches), global_step


@torch.no_grad()
def validate_diffusion(
    ema_model: nn.Module, encoder: ContextEncoder, annotations_by_id: Dict,
    loader, device: torch.device,
) -> Dict[str, float]:
    """Compute L_simple on the EMA model over the val pose split.

    Cheap proxy for sampling quality; the full PFC + cross-dataset
    eval lives in :mod:`evaluate`.
    """
    ema_model.eval()
    encoder.eval()
    total = 0.0
    n = 0
    kinematics = ema_model.kinematics
    for batch in loader:
        theta = batch["theta"].to(device, non_blocking=True)
        tau = batch["tau"].to(device, non_blocking=True)
        action_types = [annotations_by_id[d].action_type for d in batch["dive_ids"]]
        difficulties = torch.tensor(
            [annotations_by_id[d].difficulty for d in batch["dive_ids"]],
            dtype=torch.float32, device=device,
        )
        x = _tokenize_pose_batch(theta, tau, kinematics)
        context = encoder(action_types, difficulties)
        # Same noising as training but use the EMA model. The validation
        # target matches the diffusion's prediction mode (x0 vs eps):
        # in x0 mode the raw output should equal x; in eps mode it
        # should equal the sampled noise. predict_x0_clean unifies
        # both for a consistent proxy metric.
        B = x.shape[0]
        t = torch.randint(0, ema_model.scheduler.num_steps, (B,), device=device)
        noise = torch.randn_like(x)
        z_t = ema_model.scheduler.q_sample(x, t, noise=noise)
        x_hat = ema_model.predict_x0_clean(z_t, t, context)
        total += float(F.mse_loss(x_hat, x).item())
        n += 1
    return {"L_simple": total / max(1, n)}


def train_diffusion(cfg: Cfg, args: argparse.Namespace, world_size: int, device: torch.device) -> None:
    run_dir = prepare_run_dir(cfg)
    setup_logging(run_dir)
    log.info("Diffusion training run %s (rank %d / %d, device=%s)",
             cfg.run_name, int(os.environ.get("RANK", 0)), world_size, device)
    if is_main_process():
        log.info(
            "Diffusion settings: ema_decay=%.4f, predict_x0=%s, p2_weighting=%s, "
            "λ_simple=%.3f λ_vel=%.3f λ_pos=%.3f λ_contact=%.3f",
            cfg.train.diffusion_ema_decay,
            cfg.diffusion.predict_x0,
            cfg.diffusion.p2_loss_weighting,
            cfg.diffusion.lambda_simple, cfg.diffusion.lambda_vel,
            cfg.diffusion.lambda_pos, cfg.diffusion.lambda_contact,
        )

    # Optional: load exact SMPL rest positions.
    rest_positions = None
    if args.smpl_rest_positions:
        rest_positions = load_smpl_rest_positions(args.smpl_rest_positions)
        log.info("loaded SMPL rest positions from %s", args.smpl_rest_positions)

    # Data + annotations.
    train_ds, val_ds, action_types = _build_diffusion_dataset(cfg)
    train_pkl = cfg.data.path(cfg.data.finediving_anno_root) / "train.pkl"
    train_anns = load_finediving_annotations(
        train_pkl, num_step_transitions=cfg.data.num_step_transitions,
        num_judges=cfg.data.num_judges,
    )
    annotations_by_id = {a.dive_id: a for a in train_anns}

    # Model + encoder.
    model = MotionDiffusion(cfg.diffusion, cfg.data, rest_positions=rest_positions).to(device)
    encoder = ContextEncoder(
        action_types,
        cond_action_dim=cfg.diffusion.cond_action_embedding_dim,
        cond_summary_dim=cfg.diffusion.cond_summary_dim,
    ).to(device)
    ema = EMA(model, decay=cfg.train.diffusion_ema_decay)
    ema.shadow.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=cfg.train.find_unused_parameters)
        encoder = DDP(encoder, device_ids=[device.index] if device.type == "cuda" else None,
                      find_unused_parameters=cfg.train.find_unused_parameters)

    optimizer, scheduler = _build_diffusion_optimizer_and_scheduler(
        list(model.parameters()) + list(encoder.parameters()), cfg,
    )
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.train.amp_dtype]
    scaler = torch.amp.GradScaler() if cfg.train.amp_dtype == "fp16" else None

    train_loader, val_loader, train_sampler = _build_diffusion_loaders(
        cfg, train_ds, val_ds, world_size,
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        state = load_checkpoint_into(model, args.resume, optimizer=optimizer,
                                     scheduler=scheduler, map_location="cpu")
        if "ema" in state:
            ema.load_state_dict(state["ema"])
        if "encoder" in state:
            _strip_ddp(encoder).load_state_dict(state["encoder"])
        start_epoch = int(state.get("epoch", -1)) + 1
        global_step = int(state.get("global_step", 0))

    # L_simple is *lower* is better.
    keeper = CheckpointKeeper(run_dir, cfg.train.keep_top_k_checkpoints, higher_is_better=False)

    for epoch in range(start_epoch, cfg.train.diffusion_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main_process():
            log.info("== epoch %d / %d ==", epoch, cfg.train.diffusion_epochs)
        train_loss, global_step = train_one_epoch_diffusion(
            model, _strip_ddp(encoder), ema, annotations_by_id,
            train_loader, optimizer, scheduler, scaler, autocast_dtype,
            device, cfg, global_step,
        )
        if (epoch + 1) % cfg.train.val_every_n_epochs == 0:
            metrics = validate_diffusion(
                ema.shadow, _strip_ddp(encoder), annotations_by_id, val_loader, device,
            )
            if is_main_process():
                log.info("epoch %d val: L_simple=%.5f train_loss=%.5f",
                         epoch, metrics["L_simple"], train_loss)
                state = {
                    "state_dict": _strip_ddp(model).state_dict(),
                    "ema": ema.state_dict(),
                    "encoder": _strip_ddp(encoder).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "config_fingerprint": cfg.fingerprint(),
                    "metrics": metrics,
                    "action_types": list(action_types),
                }
                ckpt_path = run_dir / f"ckpt_epoch{epoch:04d}.pt"
                save_checkpoint(state, ckpt_path)
                is_best = keeper.add(metrics["L_simple"], ckpt_path)
                if is_best:
                    keeper.maybe_symlink_best(run_dir / "best.pt")
            if world_size > 1:
                dist.barrier()


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="src.train", description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)

    aqa = sub.add_parser("aqa", help="train the AQA scorer (TSA / STSA)")
    aqa.add_argument("--run-name", type=str, required=True)
    aqa.add_argument("--config", type=str, default=None, help="YAML Cfg override")
    aqa.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    aqa.add_argument("--seed-override", type=int, default=None)
    aqa.add_argument("--use-stsa", action="store_true",
                     help="enable the IJCV-2024 STSA extension")

    diff = sub.add_parser("diffusion", help="train the pose motion diffusion")
    diff.add_argument("--run-name", type=str, required=True)
    diff.add_argument("--config", type=str, default=None)
    diff.add_argument("--resume", type=str, default=None)
    diff.add_argument("--seed-override", type=int, default=None)
    diff.add_argument("--smpl-rest-positions", type=str, default=None,
                      help="path to data/smpl/rest_positions.npy (exact SMPL T-pose)")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = load_cfg(args)
    cfg.seed_everything()
    rank, world_size, local_rank, device = setup_ddp(cfg.train.ddp_backend)
    try:
        if args.task == "aqa":
            train_aqa(cfg, args, world_size, device)
        elif args.task == "diffusion":
            train_diffusion(cfg, args, world_size, device)
        else:
            parser.error(f"unknown task: {args.task!r}")
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()


__all__ = [
    # Utilities exported for evaluate.py / experiments.py
    "ContextEncoder", "EMA", "CheckpointKeeper",
    "setup_ddp", "cleanup_ddp", "is_main_process",
    "load_cfg", "prepare_run_dir",
]

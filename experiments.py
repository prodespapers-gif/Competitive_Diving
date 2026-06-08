"""experiments.py — Generate paper artifacts (tables + figures) from evaluation outputs.

One function per paper artifact. Each function reads CSVs or the CF
cache JSON produced by :mod:`evaluate`, formats the result as a
LaTeX-ready table (booktabs) or a vector PDF figure, and writes both
to disk under ``results/<artifact_name>.{tex,pdf,csv}``.

Workflow
--------
1. ``train`` produces checkpoints under ``experiments/<run>/``
2. ``evaluate`` produces CSVs + the CF cache under ``results/``
3. ``experiments`` formats those into the artifacts the paper includes

Nothing here re-runs models or recomputes metrics. This is the formatting
layer; if a number looks wrong, the source CSV is wrong, not this file.

Subcommands (one per artifact)
------------------------------
- ``fig_2_pipeline``           — system overview block diagram
- ``tab_1_aqa_benchmark``      — AQA Spearman / R-ℓ2 vs prior methods
- ``tab_2_counterfactual_ablation`` — CF-XAI columns (validity / sparsity /
                                       proximity / realism / completion) per
                                       search variant (Verma et al. ACM CSUR 2024)
- ``tab_3_attribution_accuracy`` — strict / step_fault / fault_only matching,
                                    top-K-per-dive headline
- ``fig_5_qualitative_cf``     — per-fault representative examples (metadata cards)
- ``tab_4_cross_dataset_finediving_plus`` — same headlines on FineDiving+
- ``fig_6_expert_study``       — preference bars + κ heatmap

Notes on figure 5
-----------------
Producing SMPL body renders would require: the SMPL model files (gated),
a differentiable renderer (heavy dep), and ethics review for synthesising
person imagery. None of those belong in the orchestration layer. The
function here produces a "qualitative examples card" showing intervention
metadata for the best-by-plausibility example per fault category. The
user supplies their own rendering script separately and overlays renders
manually in the final paper PDF.

Changes from previous version (camera-ready)
---------------------------------------------
- Issue 7 (paper-blocking): the FineDiving baseline numbers had drifted
  to the wrong values. They are now the verified IJCV'24 Table 2 values
  (Xu et al., STSA, IJCV 2024). All five baselines tabulated explicitly.
- Issue 15 (naming collision): TSA-Net (Wang et al., CVPR 2021) is
  separated from TSA (Xu et al., CVPR 2022) — they are distinct methods
  with different headline numbers in IJCV'24 Table 2 and were
  inappropriately collapsed into a single row before.
- :func:`tab_2_counterfactual_ablation` now reports the canonical
  CF-XAI columns from Verma et al. (ACM CSUR 2024):
  validity / sparsity / proximity / realism / completion. Computed
  via :func:`metrics.cf_metrics_cfxai`. The legacy
  faithfulness/minimality/plausibility/completion columns are still
  derivable from the cache via :func:`_summarize_cf_cache_legacy`.
- :func:`tab_3_attribution_accuracy` now reads ``top_<k>_per_dive``
  scope names emitted by the new :mod:`evaluate`.
- :func:`fig_5_qualitative_cf` ranks examples by the new ``pfc_edge``
  field when present, falling back to the legacy ``plausibility``
  field for older caches.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")                                       # headless
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .metrics import cf_metrics_cfxai

# IVC submission requires embedded (not subset-encoded) fonts in PDFs.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

log = logging.getLogger(__name__)


# =============================================================================
# Baseline numbers — explicit citations, never auto-fetched
# =============================================================================
#
# **Source of truth**: Xu et al., "Likert Scoring with Grade Decoupling for
# Long-term Action Assessment" (STSA), IJCV 2024, Table 2 (FineDiving).
# This is the only public table that re-evaluates ALL the major AQA methods
# on FineDiving under a unified protocol; it is what reviewers expect to
# see as the comparison baseline. Individual papers' headline numbers
# sometimes use slightly different evaluation protocols and disagree by
# 0.01-0.04 in ρ — we cite the unified-protocol numbers, not those.
#
# Issue 15 (camera-ready): TSA-Net (Wang et al. 2021) and TSA (Xu et al.
# 2022, the FineDiving paper) are DIFFERENT methods with DIFFERENT
# numbers. They were inadvertently collapsed into a single row before;
# they are now disambiguated with citations.
# -----------------------------------------------------------------------------


BASELINE_NUMBERS_FINEDIVING: Dict[str, Dict[str, Any]] = {
    # All five rows are from Xu et al., STSA, IJCV 2024, Table 2.
    # Method names match the IJCV'24 table headers exactly so the
    # paper's reviewers can cross-check.

    # Tang et al., "MUSDL", CVPR 2020.
    "MUSDL": {
        "spearman": 0.9241, "r_l2": 0.3474,
        "citation": "Tang et al., CVPR 2020",
    },
    # Yu et al., "CoRe" (Contrastive Regression), ICCV 2021.
    "CoRe": {
        "spearman": 0.9308, "r_l2": 0.3148,
        "citation": "Yu et al., ICCV 2021",
    },
    # Wang et al., "TSA-Net: Tube Self-Attention Network for Action
    # Quality Assessment", CVPR 2021. Distinct from the TSA method
    # introduced one year later by Xu et al. — see Issue 15.
    "TSA-Net (Wang)": {
        "spearman": 0.8515, "r_l2": 0.6000,
        "citation": "Wang et al., CVPR 2021",
    },
    # Xu et al., "Finediving: A Fine-grained Dataset for Procedure-aware
    # Action Quality Assessment" (introduces both FineDiving and TSA),
    # CVPR 2022. This row is what the paper builds its method on.
    "TSA": {
        "spearman": 0.9324, "r_l2": 0.3022,
        "citation": "Xu et al., CVPR 2022",
    },
    # Xu et al., STSA, IJCV 2024 (the same paper that supplies Table 2).
    "STSA": {
        "spearman": 0.9397, "r_l2": 0.2707,
        "citation": "Xu et al., IJCV 2024",
    },
}


BASELINE_NUMBERS_FINEDIVING_PLUS: Dict[str, Dict[str, Any]] = {
    # **CITATION-NEEDED for camera-ready**: the FineDiving+ baseline
    # numbers below are placeholders carried over from the previous
    # codebase iteration. Before submission, verify each value against
    # the appropriate table in Xu et al., STSA, IJCV 2024 (likely
    # Table 5 — the FineDiving+ extension) and update.
    #
    # TSA-Net (Wang) vs TSA disambiguation is applied here too,
    # consistent with the FineDiving table. If IJCV'24 only reports
    # one row for these two, drop the row that isn't reported and
    # cite a single method.
    "MUSDL": {
        "spearman": 0.7674, "r_l2": 0.7126,
        "citation": "TODO: verify in IJCV 2024 Table 5",
    },
    "CoRe": {
        "spearman": 0.8447, "r_l2": 0.4534,
        "citation": "TODO: verify in IJCV 2024 Table 5",
    },
    "TSA-Net (Wang)": {
        "spearman": 0.8612, "r_l2": 0.3854,
        "citation": "TODO: verify in IJCV 2024 Table 5",
    },
    "TSA": {
        "spearman": 0.8612, "r_l2": 0.3854,
        "citation": "TODO: verify in IJCV 2024 Table 5 (may share row with TSA-Net)",
    },
    "STSA": {
        "spearman": 0.8835, "r_l2": 0.3526,
        "citation": "TODO: verify in IJCV 2024 Table 5",
    },
}


def load_baselines_yaml(path: Optional[Path]) -> Optional[Dict[str, Dict[str, float]]]:
    """Override the built-in baselines with values from a YAML file.

    Useful for the camera-ready pass: once the FineDiving+ numbers are
    verified against the published table, drop them in a YAML file and
    pass via ``--baselines-yaml`` rather than re-editing this module.
    """
    if path is None:
        return None
    import yaml
    return yaml.safe_load(Path(path).read_text())


# =============================================================================
# LaTeX table helpers
# =============================================================================


def _format_number(value: Any, precision: int = 4) -> str:
    """Format a numeric value to ``precision`` decimal places; non-numeric returned as-is."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:  # NaN
        return "--"
    return f"{f:.{precision}f}"


_LEADING_NUMBER_RE = __import__("re").compile(r"-?\d+\.?\d*")


def _select_best_rows(
    rows: List[List[str]], col_idx: int, higher_is_better: bool,
) -> Optional[int]:
    """Return the row index whose cell at ``col_idx`` is best, or None if no parse.

    Cells often carry LaTeX decorations (e.g. ``"0.9112 \\tiny[0.89,0.92]"``)
    so we extract the first numeric token rather than ``float()``-ing the
    whole cell.
    """
    parsed: List[Tuple[int, float]] = []
    for i, row in enumerate(rows):
        m = _LEADING_NUMBER_RE.search(str(row[col_idx]))
        if m is None:
            continue
        try:
            parsed.append((i, float(m.group(0))))
        except ValueError:
            continue
    if not parsed:
        return None
    return max(parsed, key=lambda kv: kv[1] if higher_is_better else -kv[1])[0]


def latex_table(
    *,
    rows: List[List[str]],
    header: List[str],
    caption: str,
    label: str,
    bold_best: Optional[Dict[int, bool]] = None,
    col_spec: Optional[str] = None,
    notes: Optional[str] = None,
) -> str:
    """Render a booktabs-style LaTeX table to a string.

    Parameters
    ----------
    rows : list of cell-string lists (first column treated as a row label)
    header : column titles
    caption / label : paper standards
    bold_best : ``{col_idx: higher_is_better}`` — bold the best cell in each
        listed column. ``higher_is_better=True`` for ρ, F1, precision;
        ``False`` for R-ℓ2, minimality norm, etc.
    col_spec : LaTeX tabular column spec; default ``"l" + "c"*(n-1)``
    notes : optional footer text rendered with ``\\footnotesize``
    """
    n_cols = len(header)
    if col_spec is None:
        col_spec = "l" + "c" * (n_cols - 1)
    # Bold best cells (mutates a copy so the caller's list survives).
    rows = [row[:] for row in rows]
    for col_idx, higher_is_better in (bold_best or {}).items():
        best_idx = _select_best_rows(rows, col_idx, higher_is_better)
        if best_idx is not None:
            rows[best_idx][col_idx] = r"\textbf{" + rows[best_idx][col_idx] + "}"
    out: List[str] = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    out.extend(" & ".join(row) + r" \\" for row in rows)
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    if notes:
        out.append(r"\\[2pt] \footnotesize " + notes)
    out.append(r"\end{table}")
    return "\n".join(out) + "\n"


def write_tex(content: str, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    log.info("wrote LaTeX table to %s", output)


def write_pdf(fig: plt.Figure, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    log.info("wrote figure to %s", output)


# =============================================================================
# CSV parsing helpers
# =============================================================================


def _read_eval_csv(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Read an evaluate.py CSV into ``{(stratum, metric): {col: value}}``."""
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["stratum"], row["metric"])] = row
    return out


def _read_expert_csv(path: Path) -> Dict[str, float]:
    """Read the expert-study CSV into ``{metric_name: value}``."""
    out: Dict[str, float] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["metric"]] = float(row["value"])
            except (TypeError, ValueError):
                out[row["metric"]] = float("nan")
    return out


def _read_attr_csv(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Read an attribution CSV into ``{(scope, metric): row}``."""
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["scope"], row["metric"])] = row
    return out


# =============================================================================
# Figure 2: Pipeline diagram
# =============================================================================


def _box(
    ax: plt.Axes, x: float, y: float, w: float, h: float,
    text: str, color: str = "#E8F0FE", edge: str = "black",
) -> None:
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def _arrow(
    ax: plt.Axes, p1: Tuple[float, float], p2: Tuple[float, float],
    label: Optional[str] = None, dashed: bool = False,
) -> None:
    style = "dashed" if dashed else "solid"
    arr = FancyArrowPatch(
        p1, p2, arrowstyle="->", mutation_scale=12,
        linewidth=0.8, linestyle=style, color="black",
    )
    ax.add_patch(arr)
    if label:
        ax.text(
            (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.015, label,
            fontsize=7, ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=1),
        )


def fig_2_pipeline(output_dir: Path) -> None:
    """Figure 2: system overview block diagram."""
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 0.6); ax.set_aspect("equal")
    ax.axis("off")

    # Top row: AQA pipeline (left to right)
    _box(ax, 0.02, 0.42, 0.10, 0.08, "Dive\nvideo",   color="#F5F5F5")
    _box(ax, 0.16, 0.42, 0.12, 0.08, "I3D\nbackbone", color="#E8F0FE")
    _box(ax, 0.32, 0.42, 0.14, 0.08, "Procedure\nsegmentation", color="#E8F0FE")
    _box(ax, 0.50, 0.42, 0.16, 0.08, "Procedure-aware\ncross-attention", color="#E8F0FE")
    _box(ax, 0.70, 0.42, 0.12, 0.08, "Contrastive\nregressor", color="#E8F0FE")
    _box(ax, 0.86, 0.42, 0.10, 0.08, "Score\n$\\hat{s}$", color="#FFF4E5")

    # Arrows between top boxes
    for (x1, x2) in [(0.12, 0.16), (0.28, 0.32), (0.46, 0.50), (0.66, 0.70), (0.82, 0.86)]:
        _arrow(ax, (x1, 0.46), (x2, 0.46))

    # Exemplar input — comes in from below to cross-attention
    _box(ax, 0.46, 0.26, 0.12, 0.08, "Exemplar\nfeatures", color="#F5F5F5")
    _arrow(ax, (0.55, 0.34), (0.55, 0.42))

    # Bottom row: CF pipeline
    _box(ax, 0.02, 0.08, 0.12, 0.08, "SMPL\npose tokens", color="#F5F5F5")
    _box(ax, 0.18, 0.08, 0.18, 0.08, "EDGE-style\nmotion diffusion", color="#E2F1E2")
    _box(ax, 0.40, 0.08, 0.16, 0.08, "Half-point\nCF search", color="#E2F1E2")
    _box(ax, 0.60, 0.08, 0.12, 0.08, "Fault\nattribution", color="#FFF4E5")
    _box(ax, 0.76, 0.08, 0.20, 0.08, "Per-step decomposition\n$\\{(\\Delta_j, F_j, J_j)\\}$",
         color="#FFE5E5")

    for (x1, x2) in [(0.14, 0.18), (0.36, 0.40), (0.56, 0.60), (0.72, 0.76)]:
        _arrow(ax, (x1, 0.12), (x2, 0.12))

    # Cross-pipeline arrow: AQA score → CF search (the score_fn dependency)
    _arrow(ax, (0.76, 0.42), (0.48, 0.16), label=r"$s_\theta$", dashed=True)

    # Section labels
    ax.text(0.02, 0.55, "Action Quality Assessment (TSA backbone)",
            fontsize=9, fontweight="bold")
    ax.text(0.02, 0.21, "Counterfactual decomposition (ours)",
            fontsize=9, fontweight="bold")

    write_pdf(fig, Path(output_dir) / "fig_2_pipeline.pdf")


# =============================================================================
# Table 1: AQA benchmark
# =============================================================================


def _ours_aqa_overall(path: Path) -> Tuple[float, float, Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Extract overall Spearman + R-ℓ2 (with bootstrap CI) from evaluate aqa CSV."""
    data = _read_eval_csv(path)

    def _get(metric: str) -> Tuple[float, Optional[Tuple[float, float]]]:
        row = data.get(("overall", metric))
        if row is None:
            raise ValueError(f"no 'overall' row for {metric!r} in {path}")
        point = float(row["point"])
        try:
            ci = (float(row["ci_low"]), float(row["ci_high"]))
        except (ValueError, KeyError):
            ci = None
        return point, ci

    rho, rho_ci = _get("spearman")
    rl2, rl2_ci = _get("r_l2")
    return rho, rl2, rho_ci, rl2_ci


def tab_1_aqa_benchmark(
    ours_csv: Path, output_dir: Path,
    *, baselines: Optional[Dict[str, Dict[str, Any]]] = None,
    label: str = "tab:aqa_benchmark", dataset_name: str = "FineDiving",
    file_stem: str = "tab_1_aqa_benchmark",
) -> None:
    """Table 1: AQA Spearman + R-ℓ2 vs prior methods on a given dataset.

    Baselines come from a unified-protocol source (Xu et al., IJCV 2024,
    Table 2 for FineDiving). TSA-Net (Wang) and TSA are separate rows
    per Issue 15.
    """
    baselines = baselines or BASELINE_NUMBERS_FINEDIVING
    rho, rl2, rho_ci, rl2_ci = _ours_aqa_overall(ours_csv)

    rows: List[List[str]] = []
    for name, vals in baselines.items():
        rows.append([name,
                     _format_number(vals["spearman"]),
                     _format_number(vals["r_l2"])])

    def _fmt_with_ci(val: float, ci: Optional[Tuple[float, float]]) -> str:
        cell = _format_number(val)
        if ci:
            cell += rf" \tiny[{_format_number(ci[0])},{_format_number(ci[1])}]"
        return cell

    rows.append(["Ours (TSA backbone)",
                 _fmt_with_ci(rho, rho_ci),
                 _fmt_with_ci(rl2, rl2_ci)])

    table = latex_table(
        rows=rows,
        header=["Method", r"$\rho \uparrow$", r"R-$\ell_2$ $\downarrow$"],
        caption=(f"Action Quality Assessment on {dataset_name}. "
                 r"Spearman's $\rho$ (higher is better) and the "
                 r"$\ell_2$-relative score error R-$\ell_2$ (lower is better). "
                 r"Baselines from Xu et al., IJCV 2024 (Table 2), under their "
                 r"unified evaluation protocol. "
                 r"95\% bootstrap CIs reported for our method."),
        label=label,
        bold_best={1: True, 2: False},
        notes=(r"\textbf{TSA-Net} (Wang et al., CVPR 2021) and \textbf{TSA} "
               r"(Xu et al., CVPR 2022 — the paper that introduced FineDiving) "
               r"are distinct methods."),
    )
    write_tex(table, Path(output_dir) / f"{file_stem}.tex")
    # Also dump a flat CSV for reproducibility — includes citation column.
    csv_out = Path(output_dir) / f"{file_stem}.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "spearman", "r_l2", "citation"])
        for name, vals in baselines.items():
            w.writerow([name, vals["spearman"], vals["r_l2"],
                        vals.get("citation", "")])
        w.writerow(["Ours", rho, rl2, "this work"])


# =============================================================================
# Table 2: Counterfactual ablation — CF-XAI canonical columns (Issue 12)
# =============================================================================


def _summarize_cf_cache_cfxai(
    path: Path, pfc_primary: str = "edge",
) -> Dict[str, float]:
    """Compute CF-XAI canonical columns (Verma et al. ACM CSUR 2024)
    from one CF cache.

    The cache layout is the one written by
    :func:`evaluate._decomposition_to_dict`: each dive entry has a
    ``records`` list with per-peel diagnostics (actual_uplift,
    claimed_uplift, minimality_norm, joint_groups, completed flag,
    plus the new ``pfc_edge`` / ``pfc_lateral_accel`` fields when
    available).

    Args:
        path: cache JSON written by ``evaluate cf``.
        pfc_primary: ``"edge"`` (recommended) or ``"lateral_accel"``;
            picks which PFC variant feeds the ``realism`` column.

    Returns:
        Dict with the five canonical CF-XAI columns plus ``n``. The
        five columns are passed straight through from
        :func:`metrics.cf_metrics_cfxai`.
    """
    with Path(path).open() as f:
        cache = json.load(f)

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
        for r in entry["records"]:
            actual.append(float(r.get("actual_uplift", 0.0)))
            claimed.append(float(r.get("claimed_uplift", 0.0)))
            norms.append(float(r.get("minimality_norm", 0.0)))
            n_joints_changed.append(len(r.get("joint_groups", [])))
            # Realism: the chosen PFC variant if present in the cache,
            # otherwise the legacy ``plausibility`` field (which carries
            # whatever plausibility_fn was passed at search time).
            if pfc_key in r:
                pfc_scores.append(float(r[pfc_key]))
            else:
                pfc_scores.append(float(r.get("plausibility", 0.0)))
        completed_flags.append(bool(entry.get("completed", False)))

    if not actual:
        # No usable records; return NaN-filled structure so downstream
        # formatting doesn't crash.
        return {"validity": float("nan"), "sparsity": float("nan"),
                "proximity": float("nan"), "realism": float("nan"),
                "completion": float("nan"), "n": 0}

    return cf_metrics_cfxai(
        actual_uplifts=actual, claimed_uplifts=claimed,
        pose_l2_norms=norms,
        n_joint_groups_changed=n_joints_changed,
        pfc_scores=pfc_scores,
        completed_flags=completed_flags,
    )


def _summarize_cf_cache_legacy(path: Path) -> Dict[str, float]:
    """Legacy aggregator — faithfulness / minimality / plausibility.

    Kept for backwards compatibility with the previous ablation table
    format. New camera-ready code should use :func:`_summarize_cf_cache_cfxai`
    for the canonical Verma-et-al. CF-XAI columns.
    """
    with Path(path).open() as f:
        cache = json.load(f)
    deltas: List[float] = []
    minimalities: List[float] = []
    plausibilities: List[float] = []
    actual_uplifts: List[float] = []
    claimed_uplifts: List[float] = []
    faithfulness_flags: List[bool] = []
    completed_count = 0
    total = 0
    for dive_id, entry in cache.items():
        if "error" in entry or "records" not in entry:
            continue
        total += 1
        if entry.get("completed"):
            completed_count += 1
        for r in entry["records"]:
            deltas.append(r["delta_score"])
            minimalities.append(r["minimality_norm"])
            plausibilities.append(r["plausibility"])
            actual_uplifts.append(r["actual_uplift"])
            claimed_uplifts.append(r["claimed_uplift"])
            faithfulness_flags.append(bool(r["faithfulness_passed"]))
    if total == 0 or not deltas:
        return {"n_dives": 0, "n_peels": 0}
    return {
        "n_dives": total,
        "n_peels": len(deltas),
        "completion_rate": completed_count / total,
        "mean_minimality": float(np.mean(minimalities)),
        "mean_plausibility": float(np.mean(plausibilities)),
        "faithfulness_rate": float(np.mean(faithfulness_flags)),
        "mean_actual_uplift": float(np.mean(actual_uplifts)),
        "mean_claimed_uplift": float(np.mean(claimed_uplifts)),
    }


def tab_2_counterfactual_ablation(
    variant_paths: Mapping[str, Path], output_dir: Path,
    *,
    file_stem: str = "tab_2_counterfactual_ablation",
    pfc_primary: str = "edge",
    legacy_columns: bool = False,
) -> None:
    """Table 2: CF search ablation.

    ``variant_paths`` is ``{variant_label: cf_cache_json_path}``. Recommended
    rows:
        "Random baseline", "Greedy",
        "Greedy + score guidance", "Beam (w=4)", "Beam + score guidance".

    Camera-ready format uses the canonical CF-XAI vocabulary from
    Verma et al., ACM CSUR 2024:
        validity   — fraction of CFs that achieved their target uplift
                      within tolerance (↑)
        sparsity   — mean number of joint groups changed (↓ is sparser)
        proximity  — mean ℓ2 of the masked perturbation (↓ is closer)
        realism    — mean physical foot-contact score (↓ is more in-distribution).
                      Uses the EDGE-style PFC by default; pass
                      ``pfc_primary="lateral_accel"`` for the legacy variant.
        completion — fraction of dives whose target uplift was fully
                      decomposed (this paper's contribution).

    Args:
        pfc_primary: ``"edge"`` (default, recommended) or
            ``"lateral_accel"``. Determines the realism source field.
        legacy_columns: if True, emit the previous-version columns
            (faithfulness, minimality, plausibility, completion, N)
            instead of the CF-XAI canonical ones. The cfg ablation row
            in the paper uses this to reproduce the pre-camera-ready
            numbers.
    """
    if legacy_columns:
        _tab_2_legacy(variant_paths, output_dir, file_stem=file_stem)
        return

    rows: List[List[str]] = []
    for label, path in variant_paths.items():
        stats = _summarize_cf_cache_cfxai(Path(path), pfc_primary=pfc_primary)
        if stats.get("n", 0) == 0:
            rows.append([label, "--", "--", "--", "--", "--", "--"])
            continue
        rows.append([
            label,
            _format_number(stats["validity"], precision=3),
            _format_number(stats["sparsity"], precision=2),
            _format_number(stats["proximity"], precision=3),
            _format_number(stats["realism"], precision=4),
            _format_number(stats["completion"], precision=3),
            f"{int(stats['n'])}",
        ])

    pfc_label = "PFC-EDGE" if pfc_primary == "edge" else "PFC-lat.acc."
    table = latex_table(
        rows=rows,
        header=["Variant",
                r"Val.\,$\uparrow$",        # validity
                r"Spars.\,$\downarrow$",    # sparsity
                r"Prox.\,$\downarrow$",     # proximity
                rf"Real.\,({pfc_label})\,$\downarrow$",  # realism
                r"Compl.\,$\uparrow$",      # completion
                r"$N$"],
        caption=(
            "Counterfactual ablation under the CF-XAI canonical "
            "vocabulary (Verma et al., ACM CSUR 2024). "
            "\\emph{Validity}: fraction of CFs reaching their target "
            "uplift within tolerance; \\emph{Sparsity}: mean number "
            "of joint groups perturbed; \\emph{Proximity}: mean "
            "$\\ell_2$ of the masked perturbation; \\emph{Realism}: "
            f"{pfc_label} foot-contact score (lower is more in-distribution); "
            "\\emph{Completion}: fraction of dives whose target uplift "
            "was fully decomposed (this paper's contribution)."
        ),
        label="tab:cf_ablation",
        bold_best={1: True, 2: False, 3: False, 4: False, 5: True},
    )
    write_tex(table, Path(output_dir) / f"{file_stem}.tex")


def _tab_2_legacy(
    variant_paths: Mapping[str, Path], output_dir: Path,
    *, file_stem: str,
) -> None:
    """Pre-camera-ready ablation table — kept for reproducibility."""
    rows: List[List[str]] = []
    for label, path in variant_paths.items():
        stats = _summarize_cf_cache_legacy(Path(path))
        if stats["n_dives"] == 0:
            rows.append([label, "--", "--", "--", "--", "--"])
            continue
        rows.append([
            label,
            _format_number(stats["faithfulness_rate"], precision=3),
            _format_number(stats["mean_minimality"], precision=3),
            _format_number(stats["mean_plausibility"], precision=3),
            _format_number(stats["completion_rate"], precision=3),
            f"{int(stats['n_dives'])}",
        ])

    table = latex_table(
        rows=rows,
        header=["Variant", r"Faithf.\,$\uparrow$", r"Minim.\,$\downarrow$",
                r"Plaus.\,$\downarrow$", r"Compl.\,$\uparrow$", r"$N$"],
        caption=("Counterfactual ablation (legacy format). "
                 "Faithfulness is the rate at which "
                 "$|s(x_{cf}) - s(x_q) - \\Delta| < \\tau$; minimality is "
                 "the mean $\\ell_2$ norm of the masked perturbation; "
                 "plausibility is the diffusion-prior score (lower is more "
                 "in-distribution); completion is the fraction of dives whose "
                 "target uplift was fully explained."),
        label="tab:cf_ablation_legacy",
        bold_best={1: True, 2: False, 3: False, 4: True},
    )
    write_tex(table, Path(output_dir) / f"{file_stem}_legacy.tex")


# =============================================================================
# Table 3: Attribution accuracy
# =============================================================================


def tab_3_attribution_accuracy(
    paths_by_match: Mapping[str, Path], output_dir: Path,
    *, file_stem: str = "tab_3_attribution_accuracy",
    top_k_values: Sequence[int] = (1, 3),
) -> None:
    """Table 3: attribution accuracy under each match criterion.

    ``paths_by_match`` is ``{"strict": csv_path, "step_fault": csv_path,
    "fault_only": csv_path}`` — output CSVs from ``evaluate attribution
    --match X``.

    Camera-ready: reads the ``top_<k>_per_dive`` scope names emitted by
    the new :mod:`evaluate` (the headline; ``top_<k>_union`` is also
    in the CSV but isn't shown here). Falls back to legacy ``topk@K``
    if the per-dive scope isn't present, for compatibility with older
    CSVs produced before the camera-ready evaluate.py.
    """
    rows: List[List[str]] = []
    header = [r"Match", r"P", r"R", r"F1"]
    for k in top_k_values:
        header.append(rf"F1@{k}")
    header.append(r"Expl.")

    for match_name, path in paths_by_match.items():
        data = _read_attr_csv(Path(path))

        def get(scope: str, metric: str, fallback_scope: Optional[str] = None) -> str:
            row = data.get((scope, metric))
            if row is None and fallback_scope is not None:
                row = data.get((fallback_scope, metric))
            return _format_number(row["value"], precision=3) if row else "--"

        row = [
            match_name.replace("_", "+"),
            get("overall", "precision"),
            get("overall", "recall"),
            get("overall", "f1"),
        ]
        for k in top_k_values:
            row.append(get(
                f"top_{k}_per_dive", "f1",
                fallback_scope=f"topk@{k}",   # legacy CSV compatibility
            ))
        row.append(get("overall", "fully_explained_rate"))
        rows.append(row)

    # Bold best across all P/R/F1/F1@K/Expl columns.
    bold_best = {i: True for i in range(1, 1 + 3 + len(top_k_values) + 1)}

    table = latex_table(
        rows=rows,
        header=header,
        caption=("Attribution accuracy against FineDiving-HF expert labels "
                 "under three match criteria: \\emph{strict} (step, joint set, "
                 "and fault category all match), \\emph{step+fault} (joint "
                 "set ignored), and \\emph{fault-only} (only the fault "
                 "category must match). F1@$k$ is the per-dive top-$k$ "
                 "F1 (the recommended top-$k$ aggregation). Last column "
                 "is the fraction of dives whose target uplift was fully "
                 "decomposed."),
        label="tab:attr_accuracy",
        bold_best=bold_best,
    )
    write_tex(table, Path(output_dir) / f"{file_stem}.tex")


# =============================================================================
# Figure 5: Qualitative CF examples (metadata cards)
# =============================================================================


def _pick_representative_per_fault(
    cache: Dict[str, Dict[str, Any]], faults: Sequence[str],
    *, rank_field: str = "pfc_edge",
) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """Best example per fault category, ranked by ``rank_field`` (lower is better).

    Tries ``rank_field`` first (the new EDGE-style PFC), falls back to
    the legacy ``plausibility`` field when ``rank_field`` is absent
    from a record (e.g. old caches generated before the camera-ready
    PFC switch).
    """
    best: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
    for dive_id, entry in cache.items():
        if "error" in entry or not entry.get("records"):
            continue
        for r in entry["records"]:
            f = r["fault"]
            if f not in faults:
                continue
            if rank_field in r:
                score = float(r[rank_field])
            else:
                # Legacy cache without per-record PFC; use plausibility.
                score = float(r.get("plausibility", float("inf")))
            if f not in best or score < best[f][0]:
                best[f] = (score, dive_id, r)
    return {f: (dive_id, rec) for f, (_, dive_id, rec) in best.items()}


def fig_5_qualitative_cf(
    cf_cache: Path, output_dir: Path,
    *, faults: Optional[Sequence[str]] = None,
    file_stem: str = "fig_5_qualitative_cf",
    rank_field: str = "pfc_edge",
) -> None:
    """Figure 5: qualitative CF examples — one card per fault category.

    Each card shows the intervention metadata (step, joints, Δ, plausibility)
    in a fixed layout. The user is expected to overlay actual SMPL renders
    in the camera-ready PDF; this function lays out the figure skeleton and
    text annotations.

    Args:
        rank_field: which cache field to use for picking the best example
            per fault (``"pfc_edge"`` by default, the EDGE-style PFC from
            the camera-ready :func:`evaluate.cmd_cf`). Falls back to
            ``"plausibility"`` per record when the chosen field is absent.
    """
    if faults is None:
        faults = ("bent_knees", "separated_feet", "poor_body_line",
                  "over_rotation", "large_splash")
    with Path(cf_cache).open() as f:
        cache = json.load(f)
    examples = _pick_representative_per_fault(cache, faults, rank_field=rank_field)

    n = len(faults)
    fig, axes = plt.subplots(1, n, figsize=(min(7.0, 1.6 * n), 2.4))
    if n == 1:
        axes = [axes]
    for ax, fault in zip(axes, faults):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        # Title bar.
        ax.add_patch(FancyBboxPatch(
            (0.02, 0.84), 0.96, 0.14,
            boxstyle="round,pad=0.02", linewidth=0.8,
            edgecolor="black", facecolor="#FFF4E5",
        ))
        ax.text(0.5, 0.91, fault.replace("_", " "),
                ha="center", va="center", fontsize=9, fontweight="bold")
        # Body: schematic placeholder for the SMPL render.
        ax.add_patch(FancyBboxPatch(
            (0.10, 0.32), 0.80, 0.46,
            boxstyle="round,pad=0.02", linewidth=0.8,
            edgecolor="gray", facecolor="#FAFAFA",
        ))
        ax.text(0.5, 0.55, "[SMPL render\nplaceholder]",
                ha="center", va="center", fontsize=7, color="gray",
                style="italic")
        # Metadata footer.
        if fault in examples:
            dive_id, rec = examples[fault]
            joints = ", ".join(rec["joint_groups"]) or "(transition)"
            ax.text(0.5, 0.24, f"step {rec['step']} \u00b7 {joints}",
                    ha="center", va="center", fontsize=7)
            ax.text(0.5, 0.14, rf"$\Delta=$ {rec['delta_score']:.2f}",
                    ha="center", va="center", fontsize=7)
            ax.text(0.5, 0.05, dive_id.split("_")[-1],
                    ha="center", va="center", fontsize=6, color="gray")
        else:
            ax.text(0.5, 0.14, "no example", ha="center", va="center",
                    fontsize=7, color="gray", style="italic")

    fig.suptitle("Qualitative counterfactual examples (one per fault category)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    write_pdf(fig, Path(output_dir) / f"{file_stem}.pdf")


# =============================================================================
# Table 4: cross-dataset FineDiving+
# =============================================================================


def tab_4_cross_dataset_finediving_plus(
    aqa_csv: Path, attr_csv: Path, output_dir: Path,
    *, file_stem: str = "tab_4_cross_dataset_finediving_plus",
    baselines: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Table 4: AQA + attribution headlines on FineDiving+.

    Combines the AQA Table 1 headline numbers and the attribution Table 3
    overall F1 into a single side-by-side panel. Baselines are the
    FineDiving+ values from :data:`BASELINE_NUMBERS_FINEDIVING_PLUS`
    (still TODO-flagged for camera-ready verification — see that dict's
    comment).
    """
    baselines = baselines or BASELINE_NUMBERS_FINEDIVING_PLUS
    rho, rl2, _, _ = _ours_aqa_overall(Path(aqa_csv))
    attr = _read_attr_csv(Path(attr_csv))

    def get(scope: str, metric: str) -> str:
        row = attr.get((scope, metric))
        return _format_number(row["value"], precision=3) if row else "--"

    rows = []
    for name, vals in baselines.items():
        rows.append([name,
                     _format_number(vals["spearman"]),
                     _format_number(vals["r_l2"]),
                     "--", "--", "--"])
    rows.append([
        "Ours",
        _format_number(rho), _format_number(rl2),
        get("overall", "precision"),
        get("overall", "recall"),
        get("overall", "f1"),
    ])

    table = latex_table(
        rows=rows,
        header=["Method",
                r"$\rho\uparrow$", r"R-$\ell_2$\,$\downarrow$",
                r"Attr.\,P", r"Attr.\,R", r"Attr.\,F1"],
        caption=("Cross-dataset evaluation on FineDiving+. AQA columns are "
                 "directly comparable to prior work; attribution columns "
                 "use the strict match criterion on the held-out 50-dive "
                 "FineDiving-HF\\textsuperscript{+} subset."),
        label="tab:cross_dataset",
        col_spec="lcccccc",
        bold_best={1: True, 2: False, 3: True, 4: True, 5: True},
        notes=(r"TSA-Net (Wang) and TSA are distinct methods (see Tab.~1 caption)."),
    )
    write_tex(table, Path(output_dir) / f"{file_stem}.tex")


# =============================================================================
# Figure 6: Expert study
# =============================================================================


def fig_6_expert_study(
    expert_csv: Path, output_dir: Path,
    *, file_stem: str = "fig_6_expert_study",
) -> None:
    """Figure 6: preference bars + inter-rater κ heatmap."""
    metrics = _read_expert_csv(Path(expert_csv))

    # Extract preference rates by CF id.
    cf_rates: Dict[str, float] = {}
    for k, v in metrics.items():
        if k.startswith("majority_preferred_rate::"):
            cf_rates[k.split("::", 1)[1]] = v

    # Extract pairwise κ values into a symmetric matrix.
    rater_set: set = set()
    pair_kappa: Dict[Tuple[str, str], float] = {}
    for k, v in metrics.items():
        if k.startswith("kappa::"):
            _, r1, r2 = k.split("::")
            rater_set.update({r1, r2})
            pair_kappa[(r1, r2)] = v
    raters = sorted(rater_set)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6),
                                   gridspec_kw={"width_ratios": [1.0, 0.9]})

    # (a) Preference bars
    if cf_rates:
        names = sorted(cf_rates)
        values = [cf_rates[n] for n in names]
        bars = ax1.bar(names, values,
                       color=["#4C72B0" if "model" in n else "#999999" for n in names],
                       edgecolor="black", linewidth=0.6)
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("Majority-preferred rate")
        ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.6)
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax1.set_title("(a) CF variant preference")
        for tl in ax1.get_xticklabels():
            tl.set_rotation(15); tl.set_ha("right")
    else:
        ax1.text(0.5, 0.5, "(no preference data)", ha="center", va="center",
                 transform=ax1.transAxes, color="gray", style="italic")

    # (b) κ heatmap
    if raters:
        n = len(raters)
        M = np.full((n, n), np.nan)
        for i, r1 in enumerate(raters):
            for j, r2 in enumerate(raters):
                if i == j:
                    M[i, j] = 1.0
                elif (r1, r2) in pair_kappa:
                    M[i, j] = M[j, i] = pair_kappa[(r1, r2)]
                elif (r2, r1) in pair_kappa:
                    M[i, j] = M[j, i] = pair_kappa[(r2, r1)]
        im = ax2.imshow(M, vmin=0, vmax=1, cmap="Blues")
        ax2.set_xticks(range(n)); ax2.set_xticklabels(raters, rotation=30, ha="right")
        ax2.set_yticks(range(n)); ax2.set_yticklabels(raters)
        for i in range(n):
            for j in range(n):
                if not np.isnan(M[i, j]):
                    ax2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                             color=("white" if M[i, j] > 0.55 else "black"),
                             fontsize=7)
        ax2.set_title(r"(b) Inter-rater $\kappa$")
        plt.colorbar(im, ax=ax2, shrink=0.7)
    else:
        ax2.text(0.5, 0.5, "(no κ data)", ha="center", va="center",
                 transform=ax2.transAxes, color="gray", style="italic")

    fig.tight_layout()
    write_pdf(fig, Path(output_dir) / f"{file_stem}.pdf")


# =============================================================================
# CLI
# =============================================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="src.experiments", description=__doc__)
    p.add_argument("--output-dir", type=str, default="results/paper_artifacts")
    p.add_argument("--baselines-yaml", type=str, default=None,
                   help="override BASELINE_NUMBERS_FINEDIVING from YAML")
    sub = p.add_subparsers(dest="artifact", required=True)

    sub.add_parser("fig_2_pipeline", help="system pipeline diagram")

    t1 = sub.add_parser("tab_1_aqa_benchmark", help="AQA Spearman + R-ℓ2 vs baselines")
    t1.add_argument("--ours-csv", type=str, required=True)
    t1.add_argument("--dataset-name", type=str, default="FineDiving")

    t2 = sub.add_parser("tab_2_counterfactual_ablation",
                        help="CF search ablation (variant_label=cache_path pairs)")
    t2.add_argument("--variant", action="append", required=True,
                    metavar="LABEL=PATH",
                    help="repeat: --variant 'Greedy=path/to/cache.json'")
    t2.add_argument("--pfc-primary", type=str, default="edge",
                    choices=("edge", "lateral_accel"),
                    help="which PFC variant feeds the 'realism' column "
                         "(default: edge — recommended)")
    t2.add_argument("--legacy-columns", action="store_true",
                    help="emit the pre-camera-ready table format instead "
                         "of the CF-XAI canonical columns")

    t3 = sub.add_parser("tab_3_attribution_accuracy",
                        help="attribution accuracy table")
    t3.add_argument("--strict-csv", type=str, required=True)
    t3.add_argument("--step-fault-csv", type=str, required=True)
    t3.add_argument("--fault-only-csv", type=str, required=True)
    t3.add_argument("--top-k", type=int, nargs="+", default=[1, 3],
                    help="K values for the F1@K columns (must be in the "
                         "attribution CSV; default: 1 3)")

    f5 = sub.add_parser("fig_5_qualitative_cf", help="qualitative CF cards")
    f5.add_argument("--cf-cache", type=str, required=True)
    f5.add_argument("--rank-field", type=str, default="pfc_edge",
                    help="cache field used to pick the best example per "
                         "fault (default: pfc_edge; falls back to "
                         "plausibility for older caches)")

    t4 = sub.add_parser("tab_4_cross_dataset_finediving_plus",
                        help="combined AQA + attribution on FineDiving+")
    t4.add_argument("--aqa-csv", type=str, required=True,
                    help="output of `evaluate aqa` on FineDiving+ test split")
    t4.add_argument("--attr-csv", type=str, required=True,
                    help="output of `evaluate attribution` on FineDiving-HF+")

    f6 = sub.add_parser("fig_6_expert_study", help="preference bars + κ heatmap")
    f6.add_argument("--expert-csv", type=str, required=True)

    return p


def _parse_variants(items: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"--variant must be LABEL=PATH, got {item!r}",
            )
        label, _, path = item.partition("=")
        out[label.strip()] = Path(path.strip())
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    baselines = load_baselines_yaml(Path(args.baselines_yaml)) if args.baselines_yaml else None

    if args.artifact == "fig_2_pipeline":
        fig_2_pipeline(out)
    elif args.artifact == "tab_1_aqa_benchmark":
        tab_1_aqa_benchmark(
            Path(args.ours_csv), out,
            baselines=baselines or BASELINE_NUMBERS_FINEDIVING,
            dataset_name=args.dataset_name,
        )
    elif args.artifact == "tab_2_counterfactual_ablation":
        tab_2_counterfactual_ablation(
            _parse_variants(args.variant), out,
            pfc_primary=args.pfc_primary,
            legacy_columns=args.legacy_columns,
        )
    elif args.artifact == "tab_3_attribution_accuracy":
        tab_3_attribution_accuracy({
            "strict":     Path(args.strict_csv),
            "step_fault": Path(args.step_fault_csv),
            "fault_only": Path(args.fault_only_csv),
        }, out, top_k_values=tuple(args.top_k))
    elif args.artifact == "fig_5_qualitative_cf":
        fig_5_qualitative_cf(
            Path(args.cf_cache), out,
            rank_field=args.rank_field,
        )
    elif args.artifact == "tab_4_cross_dataset_finediving_plus":
        tab_4_cross_dataset_finediving_plus(
            Path(args.aqa_csv), Path(args.attr_csv), out,
            baselines=baselines or BASELINE_NUMBERS_FINEDIVING_PLUS,
        )
    elif args.artifact == "fig_6_expert_study":
        fig_6_expert_study(Path(args.expert_csv), out)
    else:
        parser.error(f"unknown artifact {args.artifact!r}")


if __name__ == "__main__":
    main()


__all__ = [
    "BASELINE_NUMBERS_FINEDIVING", "BASELINE_NUMBERS_FINEDIVING_PLUS",
    "latex_table", "write_tex", "write_pdf",
    "fig_2_pipeline", "tab_1_aqa_benchmark", "tab_2_counterfactual_ablation",
    "tab_3_attribution_accuracy", "fig_5_qualitative_cf",
    "tab_4_cross_dataset_finediving_plus", "fig_6_expert_study",
]

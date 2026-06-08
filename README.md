# What Cost Half a Point?

### Counterfactual Action Quality Assessment for Competitive Diving

**Paper:** Accepted at *Image and Vision Computing* (Elsevier)

**Authors:** Aihui Shang, Anping Chen\*, FIRSTNAME Y  
*School of Physical Education, Shanxi University, Taiyuan, China*  
\* Corresponding author

---

## Overview

This repository contains the code for a counterfactual framework that decomposes a competitive dive's execution score gap into an ordered list of **half-point fault attributions**. Each attribution is grounded by a physically plausible counterfactual pose edit generated via diffusion-based masked inpainting, and mapped to a named fault category from a nine-label FINA taxonomy.

The framework comprises four stages:

| Stage | Component | Module |
|-------|-----------|--------|
| A | Frozen AQA Oracle (TSA/STSA) | `aqa_model.py` |
| B | Motion Diffusion Prior (EDGE lineage) | `diffusion.py` |
| C | Half-Point Counterfactual Search | `counterfactual.py` |
| D | Rule-Based Fault Attribution | `attribution.py` |

We also release **FineDiving-HF**, the first expert-annotated dataset providing half-point fault labels for 200 competitive dives across nine FINA fault categories (Cohen's κ = 0.711).

---

## Results

| Metric | Value |
|--------|-------|
| Spearman ρ (AQA oracle) | 0.9415 |
| Relative ℓ₂ (AQA oracle) | 0.2680 |
| CF validity rate | 81.2% |
| Strict-match attribution F1 | 0.359 |
| Fault-only F1 | 0.595 |
| Fully-explained rate | 73.0% |
| Expert preference (vs. random) | 78% |

---

## Repository Structure

```
├── aqa_model.py        # AQA oracle: I3D backbone, PSNet, cross-attention
│                       #   decoder, contrastive regressor, frozen-oracle
│                       #   contract (TSA/STSA from FineDiving)
├── diffusion.py        # Motion diffusion prior: transformer decoder with
│                       #   FiLM conditioning, DDPM/DDIM sampling, masked
│                       #   inpainting with RePaint resampling
├── counterfactual.py   # Half-point CF search: candidate enumeration,
│                       #   mask construction, batched inpainting,
│                       #   acceptance envelope, iterative peel-off loop
├── attribution.py      # Rule-based fault attribution: 17 rules (R1–R17),
│                       #   9 FINA fault categories, 8 joint groups,
│                       #   applicability gating
├── configs.py          # 6 frozen dataclasses (DataCfg, AQACfg,
│                       #   DiffusionCfg, CFCfg, EvalCfg, TrainCfg)
│                       #   totalling 68 hyperparameters
├── data.py             # FineDiving data loading, SMPL pose tokenisation,
│                       #   exemplar sampling, augmentation
├── train.py            # Training loops for AQA oracle and diffusion prior
├── evaluate.py         # AQA benchmark evaluation (Spearman ρ, R-ℓ₂)
├── experiments.py      # Full experimental pipeline: CF decomposition,
│                       #   attribution scoring, ablations, expert study
├── metrics.py          # CF-XAI metrics (validity, sparsity, proximity,
│                       #   realism/PFC, completion), attribution P/R/F1
├── default.yaml        # Default configuration (all 68 hyperparameters)
└── README.md
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 with CUDA support
- 4 × NVIDIA RTX 3090 (24 GB each) or equivalent

```bash
pip install torch torchvision torchaudio
pip install scipy numpy pyyaml tqdm matplotlib
```

---

## Data Preparation

### FineDiving

Download the FineDiving dataset from the [official repository](https://github.com/xujinglin/FineDiving) and extract frames following their protocol.

### SMPL Pose Extraction

Extract SMPL poses from broadcast video using [HybrIK](https://github.com/Jeff-sjtu/HybrIK). The axis-angle output is converted to 6D rotations at tokenisation time; foot contacts are derived by thresholding ankle/toe joint velocities at 0.05 m/s.

### FineDiving-HF Annotations

The FineDiving-HF annotations are provided as a JSON overlay on existing FineDiving identifiers. Place the annotation file in the data directory:

```
data/
├── FineDiving/              # original FineDiving frames and annotations
├── finediving_hf.json       # half-point fault labels (200 dives)
└── smpl_poses/              # pre-extracted SMPL pose sequences
```

---

## Usage

### 1. Train the AQA Oracle

```bash
python -m train \
    --stage aqa \
    --config default.yaml \
    --gpus 4 \
    --epochs 200
```

Training uses Adam with a two-tier learning rate (10⁻⁴ for I3D backbone, 10⁻³ for head modules), symmetric forward passes, and a curriculum that switches from ground-truth to predicted step boundaries at epoch 50. Total time: ~14 hours on 4× RTX 3090.

### 2. Train the Motion Diffusion Prior

```bash
python -m train \
    --stage diffusion \
    --config default.yaml \
    --gpus 4 \
    --epochs 2000
```

The diffusion model is trained exclusively on top-quartile dives (by final score within each action type) using AdamW with cosine noise schedule and classifier-free guidance (25% unconditional dropout). Total time: ~22 hours.

### 3. Run Counterfactual Decomposition

```bash
python -m experiments \
    --mode decompose \
    --config default.yaml \
    --aqa-checkpoint checkpoints/aqa_best.pth \
    --diffusion-checkpoint checkpoints/diffusion_ema.pth \
    --variant beam_sg
```

This runs the full half-point peel-off loop on all 200 FineDiving-HF dives. Five search variants are available:

| Variant | Description |
|---------|-------------|
| `random` | Random candidate selection (baseline) |
| `greedy` | Minimum-norm valid candidate |
| `greedy_sg` | Greedy + oracle score guidance |
| `beam` | Beam search (w=4) over joint groups |
| `beam_sg` | Beam + score guidance (**default**) |

Total time: ~25 minutes on a single GPU (~7.5 s/dive).

### 4. Evaluate Attribution Accuracy

```bash
python -m experiments \
    --mode evaluate \
    --config default.yaml \
    --results-dir results/beam_sg/
```

Reports strict-match, step+fault, and fault-only P/R/F1 against the FineDiving-HF expert labels, stratified by score-gap quintile, with 95% bootstrap confidence intervals.

---

## Configuration

All 68 hyperparameters are defined in six frozen dataclasses in `configs.py` and serialised in `default.yaml`:

| Dataclass | Fields | Scope |
|-----------|--------|-------|
| `DataCfg` | 12 | Frame counts, snippet layout, normalisation |
| `AQACfg` | 22 | Oracle architecture, training, curriculum |
| `DiffusionCfg` | 14 | Noise schedule, architecture, loss weights |
| `CFCfg` | 10 | Acceptance envelope, guidance ramp, late-start |
| `EvalCfg` | 5 | Exemplar count, bootstrap resamples |
| `TrainCfg` | 5 | Learning rates, weight decay, mixed precision |

Every field has a documented default and a provenance annotation. Override any parameter via the YAML file or command line.

---

## FINA Fault Taxonomy

The nine fault categories, with associated joint groups:

| Fault | Joint Group(s) | Phase |
|-------|----------------|-------|
| `bent_knees` | knees (4, 5) | somersault |
| `separated_feet` | ankles (7, 8, 10, 11) | somersault |
| `poor_body_line` | torso (0, 3, 6, 9), hips (1, 2) | any |
| `over_rotation` | hips (1, 2) | entry |
| `under_rotation` | hips (1, 2) | entry |
| `late_twist` | torso, shoulders (13, 14, 16, 17) | twist |
| `crooked_entry` | head (12, 15), shoulders | entry |
| `large_splash` | wrists (20–23), elbows (18, 19) | entry |
| `unstable_entry` | ankles, knees | entry |

The 17 deterministic rules (R1–R17) that map interventions to faults are implemented in `attribution.py`.

---

## Compute Footprint

| Stage | Wall-clock time (4× RTX 3090) |
|-------|-------------------------------|
| AQA training (200 epochs) | ~14 h |
| Diffusion training (2000 epochs) | ~22 h |
| CF decomposition (200 dives, 5 variants) | ~2.1 h |
| Attribution scoring | < 2 min |
| **Single-pass total** | **~37 h** |

Peak GPU memory: 14.8 GB (AQA), 16.2 GB (diffusion), 8.4 GB (CF inference).


---

## Acknowledgements

The FineDiving dataset is from [Xu et al., CVPR 2022 / IJCV 2024](https://github.com/xujinglin/FineDiving). SMPL pose extraction uses [HybrIK](https://github.com/Jeff-sjtu/HybrIK). The diffusion architecture builds on [EDGE](https://github.com/Stanford-TML/EDGE).

## License

This repository is released for academic research purposes. See `LICENSE` for details.

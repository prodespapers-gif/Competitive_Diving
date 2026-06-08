# Half-point counterfactual AQA for competitive diving

Reference implementation for the paper:

> **What Cost Half a Point? Counterfactual Action Quality Assessment for Competitive Diving.**
> *Submitted to Image and Vision Computing, 2026.*

The codebase decomposes the gap between an AQA model's predicted score and a hypothetical perfect score into a sequence of half-point counterfactual edits, each tagged with a FINA fault category and a body region. Built on:

- **FineDiving / FineDiving+** datasets (Xu et al., CVPR 2022 / IJCV 2024) for the underlying dive videos and AQA protocol;
- **EDGE** motion diffusion (Tseng et al., CVPR 2023) for the pose prior, ported verbatim where the numerical defaults matter (loss weights 0.636 / 0.646 / 2.964 / 10.942, EMA 0.9999, P2 SNR-based loss weighting, β̃-posterior cosine schedule);
- A new layer of expert fault annotations (**FineDiving-HF**: 200 dives, 9-category FINA taxonomy, 8 joint groups, κ = 0.71 inter-rater agreement) released alongside the code.

---

## Repository layout

```
half-point-cf/
├── configs/
│   └── default.yaml              # explicit Cfg() defaults; documentation
├── data/                         # placeholder — see "Dataset preparation"
│   ├── annotations/
│   │   ├── finediving/           # FineDiving train.pkl / test.pkl
│   │   └── finediving_plus/      # FineDiving+ extension (optional)
│   ├── frames/finediving/        # extracted video frames per dive
│   ├── poses/                    # extracted SMPL poses per dive (.npz)
│   ├── faults/                   # FineDiving-HF expert annotations
│   ├── smpl/                     # SMPL model files (user-provided)
│   └── i3d/                      # I3D Kinetics pretrained weights
├── experiments/                  # output: checkpoints, logs, configs, provenance
├── results/                      # output: CSVs, JSON caches, paper artifacts
├── scripts/
│   └── train_pose_proxy.py       # distil TSA-Net → PoseAQAProxy (required for CF)
├── src/
│   ├── configs.py                # frozen-dataclass Cfg + snapshot/fingerprint
│   ├── data.py                   # datasets, ExemplarSelector, FaultAnnotations
│   ├── aqa_model.py              # TSA + I3D backbone, symmetric training, curriculum
│   ├── diffusion.py              # EDGE-style motion diffusion + SMPL kinematics
│   ├── counterfactual.py         # half-point CF search algorithm
│   ├── attribution.py            # FINA fault taxonomy + dive-type gating
│   ├── metrics.py                # Spearman, R-ℓ2, attribution P/R/F1, EDGE PFC, CF-XAI
│   ├── train.py                  # DDP training (AQA + diffusion)
│   ├── evaluate.py               # CLI evaluator (4 subcommands)
│   └── experiments.py            # paper artifact generation (tables + figures)
├── LICENSE                       # MIT
├── README.md                     # this file
└── requirements.txt
```

Total source: ~12,000 lines of Python (camera-ready).

---

## Installation

```bash
git clone <repo-url> half-point-cf
cd half-point-cf
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The codebase was developed and tested on CentOS 7 / GCC 4.8.5 / CUDA 11.8 / 4× NVIDIA RTX 3090. See `requirements.txt` for notes on PyTorch version pinning if you're on an older system.

---

## Dataset preparation

Three datasets and three asset bundles are required for full reproduction. We do **not** redistribute any of them; obtain them from their respective sources.

| What | Where it goes | Source |
|---|---|---|
| FineDiving videos + annotations | `data/frames/finediving/`, `data/annotations/finediving/` | [FineDiving repo](https://github.com/xujinglin/FineDiving) (Xu et al., CVPR 2022) |
| FineDiving+ extension (optional) | `data/annotations/finediving_plus/` | Xu et al., IJCV 2024 |
| FineDiving-HF fault annotations | `data/faults/finediving_hf.json` | Released as part of this paper (see supplement) |
| SMPL body model files | `data/smpl/rest_positions.npy` | [SMPL website](https://smpl.is.tue.mpg.de) (gated; create account) |
| I3D Kinetics weights | `data/i3d/i3d_kinetics_rgb.pt` | [piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) |

Pose extraction (SMPL θ + τ per dive) is **not** part of this codebase. We used [HybrIK](https://github.com/Jeff-sjtu/HybrIK) for our experiments; any monocular SMPL estimator that outputs (axis-angle θ, root τ) at 30 fps will work. Outputs go to `data/poses/<dive_id>.npz` with two arrays: `theta` of shape `(T, 24, 3)` and `tau` of shape `(T, 3)`.

If `data/smpl/rest_positions.npy` is missing, the diffusion model falls back to approximate SMPL T-pose joint positions baked into `src/diffusion.py`. This works for development but reduces the fidelity of the position / contact losses; provide the real file for paper-quality numbers. The expected content is the SMPL_NEUTRAL joint regressor applied to the β=0 mean template:

```python
import numpy as np, pickle
with open("SMPL_NEUTRAL.pkl", "rb") as f:
    model = pickle.load(f, encoding="latin1")
J = model["J_regressor"] @ model["v_template"]
np.save("data/smpl/rest_positions.npy", J.astype("float32"))
```

---

## End-to-end pipeline

Each step's output is the next step's input. Stages 1–3 can be parallelised across GPUs via `torchrun`; stages 4–6 are single-GPU evaluation.

### 1. Train the AQA scorer

```bash
torchrun --nproc_per_node=4 -m src.train aqa --run-name aqa_baseline
```

Optionally enable the IJCV 2024 STSA extension:

```bash
torchrun --nproc_per_node=4 -m src.train aqa --run-name aqa_stsa --use-stsa
```

The training loop uses two features that take effect from epoch 0:

- **Symmetric (union-batch) training** (`cfg.aqa.use_symmetric_training`, default `true`): the loss includes both query→exemplar and exemplar→query MSE terms, matching FineDiving `helper.py:163`.
- **Step-transition curriculum** (`cfg.aqa.curriculum_threshold_epochs=50`): for the first 50 epochs, the model is fed ground-truth step transitions; from epoch 50 onward, it uses its own predicted transitions (matching FineDiving `helper.py:145-164`).

Output: `experiments/aqa_baseline/best.pt`. About 12 hours on 4× 3090 for 200 epochs.

### 2. Train the motion diffusion

```bash
torchrun --nproc_per_node=4 -m src.train diffusion \
    --run-name diff_v1 \
    --smpl-rest-positions data/smpl/rest_positions.npy
```

The diffusion training is a verbatim port of EDGE's numerical recipe:

- Loss weights `λ_simple=0.636`, `λ_pos=0.646`, `λ_vel=2.964`, `λ_contact=10.942` (verified against EDGE `model/diffusion.py:514-519`).
- EMA decay `0.9999` (EDGE `model/diffusion.py:63`).
- P2 SNR-based loss weighting on L_simple, L_vel, L_pos (NOT L_contact, matching EDGE lines 464/477/496).
- Predicts x̂ (the clean sample), cosine schedule with s=0.008, β̃-posterior — all configurable via `cfg.diffusion.*`.

Output: `experiments/diff_v1/best.pt`. The checkpoint includes the EMA weights and the `ContextEncoder` (action-type + difficulty conditioning). About 18 hours on 4× 3090 for 2000 epochs.

### 3. Train the pose-AQA proxy (required for CF search)

```bash
python -m scripts.train_pose_proxy \
    --aqa-ckpt experiments/aqa_baseline/best.pt \
    --run-name pose_proxy_v1 \
    --smpl-rest-positions data/smpl/rest_positions.npy
```

This is the bridge between pose space (diffusion) and AQA-score space (TSA-Net). It distils the frozen AQA into a small transformer that takes pose tokens and outputs scalar scores. Two phases:

- **Phase 1** (~10 min, single GPU): cache the AQA's M-exemplar voted predictions for every train+val dive. Saved as `experiments/pose_proxy_v1/teacher_cache.json`. Skipped automatically if the file already exists.
- **Phase 2** (~30 min, single GPU): train `PoseAQAProxy` against the cached targets via MSE distillation.

Output: `experiments/pose_proxy_v1/best.pt`. **Without this, `evaluate cf` runs with a random-init proxy and the resulting numbers are meaningless** (the script logs a loud warning when this happens).

### 4. Evaluate the AQA on the test set (paper Table 1)

```bash
python -m src.evaluate aqa \
    --ckpt experiments/aqa_baseline/best.pt \
    --output results/aqa_eval.csv
```

Output: a CSV with overall Spearman / R-ℓ2 (with 95% bootstrap CIs) plus per-action-type / per-quintile / per-height strata. About 5 minutes on one 3090.

### 5. Run counterfactual decomposition (the expensive step)

```bash
python -m src.evaluate cf \
    --aqa-ckpt experiments/aqa_baseline/best.pt \
    --diff-ckpt experiments/diff_v1/best.pt \
    --pose-proxy-ckpt experiments/pose_proxy_v1/best.pt \
    --output results/cf_cache.json
```

Output: a JSON cache with one entry per FineDiving-HF dive (~200). Each entry holds the full decomposition: per-peel interventions, fault categories, faithfulness diagnostics, per-record PFC scores, and the four diffusion-sampler knobs that produced each CF (`noise_level_tau_used`, `dime_gradient_scaling_used`, `inpaint_resamples_used`, `guidance_weight_used`). About 20 minutes on one 3090.

The CF search has several knobs (all read from `cfg.cf`):

| Knob | Default | What it does |
|---|---|---|
| `iterative_guidance_weights` | `[3.0, 5.0, 8.0]` | When the first weight yields no valid CF, retry at the next weight. Single-element list = legacy one-shot behaviour. |
| `noise_level_tau` (τ) | `1.0` | DiME late-start. τ<1.0 initialises from `q(z_τT \| x_known)` instead of pure noise — tighter CFs. |
| `dime_gradient_scaling` | `false` | DiME explicit `1/√ᾱ_t` factor on the score gradient (only meaningful with `--use-score-guidance`). |
| `inpaint_resamples` (R) | `0` | RePaint Algorithm 1: R inner iterations over the last `inpaint_resample_fraction` of steps. R=0 = single-pass EDGE. |
| `enforce_fault_dive_gating` | `true` | Drop predicted faults that don't apply to this dive (e.g. `late_twist` on a no-twist dive). |
| `pfc_metric` | `"edge"` | Realism metric. `edge` (EDGE-style), `lateral_accel` (legacy), or `both`. |

For ablations (greedy vs beam, with/without score guidance), repeat with different flags and different `--output` paths — each cache is a complete record.

### 6. Score attributions against the expert labels (paper Table 3)

```bash
python -m src.evaluate attribution \
    --decompositions results/cf_cache.json \
    --output results/attr_strict.csv \
    --match strict
```

Repeat with `--match step_fault` and `--match fault_only` for the loose-match variants. Takes seconds; reads the JSON cache.

Output columns include the new CF-XAI canonical vocabulary (validity / sparsity / proximity / realism / completion) under `cfxai_edge` and `cfxai_lateral_accel` scopes when the cache carries both PFC variants.

The eval-time fault gating (`cfg.eval.attribution_apply_dive_gating=true`) drops predicted tuples whose fault doesn't structurally apply to that dive type before computing per-fault precision/recall/F1. A `n_gated_by_dive_type` counter in the `meta` scope reports how many tuples were filtered.

### 7. Aggregate expert preference study (paper Table 4)

```bash
python -m src.evaluate expert_study \
    --rater-csvs ratings/rater_a.csv ratings/rater_b.csv ratings/rater_c.csv \
    --output results/expert_study.csv
```

Each rater CSV has columns `dive_id, cf_id, preferred` (bool). Outputs Cohen's κ pairwise agreement, per-rater rates, and majority-preferred rate per CF variant.

### 8. Generate paper-ready tables and figures

```bash
python -m src.experiments fig_2_pipeline           --output-dir paper/figs

python -m src.experiments tab_1_aqa_benchmark      --output-dir paper/tabs \
    --ours-csv results/aqa_eval.csv

# tab_2 uses the CF-XAI canonical columns by default. Pass --legacy-columns
# to emit the previous-version faithfulness/minimality/plausibility format.
python -m src.experiments tab_2_counterfactual_ablation --output-dir paper/tabs \
    --variant "Random baseline=results/cf_random.json" \
    --variant "Greedy=results/cf_greedy.json" \
    --variant "Greedy + score guidance=results/cf_greedy_guided.json" \
    --variant "Beam (w=4)=results/cf_beam.json" \
    --variant "Beam + score guidance=results/cf_beam_guided.json"

# tab_3 uses top_K_per_dive (per-dive aggregation, the headline).
python -m src.experiments tab_3_attribution_accuracy --output-dir paper/tabs \
    --strict-csv results/attr_strict.csv \
    --step-fault-csv results/attr_step_fault.csv \
    --fault-only-csv results/attr_fault_only.csv \
    --top-k 1 3

python -m src.experiments fig_5_qualitative_cf     --output-dir paper/figs \
    --cf-cache results/cf_cache.json

python -m src.experiments tab_4_cross_dataset_finediving_plus --output-dir paper/tabs \
    --aqa-csv results/aqa_eval_fdp.csv \
    --attr-csv results/attr_strict_fdp.csv

python -m src.experiments fig_6_expert_study       --output-dir paper/figs \
    --expert-csv results/expert_study.csv
```

LaTeX tables go to `paper/tabs/*.tex` (booktabs style, ready for `\input{}`); figures go to `paper/figs/*.pdf` (vector, with embedded fonts per IVC requirements).

**Table 1 baseline numbers** come from Xu et al., IJCV 2024, Table 2 (the unified-protocol re-evaluation of all major AQA methods on FineDiving). The table separates **TSA-Net** (Wang et al., CVPR 2021) and **TSA** (Xu et al., CVPR 2022 — the FineDiving paper) as distinct methods with distinct citations; these were collapsed into a single row in earlier code revisions and have now been disambiguated.

---

## Configuration

`configs/default.yaml` documents every available knob. To override a subset, copy the file and edit only what changes:

```bash
cp configs/default.yaml configs/big_diff.yaml
# edit configs/big_diff.yaml to set diffusion.width: 768

torchrun --nproc_per_node=4 -m src.train diffusion \
    --config configs/big_diff.yaml \
    --run-name diff_big \
    --smpl-rest-positions data/smpl/rest_positions.npy
```

Partial YAMLs are fine — unspecified fields keep their `Cfg()` defaults.

Key paper-faithful values that the validators enforce:

| Field | Value | Source |
|---|---|---|
| `diffusion.lambda_simple` | 0.636 | EDGE `model/diffusion.py:515` |
| `diffusion.lambda_vel` | 2.964 | EDGE `model/diffusion.py:516` |
| `diffusion.lambda_pos` | 0.646 | EDGE `model/diffusion.py:517` |
| `diffusion.lambda_contact` | 10.942 | EDGE `model/diffusion.py:518` |
| `train.diffusion_ema_decay` | 0.9999 | EDGE `model/diffusion.py:63` |
| `diffusion.p2_loss_weighting` | true | EDGE `model/diffusion.py:125-130` |
| `diffusion.cosine_s` | 0.008 | Nichol & Dhariwal, ICML 2021 §3 |
| `aqa.use_symmetric_training` | true | FineDiving `helper.py:36-39, 163` |
| `aqa.curriculum_threshold_epochs` | 50 | FineDiving `helper.py:145-164` |
| `data.num_frames` | 96 | FineDiving `FineDiving_TSA.yaml: frame_length` |
| `data.score_max` | 114.85 | Xu et al., IJCV 2024 §4 |

---

## Reproducibility

We've tried hard to make the runs reproducible:

- `Cfg.seed_everything()` sets all RNG seeds (Python / NumPy / PyTorch CPU+CUDA / cuDNN deterministic mode).
- `Cfg.snapshot()` writes the resolved config + provenance (git SHA, library versions, fingerprint hash) to `experiments/<run>/{config.yaml, provenance.yaml}` at run start.
- Every checkpoint carries the config fingerprint; evaluation refuses checkpoints whose fingerprint mismatches the runtime config.
- `ExemplarSelector` is deterministic in `(seed, query_dive_id)`, so the test-set inference protocol is identical across machines for a fixed seed.
- The CF search uses non-colliding per-peel seeds: `base_seed + peel_index * max(1000, len(iterative_guidance_weights)+1)`, so iterative retries within a peel don't conflict with later peels' seeds. The cache records the exact knobs that produced each CF (the four `*_used` fields), making each record independently reproducible.

Outputs that involve floating-point reductions across GPUs (DDP all-gather + bf16) may differ in the last few decimal places between runs. This is fundamental to the hardware and not under our control; the differences are below all reported CIs.
---

## License

MIT — see `LICENSE`. The FineDiving and FineDiving+ datasets, SMPL model files, and I3D pretrained weights are governed by their respective licenses; this code does not include or redistribute any of them.


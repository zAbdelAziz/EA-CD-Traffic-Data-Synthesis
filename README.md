# Conditional Diffusion for Traffic Data Synthesis and Lane-Change Prediction

Research code for **time-series entity-aware conditional diffusion for local traffic-scenario synthesis**, with downstream lane-change prediction under a **Train-on-Synthetic Test-on-Real (TSTR)** protocol.

This repository implements the thesis pipeline from raw microscopic NGSIM trajectories to fixed-slot diffusion sequences, synthetic traffic generation, deterministic downstream feature derivation, and utility-based classifier evaluation. The central research question is whether a structured conditional diffusion model can synthesize local highway traffic sequences that remain useful for real-data lane-change prediction.

![End-to-end training and evaluation pipeline](docs/assets/pipeline.png)

## Thesis alignment

The repository follows the thesis *Time-series Entity-Aware Conditional Diffusion for Traffic Data Synthesis to predict vehicle lane-change* by Mohamed Abdelaziz. The implementation is especially aligned with:

| Repository area | Thesis reference | What is implemented |
|---|---|---|
| `datasets/ngsim/interpolate.py` | Section 3.2.2 | Trajectory regularization, interpolation, Savitzky-Golay smoothing, velocity/acceleration/yaw-proxy recomputation. |
| `datasets/ngsim/diffusion_feature_builder.py` | Sections 3.1.2, 3.1.3, 3.2.3, 3.2.4 | Fixed-slot local traffic scenarios, `T=50`, six neighbor slots, horizon/boundary lane-change labels. |
| `datasets/ngsim/downstream_feature_builder.py` | Sections 3.2.5, 3.2.6 | Deterministic mapping from 20 diffusion features to 61 downstream kinematic, geometric, TTC, and gap features. |
| `models/synth/masked_gaussian_diffusion` | Sections 3.3.1, 3.3.4 | Gaussian forward process, cosine schedule, mixed reverse conversion for occupancy channels, DDPM/DDIM-style sampling. |
| `models/synth/unet_factorized_denoiser` | Section 3.3.2 | Entity-aware conditional denoiser: slot tokenization, frame-level entity self-attention, temporal U-Net, factorized output heads. |
| `trainers/synth.py` | Section 3.3.3 | Hybrid diffusion objective with ego, slot, mask, and temporal consistency losses. |
| `trainers/downstream.py` | Sections 3.4, 4.2, 4.5, 4.7 | TRTR/TSTR downstream evaluation using macro-F1, balanced accuracy, weighted F1, MCC, and confusion matrices. |
| `utils/post_process.py` | Section 3.3.4 | Structural projection: occupancy thresholding, absent-slot zeroing, front/rear sign conventions, plausible-range clipping. |

For a more detailed mapping, see [`docs/THESIS_ALIGNMENT.md`](docs/THESIS_ALIGNMENT.md).

## Model snapshots

The model is organized around a compact local traffic state and a factorized conditional denoiser. These snapshots are intentionally kept in the README because they explain the core design without requiring the reader to inspect the paper first.

### Fixed-slot traffic state

Each frame is encoded as ego velocity plus six semantically ordered neighbor slots. Missing neighbors are represented by `(dx=0, dy=0, p=0)`.

![Fixed-slot local traffic representation](docs/assets/fixed_slots.png)

The diffusion-space feature order is:

```text
[vx, vy,
 LF_dx, LF_dy, LF_p,
 LR_dx, LR_dy, LR_p,
 RF_dx, RF_dy, RF_p,
 RR_dx, RR_dy, RR_p,
 F_dx,  F_dy,  F_p,
 R_dx,  R_dy,  R_p]
```

This corresponds to the compact representation described in thesis Sections 3.1.2 and 3.1.3.

### Factorized conditional denoiser

The final denoiser separates **frame-level interaction modeling** from **temporal sequence denoising**. Entity attention models interactions among ego and slot tokens at each time step; the temporal U-Net then denoises the full sequence.

![Factorized conditional denoiser](docs/assets/denoiser.png)

The denoiser returns three heads:

| Head | Shape | Meaning |
|---|---:|---|
| `eps_ego` | `[B, T, 2]` | Noise prediction for ego velocity channels. |
| `eps_slots` | `[B, T, 6, 2]` | Noise prediction for continuous slot geometry channels. |
| `p_logits` | `[B, T, 6]` | Clean occupancy logits for binary slot presence. |

This mirrors the mixed reverse parameterization in thesis Sections 3.3.1-3.3.3.

### Repository component map

![Repository component map](docs/assets/components.png)

## Repository layout

```text
.
├── config.yaml                         # Single source of runtime configuration
├── main.py                             # Entrypoint
├── runner.py                           # End-to-end orchestration
├── datasets/
│   ├── base.py                         # Dataset cache/load/save abstraction
│   └── ngsim/                          # NGSIM cleaning, windowing, feature builders
├── models/
│   ├── common/                         # FiLM, attention, residual blocks, embeddings
│   ├── synth/                          # Diffusion kernels and denoisers
│   └── downstream/                     # Sequence classifiers
├── trainers/
│   ├── base.py                         # Splits, loaders, optimizer, scheduler, checkpointing
│   ├── synth.py                        # Diffusion training and synthetic generation
│   └── downstream.py                   # Classifier training and TSTR/TRTR evaluation
├── utils/                              # Metrics, standardizers, logging, post-processing
├── datasets-files/                     # Runtime data cache root; raw data is not committed
└── docs/                               # Research and engineering documentation
```

Directory-specific explanations are available in:

- [`datasets/README.md`](datasets/README.md)
- [`models/README.md`](models/README.md)
- [`models/synth/README.md`](models/synth/README.md)
- [`models/downstream/README.md`](models/downstream/README.md)
- [`trainers/README.md`](trainers/README.md)
- [`utils/README.md`](utils/README.md)
- [`datasets-files/README.md`](datasets-files/README.md)

## Data contract

The code expects the raw NGSIM CSV at:

```text
datasets-files/raw/ngsim.csv
```

The selected raw columns are configured in `config.yaml`:

```yaml
Vehicle_ID, Frame_ID, Local_X, Local_Y, Lane_ID, Preceding, Following
```

The thesis uses the NGSIM I-80 subset, a sampling interval of `0.1 s`, observation windows of `50` frames, a future horizon of `30` frames, stride `30`, and boundary-based lane-change labels with `theta_start = theta_end = 0.02` and `consec = 3`.

Large raw datasets, generated `.npz` caches, model checkpoints, logs, and W&B artifacts should not be committed. See [`datasets-files/README.md`](datasets-files/README.md) and [`.gitignore`](.gitignore).

## Installation

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build matching your CUDA version before installing the remaining requirements.

### 2. Place the dataset

```bash
mkdir -p datasets-files/raw
cp /path/to/ngsim.csv datasets-files/raw/ngsim.csv
```

### 3. Configure the run

Edit `config.yaml`. The default run uses:

```yaml
runner:
  model:
    diffusion: MaskedGaussianDiffusion
    synth: FactorizedUNETDenoiser
    downstream: BiGRUClassifier
  train:
    synth: true
    downstream: true
```

The repository configuration is the executable source of truth. The thesis reports a final diffusion setup in Section 4.3; if a published experiment must be reproduced exactly, confirm that the runtime values in `config.yaml` match that table before training.

## Common workflows

### Build refactored NGSIM windows only

Set `runner.train.synth=false` and `runner.train.downstream=false` after first construction is not currently useful because `Runner.run()` exits when both are false. The supported path is to run either synthesis or downstream training; the dataset cache is built automatically when `NgsimDataset(raw=True)` is created.

### Train the diffusion model and generate synthetic data

```bash
python main.py
```

With the default configuration, this performs:

1. Raw NGSIM loading and refactored cache creation if missing.
2. Diffusion standardization on the train split.
3. Synthetic model training.
4. Synthetic sequence generation.
5. Inverse standardization and structural post-processing.
6. Synthetic cache writing to `datasets-files/synth/`.
7. Downstream classifier training and real-data evaluation.

### Generate from an existing checkpoint only

Set:

```yaml
runner:
  train:
    synth: true
    synth_generate_only: true
    downstream: false
trainers:
  diffSynth:
    checkpoint:
      autoload_for_sampling: true
      autoload_tag: best
```

Then run:

```bash
python main.py
```

### Train downstream evaluation only

Set:

```yaml
runner:
  train:
    synth: false
    downstream: true
```

If a synthetic cache exists, the downstream trainer uses the synthetic cache as training data and evaluates on raw/refactored real data when the train and test dataset names match. If no synthetic cache exists, `BaseDataset` falls back to the refactored real dataset, producing a TRTR-style run.

## Output artifacts

| Output | Location | Notes |
|---|---|---|
| Clean raw CSV | `datasets-files/raw/ngsim-clean.csv` | Created after column selection, numeric conversion, filtering, and sorting. |
| Interpolated CSV | `datasets-files/raw/ngsim-interpolated-clean.csv` | Created after interpolation, smoothing, and kinematic recomputation. |
| Refactored real metadata | `datasets-files/refactored/ngsim.csv` | Window-level metadata: `Vehicle_ID`, `End_Frame_ID`, `y`. |
| Refactored real arrays | `datasets-files/refactored/ngsim.npz` | Contains `X: [N,50,20]`, `y: [N]`. |
| Diffusion standardizer | `datasets-files/refactored/ngsim.std.npz` | Feature-wise `mu`, `sigma` for diffusion-space data. |
| Synthetic metadata | `datasets-files/synth/ngsim-N*.csv` | Generated windows and labels. |
| Synthetic arrays | `datasets-files/synth/ngsim-N*.npz` | Generated `X: [N,50,20]`, `y: [N]`. |
| Downstream standardizer | `datasets-files/synth/ngsim.derived.std.npz` | Feature-wise stats for `Xd: [N,50,61]`. |
| Checkpoints | `models-checkpoints/` | Trainer-dependent checkpoint paths. |
| Logs | `logs/` | Local logger output. |

## Evaluation protocol

The key evaluation is **TSTR**:

1. Train the diffusion model on real traffic windows.
2. Generate class-balanced synthetic windows.
3. Derive downstream features from synthetic windows.
4. Train the downstream lane-change classifier on synthetic data.
5. Evaluate on real NGSIM-derived windows.

The main scalar metric is **macro-F1**, because all three maneuver classes should contribute equally. The trainer also logs accuracy, balanced accuracy, weighted F1, and MCC.

According to thesis Section 5.1, the proposed model reaches `78.53%` TSTR macro-F1 versus `80.41%` for the TRTR reference, a gap of `1.88` points. Treat these numbers as thesis-reported results, not automatic results from a fresh run unless the same data subset, preprocessing, configuration, and checkpoint-selection procedure are used.

## Reproducibility notes

- `runner.seed` controls NumPy/PyTorch seeding through `utils/seeder.py`.
- Dataset splits can be group-wise by `Vehicle_ID`, preventing windows from the same vehicle from crossing train/validation/test boundaries.
- Standardizers are fit on training indices only, then applied to validation/test or synthetic/real evaluation data as appropriate.
- Synthetic generation is chunked to avoid GPU memory overflow.
- Structural post-processing is deterministic and should be considered part of the generative pipeline.

## Production-readiness boundaries

This codebase is research-production oriented: it has clear configuration, deterministic caches, grouped splits, explicit standardization, checkpointing, and separated training/evaluation stages. It is not a packaged library yet. Before deploying this as an installable package or CI-validated project, add automated tests around feature-shape contracts, dataset cache compatibility, post-processing invariants, and checkpoint load/save round trips.

See [`docs/RESEARCH_PRODUCTION_CHECKLIST.md`](docs/RESEARCH_PRODUCTION_CHECKLIST.md) for the recommended hardening checklist.

# Thesis alignment guide

This document maps the repository to the thesis sections and clarifies where the mathematical and experimental concepts are implemented.

## Core thesis claim

The thesis evaluates whether structured conditional diffusion models can generate synthetic local highway traffic scenarios that preserve enough motion and interaction structure for real-data lane-change prediction. The code expresses that claim through a two-stage pipeline:

1. Synthesize compact traffic sequences in diffusion space.
2. Train a downstream classifier on synthetic data and evaluate it on real data.

This is the TSTR protocol described in Sections 3.1.4, 3.4, 4.2, and 4.5.

## Section-by-section implementation map

| Thesis section | Concept | Repository implementation |
|---|---|---|
| 3.1.1 | Dataset as labeled time-series samples `(X, y)` | `datasets/base.py`, `datasets/ngsim/__init__.py` |
| 3.1.2 | Ego-centered fixed-length local scenarios, `T=50`, slots `{LF, LR, RF, RR, F, R}` | `datasets/ngsim/diffusion_feature_builder.py` |
| 3.1.3 | Diffusion-space state dimension `dx=20` | `DiffusionFeatureBuilder._build_vehicle_sequence` and `README.md` feature-order contract |
| 3.1.4 | TSTR/TRTR separation | `Runner.train_synth`, `Runner.train_downstream`, `DownstreamTrainer` |
| 3.2.1 | Raw NGSIM attributes | `config.yaml: datasets.ngsim.columns` |
| 3.2.2 | Missing-frame handling, interpolation, smoothing, kinematic recomputation | `datasets/ngsim/interpolate.py` |
| 3.2.3 | Frame-wise lane index, neighbor retrieval, sliding windows | `DiffusionFeatureBuilder._build_frame_index`, `_pick_lane_lead_lag`, `_build_vehicle_sequence` |
| 3.2.4 | Horizon and boundary lane-change labels | `_label_by_horizon`, `_find_lane_change_segments`, `_label_by_boundary_crossing` |
| 3.2.5 | 61 downstream features | `datasets/ngsim/downstream_feature_builder.py` |
| 3.3.1 | Gaussian forward process and cosine schedule | `models/synth/masked_gaussian_diffusion/__init__.py` |
| 3.3.2 | Entity-wise tokenization, self-attention, temporal U-Net, FiLM, CFG | `models/synth/unet_factorized_denoiser/`, `models/common/film.py`, `models/common/time_embed.py` |
| 3.3.3 | Hybrid loss | `SynthTrainer._calc_loss` |
| 3.3.4 | Reverse sampling and structural projection | `MaskedGaussianDiffusionModel.p_sample_loop`, `utils/post_process.py` |
| 3.4 | BiGRU and Transformer-BiGRU downstream classifiers | `models/downstream/`, `trainers/downstream.py` |
| 4.1-4.5 | Dataset/training/generation/evaluation setup | `config.yaml`, trainer configs, runner flow |
| 4.7 | Distributional and downstream metrics | `utils/metrics.py`; distributional metrics are not yet consolidated as a reusable CLI |
| 5.1 | Main TSTR result | Not hard-coded; reproduced by running the configured pipeline with matching data and checkpoint selection |
| 5.3 | Cumulative ablations | Model variants are present under `models/synth/`; the exact ablation table is not automated as a single script |

## Important implementation notes

### Boundary labels

The default `config.yaml` uses boundary-based labels:

```yaml
labeling:
  method: boundary
  theta_start: 0.02
  theta_end: 0.02
  consec: 3
```

This follows Section 3.2.4. The code also supports the simpler `horizon` method, but the boundary method better aligns labels with the lane-change motion segment.

### Mixed continuous-binary diffusion

The diffusion kernel is Gaussian across all channels, including occupancy. The denoiser, however, does not predict occupancy as ordinary noise. Instead, it predicts clean occupancy logits, converts them to probabilities, and then converts those probabilities into an equivalent epsilon prediction for the reverse process. This is implemented in:

```text
models/synth/masked_gaussian_diffusion/__init__.py::_denoiser_out_to_eps
```

### Downstream feature derivation is part of evaluation

The downstream classifier sees `61` derived features, not the compact `20` generated features. Strong TSTR performance therefore requires the synthetic `20`-D sequence to preserve enough physical structure that higher-order features such as relative velocities, ranges, bearings, TTC, and lane-gap descriptors can be derived.

### Thesis configuration vs repository configuration

The thesis reports a final training configuration in Section 4.3. The repository `config.yaml` is executable and may intentionally use different values for extended or resumed experiments. When publishing a result, always archive the exact `config.yaml`, checkpoint, git commit, and data-cache metadata used for the run.

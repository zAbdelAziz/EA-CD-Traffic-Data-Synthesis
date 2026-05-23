# Training and evaluation

This document explains how training is orchestrated and how to interpret the evaluation outputs.

## Runner flow

The executable entrypoint is:

```bash
python main.py
```

`main.py` constructs `Runner`, and `Runner.run()` decides whether to run:

1. synthetic diffusion training and generation;
2. downstream lane-change classifier training;
3. both stages sequentially.

The behavior is controlled by:

```yaml
runner:
  train:
    synth: true
    downstream: true
```

## Synthetic training

Implemented by:

```text
trainers/synth.py::SynthTrainer
```

The trainer performs:

1. group-wise train/validation/test split through `BaseTrainer`;
2. optional diffusion-space standardization using training indices only;
3. random diffusion step sampling;
4. noisy input construction through the diffusion kernel;
5. denoiser prediction;
6. hybrid loss computation;
7. checkpointing on validation loss;
8. synthetic generation from the best checkpoint when configured.

### Hybrid loss

For factorized denoiser outputs, `SynthTrainer._calc_loss` combines:

| Loss component | Code name | Role |
|---|---|---|
| Ego epsilon MSE | `loss_ego` | Reconstruct noise for ego velocity. |
| Slot epsilon MSE | `loss_slots` | Reconstruct noise for neighbor geometry with presence-aware weights. |
| Occupancy BCE | `loss_mask` | Supervise clean binary occupancy. |
| Temporal slot consistency | `temp_slots` | Match frame-to-frame continuous slot-noise changes. |
| Temporal mask consistency | `temp_mask` | Match frame-to-frame occupancy changes. |

The implemented weighted sum is:

```text
2.0 * loss_ego
+ 1.0 * loss_slots
+ 0.5 * loss_mask
+ 0.1 * temp_slots
+ 0.05 * temp_mask
```

This corresponds to the objective described in thesis Section 3.3.3.

## Synthetic generation

`SynthTrainer.generate_synthetic` creates labels first. If `balance_labels=true`, it samples class labels approximately evenly across the three maneuver classes.

Generation is chunked in batches of up to `4096` samples to control GPU memory usage. The generated tensor is then inverse-standardized and passed through:

```text
utils/post_process.py::postprocess_synth_20
```

The post-processor enforces:

- occupancy probabilities in `[0, 1]`;
- hard occupancy decisions with optional hysteresis;
- zero geometry for absent slots;
- positive longitudinal offsets for front slots and negative offsets for rear slots;
- lateral and longitudinal clipping;
- ego velocity clipping.

## Downstream training

Implemented by:

```text
trainers/downstream.py::DownstreamTrainer
```

The downstream trainer derives 61-D features before training, then rebuilds data loaders. It standardizes derived features using the downstream training split and applies those statistics to validation and evaluation data.

Default classifier:

```yaml
runner:
  model:
    downstream: BiGRUClassifier
```

Default metrics:

- accuracy;
- balanced accuracy;
- macro-F1;
- weighted F1;
- Matthews correlation coefficient.

## TRTR and TSTR interpretation

### TRTR

Train-on-Real Test-on-Real is the real-data reference. It is obtained when training and testing use refactored real windows.

### TSTR

Train-on-Synthetic Test-on-Real is the primary synthetic-data utility evaluation:

1. Train/generate with the diffusion model.
2. Load synthetic cache as downstream training data.
3. Load raw/refactored real data as downstream test data when train and test dataset names match.
4. Evaluate the classifier on real data.

A high TSTR macro-F1 indicates that generated data preserves the interaction structure required by real-data lane-change prediction. It does not prove that every marginal feature distribution is perfect.

## Checkpointing

Checkpoint behavior is configured separately for diffusion and downstream training:

```yaml
trainers:
  diffSynth:
    checkpoint:
      metric: valid/loss
      mode: min
  downstream:
    checkpoint:
      metric: valid/macro_f1
      mode: max
```

The exact checkpoint directory is created by `BaseTrainer` under `models-checkpoints/` using the run name.

## W&B logging

The code imports `wandb` in trainers. To run without external logging, configure W&B in disabled/offline mode at the environment level, for example:

```bash
export WANDB_MODE=disabled
```

or configure your W&B project before running long experiments.

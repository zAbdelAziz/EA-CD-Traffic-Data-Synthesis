# Trainers

The trainer package contains shared training infrastructure plus specialized synthetic-generation and downstream-evaluation trainers.

## `BaseTrainer`

File: `trainers/base.py`

Responsibilities:

- device selection;
- dataset analysis hook;
- group-wise or random splits;
- PyTorch data loaders;
- loss, optimizer, scheduler construction from config;
- W&B initialization;
- checkpoint initialization and load/save helpers.

Group-wise splitting uses `Vehicle_ID` to reduce leakage across train/validation/test partitions.

## `SynthTrainer`

File: `trainers/synth.py`

Responsibilities:

- diffusion-space standardization;
- random diffusion-step sampling;
- denoiser training;
- hybrid loss computation;
- validation-loss checkpointing;
- synthetic generation;
- model visualization artifacts when enabled.

The trainer supports both tensor-output denoisers and factorized-output denoisers. The factorized path is the main path for the final model.

## `DownstreamTrainer`

File: `trainers/downstream.py`

Responsibilities:

- converting compact `20`-D samples to derived `61`-D features;
- feature standardization;
- classifier training;
- validation macro-F1 checkpointing;
- final test/evaluation metrics;
- optional confusion matrix logging to W&B.

## Trainer configuration

The relevant config keys are:

```yaml
trainers:
  diffSynth: ...
  downstream: ...
```

Each trainer defines its own epochs, batch size, split ratios, optimizer, scheduler, checkpoint metric, and standardization behavior.

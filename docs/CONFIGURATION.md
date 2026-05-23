# Configuration reference

Runtime behavior is controlled by `config.yaml` through OmegaConf.

## Main runner keys

```yaml
runner:
  name: HybridUNET[Ext]-BiGRU-NGSIM[20-61]
  seed: 42
  dataset:
    train: ngsim
    test: ngsim
    analyze: false
    online_synth: false
    synth_samples: 500200
  train:
    synth: true
    downstream: true
    synth_generate_only: false
  model:
    diffusion: MaskedGaussianDiffusion
    synth: FactorizedUNETDenoiser
    downstream: BiGRUClassifier
```

Important behavior:

- `dataset.train` and `dataset.test` select dataset entries under `datasets:`.
- `model.diffusion`, `model.synth`, and `model.downstream` select entries under `models:`.
- `train.synth_generate_only=true` skips training and samples from a loaded checkpoint.
- `dataset.synth_samples` controls the synthetic cache suffix.

## Dataset configuration

```yaml
datasets:
  ngsim:
    clsName: NgsimDataset
    columns: [...]
    preprocessing:
      dt: 0.1
      balance_classes: true
      sequence:
        length: 50
        stride: 30
        horizon: 30
        min_track_len: 120
      labeling:
        method: boundary
```

Changing any preprocessing field should be treated as a cache-breaking change for `datasets-files/refactored/`.

## Model configuration

Each model entry has:

```yaml
clsName: PythonClassName
hyperparams: {...}
```

The class must be importable from `models/__init__.py`, because `Runner` resolves classes dynamically with `getattr(models, clsName)`.

## Trainer configuration

Trainer entries control epochs, batch size, optimizer, scheduler, splitting, standardization, checkpointing, and sampling.

The diffusion trainer key is `diffSynth`; the downstream trainer key is `downstream`.

## Safe editing rules

- When changing `data_dim`, also update feature builders, denoisers, standardizers, post-processing, and documentation.
- When changing slot order, update every component that assumes `[LF, LR, RF, RR, F, R]`.
- When changing label semantics, update class names and all reported metrics.
- When changing split mode, document whether the experiment still prevents vehicle-level leakage.
- When changing standardization, delete old `.std.npz` files before rerunning.

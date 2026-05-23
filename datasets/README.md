# Datasets

The dataset package converts raw trajectory CSV files into cached PyTorch-compatible sequence datasets.

## Main abstractions

### `BaseDataset`

File: `datasets/base.py`

Responsibilities:

- resolve raw, refactored, synthetic, and standardizer paths from `config.yaml`;
- build refactored caches when missing;
- load real or synthetic caches;
- balance classes by downsampling to the minority class;
- expose `__len__` and dataset arrays.

`raw=True` means load/build the real refactored dataset. `raw=False` means load synthetic data; if synthetic data is missing, the class falls back to real refactored data.

### `NgsimDataset`

File: `datasets/ngsim/__init__.py`

Responsibilities:

- clean raw NGSIM CSV rows;
- select configured subsets;
- convert numeric columns;
- cast IDs/lane columns;
- build diffusion-space windows through `DiffusionFeatureBuilder`;
- return `(x, y)` tensors for PyTorch.

## NGSIM submodules

| File | Role |
|---|---|
| `interpolate.py` | Per-vehicle contiguous-frame interpolation, smoothing, and kinematic recomputation. |
| `diffusion_feature_builder.py` | Fixed-slot `20`-D diffusion-space sequence construction and label assignment. |
| `downstream_feature_builder.py` | Deterministic `20 -> 61` feature mapping for downstream classifiers. |

## Cache behavior

The dataset builders are intentionally cache-heavy because full NGSIM preprocessing is expensive. Delete stale caches whenever feature definitions, labels, preprocessing, or split assumptions change.

See [`../docs/DATA_PIPELINE.md`](../docs/DATA_PIPELINE.md) for the complete flow.

# Utilities

Utility modules support configuration, metrics, standardization, deterministic post-processing, logging, and reproducibility.

| File/directory | Purpose |
|---|---|
| `config.py` | Loads `config.yaml` through OmegaConf and exposes config values. |
| `dataset_split.py` | Group-wise and random split helpers. |
| `metrics.py` | Classification metrics from confusion matrices. |
| `pd_utils.py` | Numeric conversion helpers for CSV ingestion. |
| `post_process.py` | Deterministic structural projection for generated `[N,T,20]` sequences. |
| `seeder.py` | Global seed setup. |
| `logger/` | Local buffered logger. |
| `standardizer/` | Diffusion-space and downstream feature standardization. |
| `visuals/` | Model parameter counting and graph plotting helpers. |

## Post-processing invariants

`postprocess_synth_20` should preserve these invariants:

- output shape is `[N,T,20]`;
- occupancy channels are binary after projection;
- absent slots have zero `dx` and `dy`;
- front slots `LF`, `RF`, `F` have non-negative longitudinal offsets;
- rear slots `LR`, `RR`, `R` have non-positive longitudinal offsets;
- ego velocities and neighbor coordinates are clipped to plausible configured ranges.

These invariants are important because downstream features are derived from generated compact sequences.

# Dataset files

This directory is the runtime data root. It intentionally contains placeholders and helper scripts only. Large raw datasets and generated caches should not be committed.

Expected structure:

```text
datasets-files/
├── raw/
│   └── ngsim.csv                  # user-provided raw NGSIM data; not committed
├── refactored/
│   ├── ngsim.csv                  # generated metadata cache
│   ├── ngsim.npz                  # generated real windows: X, y
│   └── ngsim.std.npz              # generated diffusion standardizer
└── synth/
    ├── ngsim-N*.csv               # generated synthetic metadata
    ├── ngsim-N*.npz               # generated synthetic windows: X, y
    └── ngsim.derived.std.npz      # generated downstream standardizer
```

## Raw data

Place the raw NGSIM CSV here:

```text
datasets-files/raw/ngsim.csv
```

The required columns are configured in `config.yaml`.

## Version-control policy

Commit:

- README files;
- small scripts;
- empty `.gitkeep` placeholders if needed.

Do not commit:

- raw NGSIM data;
- cleaned/interpolated CSVs;
- `.npz` tensor caches;
- standardizer caches;
- generated synthetic datasets.

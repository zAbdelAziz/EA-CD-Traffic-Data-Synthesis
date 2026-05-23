# NGSIM dataset implementation

This module implements the NGSIM-specific data pipeline used by the diffusion and downstream models.

## Raw input

The expected input is a tabular trajectory file configured as `datasets-files/raw/ngsim.csv`. Required columns are listed in `config.yaml`.

## Processing order

1. `NgsimDataset._clean_raw_csv`
2. `InterpolateAndSmooth.run`
3. `DiffusionFeatureBuilder._build_frame_index`
4. `DiffusionFeatureBuilder._infer_lane_boundaries`
5. `DiffusionFeatureBuilder._build_feature_sequences`
6. optional class balancing in `BaseDataset.balance_classes`

## Label classes

```text
0 = lane keeping
1 = right lane change
2 = left lane change
```

## Slot order

```text
LF, LR, RF, RR, F, R
```

This order is shared across the whole repository. It is a hard contract.

## Thesis references

- Fixed-slot scenario representation: Sections 3.1.2 and 3.1.3.
- Preprocessing: Section 3.2.2.
- Scenario construction: Section 3.2.3.
- Lane-change label construction: Section 3.2.4.
- Downstream feature derivation: Section 3.2.5.

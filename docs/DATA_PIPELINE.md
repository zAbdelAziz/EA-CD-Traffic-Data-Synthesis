# Data pipeline

This document explains how raw NGSIM trajectories are transformed into model-ready tensors.

## 1. Raw CSV loading

Entry point:

```text
datasets/ngsim/NgsimDataset(raw=True)
```

Expected file:

```text
datasets-files/raw/ngsim.csv
```

Configured columns:

```text
Vehicle_ID, Frame_ID, Local_X, Local_Y, Lane_ID, Preceding, Following
```

`NgsimDataset._clean_raw_csv` performs:

1. optional subset selection, for example `i-80`;
2. German/European numeric-format conversion through `utils/pd_utils.py`;
3. missing-row dropping;
4. integer casting for ID/lane columns;
5. sorting by `Vehicle_ID` and `Frame_ID`.

## 2. Interpolation and smoothing

Implemented by:

```text
datasets/ngsim/interpolate.py::InterpolateAndSmooth
```

For each vehicle trajectory:

1. sort by frame;
2. remove duplicate `Frame_ID` rows, keeping the last occurrence;
3. reindex to the full contiguous frame interval;
4. forward/backward fill `Lane_ID`, `Preceding`, and `Following`;
5. linearly interpolate `Local_X` and `Local_Y`;
6. smooth positions with a Savitzky-Golay filter when track length permits;
7. recompute velocity, acceleration, and yaw-like proxy from smoothed coordinates.

This corresponds to thesis Section 3.2.2.

## 3. Frame index construction

Implemented by:

```text
DiffusionFeatureBuilder._build_frame_index
```

For each frame, the builder stores:

- lane-wise sorted vehicle lists by longitudinal position;
- a vehicle-ID to `(x, y)` map for same-lane `Preceding` and `Following` lookup.

This avoids a global nearest-neighbor search and preserves lane semantics.

## 4. Fixed-slot scenario construction

Implemented by:

```text
DiffusionFeatureBuilder._build_vehicle_sequence
```

For each ego vehicle and each sliding window:

- adjacent-lane front/rear neighbors are selected by lane-wise lead/lag search;
- same-lane front/rear neighbors are selected from `Preceding` and `Following` IDs;
- missing slots are encoded as `(0, 0, 0)`;
- each frame is packed into the fixed 20-D layout.

The default windowing parameters are:

| Parameter | Value | Meaning |
|---|---:|---|
| `length` | `50` | Observation window length. |
| `horizon` | `30` | Future frames used by horizon labels. |
| `stride` | `30` | Sliding-window stride. |
| `min_track_len` | `120` | Minimum retained vehicle track length. |

## 5. Label construction

Two labeling modes are implemented.

### Horizon labels

`_label_by_horizon` assigns the class from the first lane-index deviation in the future horizon:

- `0`: lane keeping;
- `1`: right lane change;
- `2`: left lane change.

### Boundary labels

`_find_lane_change_segments` detects lane-index changes and expands them into transition segments using the yaw-like proxy. `_label_by_boundary_crossing` labels a window as a lane-change sample only when the window end lies inside a detected segment.

The default config uses boundary labeling:

```yaml
method: boundary
theta_start: 0.02
theta_end: 0.02
consec: 3
```

This follows thesis Section 3.2.4.

## 6. Downstream feature construction

Implemented by:

```text
datasets/ngsim/downstream_feature_builder.py
```

The deterministic mapping converts `X: [N,T,20]` into `Xd: [N,T,61]`.

Feature groups:

| Group | Dimensions | Description |
|---|---:|---|
| Ego kinematics | 9 | Velocity, acceleration, yaw-like proxy, yaw derivatives, jerk. |
| Relative motion | 24 | Per-slot `dx`, `dy`, `dvx`, `dvy`. |
| Geometry | 12 | Per-slot range and bearing. |
| TTC | 6 | Per-slot longitudinal time-to-collision. |
| Gap descriptors | 10 | Left/right merge-relevant gaps, closing rates, and minimum TTC. |

This follows thesis Section 3.2.5. The diffusion model is not asked to generate engineered features directly; it generates only the compact state from which downstream features must be recoverable.

## 7. Caches and invalidation

The dataset classes cache expensive outputs to disk. Delete the corresponding cache files when changing any preprocessing, labeling, or feature-layout logic.

| Cache | Delete when changing |
|---|---|
| `datasets-files/raw/ngsim-clean.csv` | raw-column cleaning, subset selection, numeric conversion |
| `datasets-files/raw/ngsim-interpolated-clean.csv` | smoothing/interpolation/kinematic recomputation |
| `datasets-files/refactored/ngsim.csv` and `.npz` | windowing, neighbor slots, label construction, feature order |
| `datasets-files/refactored/ngsim.std.npz` | diffusion standardization logic |
| `datasets-files/synth/ngsim-N*.csv` and `.npz` | model checkpoint, sampling settings, post-processing |
| `datasets-files/synth/ngsim.derived.std.npz` | downstream feature builder or downstream train split |

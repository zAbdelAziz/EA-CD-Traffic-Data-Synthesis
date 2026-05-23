# Standardizers

Standardization is split by representation.

## `DiffusionStandardizer`

File: `utils/standardizer/diffusion.py`

Used before diffusion training. It fits feature-wise statistics on the training subset only.

Important behavior:

- ego velocity channels are standardized normally;
- slot `dx/dy` statistics are estimated only from present slots;
- occupancy channels remain in `[0, 1]`;
- absent slot geometry is kept at zero after transformation.

## `DownstreamStandardizer`

File: `utils/standardizer/downstream.py`

Used after deriving the 61-D downstream feature representation. It standardizes each derived feature using training-subset statistics.

## Cache files

| Standardizer | Cache |
|---|---|
| Diffusion | `datasets-files/refactored/ngsim.std.npz` |
| Downstream | `datasets-files/synth/ngsim.derived.std.npz` |

Delete these files when changing feature definitions or split assumptions.

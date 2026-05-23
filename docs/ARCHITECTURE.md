# Architecture

This document explains the model architecture and the exact code locations that implement each component.

## State representation

The diffusion model operates on `X` with shape:

```text
[N, T, D] = [num_windows, 50, 20]
```

Each frame uses the following 20-dimensional layout:

```text
0:  vx
1:  vy
2:  LF_dx    3: LF_dy    4: LF_p
5:  LR_dx    6: LR_dy    7: LR_p
8:  RF_dx    9: RF_dy   10: RF_p
11: RR_dx   12: RR_dy   13: RR_p
14: F_dx    15: F_dy    16: F_p
17: R_dx    18: R_dy    19: R_p
```

The slot order is fixed and semantic. It must not be changed without also updating the standardizer, downstream feature builder, denoiser tokenization, post-processing, and documentation.

## Diffusion kernel

Implemented by:

```text
models/synth/masked_gaussian_diffusion/MaskedGaussianDiffusionModel
```

Responsibilities:

- construct beta schedules (`linear` or `cosine`);
- sample `x_t` from `x_0` via `q_sample`;
- convert factorized denoiser outputs into a full epsilon tensor;
- optionally apply classifier-free guidance during sampling;
- run DDPM or DDIM-style reverse sampling with optional respacing.

The occupancy channels are handled through a clean-mask prediction path:

1. The denoiser predicts `p_logits`.
2. The logits become `p_hat = sigmoid(p_logits)`.
3. `p_hat` is converted into epsilon for the occupancy channels.
4. Continuous slot geometry can be gated toward the absent state when occupancy is low.

This implements the mixed continuous-binary parameterization described in thesis Sections 3.3.1 and 3.3.4.

## Factorized U-Net denoiser

Implemented by:

```text
models/synth/unet_factorized_denoiser/FactorizedUNetDenoiserModel
models/synth/unet_factorized_denoiser/entity_attn.py
```

Input:

```text
x_t: [B, T, 20]
t:   [B]
y:   [B] or None
```

Output:

```python
{
    "eps_ego":   Tensor[B, T, 2],
    "eps_slots": Tensor[B, T, 6, 2],
    "p_logits":  Tensor[B, T, 6],
}
```

### Conditioning path

The denoiser builds a global conditioning vector from:

- sinusoidal diffusion-step embedding;
- learned maneuver-class embedding;
- null class embedding for classifier-free guidance;
- `CondMLP` projection.

Relevant files:

```text
models/common/time_embed.py
models/common/mlp.py
models/common/film.py
```

During training, class labels are randomly replaced by the null label with probability `cfg_drop_prob`.

### Entity tokenizer

The state is split into:

- one ego token from `[vx, vy]`;
- six neighbor tokens from `[dx, dy, p]` per slot.

Separate linear projections map ego and neighbor states into a common token dimension. Learned slot embeddings preserve entity identity.

### Entity attention

`EntityAttentionBlock` applies self-attention across the seven entities independently at each physical time step. This models the local interaction topology relevant for lane changes before temporal denoising.

### Temporal U-Net

After entity attention, the seven tokens are flattened into temporal channels and passed through a 1D U-Net:

- input projection to base channels;
- two residual blocks at each resolution;
- downsampling to wider channels;
- optional temporal attention at low and middle resolutions;
- upsampling with additive skip connections;
- output projection back to token space.

Core building blocks are in `models/common/`.

## Downstream classifiers

Implemented under:

```text
models/downstream/
```

The primary classifier is `BiGRUClassifierModel`, which reads derived `61`-D features and uses the last bidirectional GRU output for three-class classification. The additional `TransformerBiGRUClassifierModel` combines Transformer encoding, stacked BiGRU modeling, and attention aggregation.

## Baseline and historical models

The repository also includes alternative denoisers and diffusion kernels:

| Model | Purpose |
|---|---|
| `GaussianDiffusionModel` | Simple epsilon-only Gaussian baseline. |
| `HybridDiffusionModel` | Historical mixed continuous/discrete attempt. |
| `UNETDenoiserModel` | Unfactorized temporal U-Net baseline. |
| `TransformerDenoiserModel` | Generic Transformer denoiser baseline. |
| `FactorizedTransformerDenoiserModel` | Entity-aware Transformer-style baseline. |
| `FactorizedUNetDenoiserModel` | Final structured model used by default. |

Use these for ablations only after confirming that the selected diffusion kernel and trainer loss are compatible with the model output format.

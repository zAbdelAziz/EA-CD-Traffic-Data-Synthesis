# Synthetic generation models

This package contains diffusion kernels and denoisers used for traffic sequence synthesis.

## Diffusion kernels

| Module | Class | Purpose |
|---|---|---|
| `gaussian_diffusion` | `GaussianDiffusionModel` | Standard Gaussian diffusion baseline with epsilon prediction. |
| `masked_gaussian_diffusion` | `MaskedGaussianDiffusionModel` | Main diffusion kernel for mixed continuous-binary fixed-slot traffic states. |
| `hybrid_diffusion` | `HybridDiffusionModel` | Historical/experimental mixed continuous-discrete implementation. |

## Denoisers

| Module | Class | Output format | Purpose |
|---|---|---|---|
| `unet_denoiser` | `UNETDenoiserModel` | Tensor epsilon | Unfactorized temporal U-Net baseline. |
| `unet_factorized_denoiser` | `FactorizedUNetDenoiserModel` | Dict heads | Final entity-aware U-Net denoiser. |
| `transformer_denoiser` | `TransformerDenoiserModel` | Tensor epsilon | Generic Transformer baseline. |
| `transformer_factorized_denoiser` | `FactorizedTransformerDenoiserModel` | Dict heads | Entity-aware Transformer-style baseline. |

## Final model

The default final model is `FactorizedUNetDenoiserModel` with `MaskedGaussianDiffusionModel`.

Key design points:

- fixed-slot entity tokenization;
- learned slot embeddings;
- entity self-attention over ego and six slots;
- time and class conditioning;
- classifier-free guidance label dropout;
- temporal U-Net with residual blocks and attention;
- separate output heads for ego noise, slot-geometry noise, and occupancy logits.

See [`../../docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) for a full explanation.

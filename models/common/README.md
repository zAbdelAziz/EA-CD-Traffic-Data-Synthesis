# Common model components

Reusable layers used by synthetic and downstream models.

| File | Component | Role |
|---|---|---|
| `film.py` | `FiLM` | Produces feature-wise scale and shift parameters from a conditioning vector. |
| `mlp.py` | `CondMLP` | Builds the global conditioning embedding. |
| `time_embed.py` | `SinusoidalTimeEmbedding` | Encodes diffusion steps. |
| `positional_encoding.py` | `PositionalEncoding`, `RoPE` | Sequence-position encodings for Transformer-style models. |
| `residual_block.py` | `ResBlock1D` | 1D residual block with conditioning. |
| `self_attn.py` | `SelfAttention1D` | Temporal self-attention over sequence channels. |
| `up_down_sample.py` | `Upsample1D`, `Downsample1D` | Temporal resolution changes for U-Net backbones. |
| `add_attn.py` | `AdditiveAttention` | Attention aggregation used by downstream sequence models. |

These components implement the conditioning and temporal modeling machinery referenced in thesis Section 3.3.2.

# Models

The model package contains reusable neural-network components, diffusion kernels, denoisers, and downstream classifiers.

## Package structure

| Directory | Purpose |
|---|---|
| `common/` | Shared layers: FiLM, conditional MLPs, residual blocks, attention, up/down sampling, positional/time embeddings. |
| `synth/` | Diffusion kernels and denoising models for synthetic traffic generation. |
| `downstream/` | Sequence classifiers for lane-change prediction. |

All selectable models are exported through `models/__init__.py` and instantiated dynamically by `Runner` from `config.yaml`.

## Active default model stack

```yaml
runner:
  model:
    diffusion: MaskedGaussianDiffusion
    synth: FactorizedUNETDenoiser
    downstream: BiGRUClassifier
```

This stack corresponds to the final structured conditional diffusion pipeline described in thesis Section 3.3 and evaluated in Chapter 5.

## Adding a new model

1. Implement a class under `models/`.
2. Export it from the relevant `__init__.py` and top-level `models/__init__.py`.
3. Add a `models.<Name>` entry in `config.yaml` with `clsName` and `hyperparams`.
4. Confirm trainer compatibility. Tensor-output denoisers and factorized-output denoisers use different loss paths.
5. Run a small import/smoke test before launching long training.

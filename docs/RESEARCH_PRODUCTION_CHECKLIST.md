# Research-production checklist

This checklist is intended for preparing the repository for public GitHub release and future reproducibility.

## Required before public release

- [ ] Confirm the raw NGSIM dataset cannot be redistributed with this repository.
- [ ] Add the chosen license file. Do not assume a license by default.
- [ ] Archive the exact thesis-result `config.yaml` used for the reported tables.
- [ ] Add a small synthetic fixture dataset for CI that does not contain restricted data.
- [ ] Add shape-contract tests for `[N,T,20]` and `[N,T,61]` tensors.
- [ ] Add post-processing invariant tests: absent slots have zero geometry, front/rear signs are valid, occupancy is binary after projection.
- [ ] Add split tests proving that `Vehicle_ID` groups do not cross partitions in group mode.
- [ ] Add checkpoint load/save tests for all actively supported models.
- [ ] Add a smoke test that imports every configured class from `config.yaml`.
- [ ] Add CLI-level documentation for running TRTR, TSTR, and generation-only modes.

## Recommended engineering improvements

- [ ] Move scripts under a formal `src/` package or add packaging metadata.
- [ ] Add type hints to dataset builders and trainers.
- [ ] Replace magic post-processing bounds with versioned dataset-stat artifacts.
- [ ] Add distributional metric scripts for Wasserstein and KS statistics from thesis Section 4.7.1.
- [ ] Add ablation-run config files under `configs/ablations/`.
- [ ] Add experiment cards under `experiments/` with data hash, config hash, checkpoint path, and metrics.
- [ ] Add pre-commit hooks for formatting, import sorting, and linting.
- [ ] Add GitHub Actions for import tests and fixture-data smoke tests.

## Reproducibility bundle for any published result

For each reported result, store:

```text
- git commit hash
- config.yaml
- raw dataset source and version
- refactored cache generation timestamp
- train/valid/test split seed
- model checkpoint
- standardizer files
- synthetic cache metadata
- final metrics JSON/CSV
- hardware and PyTorch/CUDA versions
```

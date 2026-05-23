# Downstream models

Downstream models evaluate whether generated synthetic traffic sequences preserve enough information for real-data lane-change prediction.

They are evaluation instruments, not the main methodological contribution.

## Inputs

All downstream models consume derived features:

```text
Xd: [N, T, 61]
```

These are built by `datasets/ngsim/downstream_feature_builder.py` from compact diffusion-space samples.

## Available classifiers

| Model | File | Description |
|---|---|---|
| `BiGRUClassifierModel` | `bigru_classifier/__init__.py` | Two-layer bidirectional GRU with a LayerNorm + Linear classification head. |
| `TransformerBiGRUClassifierModel` | `transformer_bigru_classifier/__init__.py` | Transformer encoder plus BiGRU and attention aggregation. |
| `XLSTMClassifierModel` | `xlstm_classifier/__init__.py` | Placeholder/experimental class. |

## Metrics

Metrics are computed in `trainers/downstream.py` using helpers from `utils/metrics.py`:

- accuracy;
- balanced accuracy;
- macro-F1;
- weighted F1;
- Matthews correlation coefficient;
- confusion matrix.

Macro-F1 is the main comparison metric because lane keeping, right lane change, and left lane change should contribute equally.

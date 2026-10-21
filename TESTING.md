# Testing

```sh
python -m unittest discover -s tests -v
```

The saved October 21, 2026 verification run passed **6 tests**:

- Deterministic stratified splits, class coverage, and disjoint contents.
- Exact duplicate removal and rejection of conflicting labels.
- Rejection of modified data and overlapping manifests.
- Notebook cells executed in order with training/evaluation disabled.
- Actual ResNet50V2 construction with `weights=None`, frozen base, six-class output, and preprocessing checked against Keras; custom CNN inference on a synthetic image.
- Tiny random fixture models saved/reloaded and evaluated on generated images, checking metric reports and plots.

These checks use no Intel data, training, or pretrained downloads. Synthetic metrics are code-path checks, not model results. The two model tests are skipped when TensorFlow is unavailable. Historical validation results are documented separately in [RESULTS.md](RESULTS.md).

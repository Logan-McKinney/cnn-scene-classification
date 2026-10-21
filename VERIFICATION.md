# Verification

`python -m unittest discover -s tests -v`: **6 tests passed**, 47.818 seconds.

- Deterministic stratified membership, disjoint contents and all-class coverage.
- Exact duplicate removal and rejection of conflicting duplicate labels.
- Rejection of modified content and overlapping manifests.
- All notebook code cells executed in order with training and evaluation disabled.
- Built actual ResNet50V2 with `weights=None` (no download); checked frozen base,
  six-way output and numeric equivalence of preprocessing to Keras V2 preprocess_input.
- Built the custom CNN and performed one inference pass on a synthetic image.
- Saved/reloaded tiny random Keras fixture models and ran evaluation on generated
  images, confirming accuracy/report/confusion-matrix JSON and curve/confusion PNG files.

The synthetic evaluation is a code-path test; its metrics do not describe either
real trained architecture or the Intel dataset. No pretrained weights were downloaded,
no fit/training was run, and no Intel images were evaluated. Historical validation
metrics are labeled separately in RESULTS.md.

Remaining: an explicitly authorized full training/evaluation run and confirmation
of complete team credit, personal contribution and redistribution permission.

Verified on 2026-10-21, Windows, Python 3.13.3.
The preinstalled TensorFlow was reused. Only FAISS, python-multipart and Uvicorn
were downloaded into a temporary test dependency directory; no ML framework was installed.
Initial sandbox attempts failed on Windows temporary-folder permissions. The suites
were rerun outside that sandbox and completed with exit code 0.

Dependency versions actually used are in `tested-environment.json`. Requirements
specify compatible ranges, not a complete cross-platform lockfile. Existing package
deprecation warnings were observed (Starlette/httpx, Keras/NumPy and TensorFlow/gast);
they did not fail the tests. No dataset was downloaded or copied. No network was trained.

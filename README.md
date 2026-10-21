# Natural Scene Classification

Compare a custom CNN with frozen ImageNet ResNet50V2 for six Intel scene classes:
buildings, forest, glacier, mountain, sea and street.

## Methods and corrections

The custom CNN retains three Conv2D blocks (32/64/128 filters), pooling and batch
normalization, then Flatten, Dense(128), Dropout(0.5) and six-way softmax. ResNet50V2
retains a frozen base, global average pooling and the same dense head. Defaults
preserve 64x64 images, 15 maximum epochs, batch size 32 and early stopping on
validation loss with best weights restored. Adam rates are 0.001 and 0.0001 respectively.

Raw RGB pixels feed both models. The custom CNN scales to [0,1]; ResNet50V2 scales
to [-1,1], equivalent to the [documented V2 preprocessing](https://keras.io/2/api/applications/resnet/).
Train-only augmentation lives inside the model and is not cached. Seeded, stratified
70/15/15 splits are made at file level before TensorFlow dataset construction.
Exact-content duplicates are removed; conflicting duplicate labels fail explicitly.
Every class appears in every split; small classes receive at least one validation
and one test image. Saved manifests contain relative paths and content hashes.
Evaluation verifies unchanged content and non-overlap. Both models share the split.

## Dataset and setup

Obtain the [Intel Image Classification dataset on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
separately; it is associated with Intel's scene-classification challenge and hosted
by Puneet Bansal. See ATTRIBUTION.md and review the source's current usage terms.
No dataset, trained weights or outputs are bundled. Point `--data` at the folder
containing the six labeled subfolders, often `seg_train` or `train/seg_train`.
This project creates an internal hold-out from that folder. It does not use the
separately supplied `seg_test` or unlabeled `seg_pred` directories.

Reuse a compatible environment, or create one with a Python version supported by
TensorFlow (3.11/3.12 is a practical starting point). Install only if needed:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Training is opt-in and was not performed for this cleanup. When separately authorized:

```powershell
python scene.py train --data "<local-seg_train-directory>" --run runs/comparison --seed 123 --epochs 15 --size 64 --batch 32
python scene.py evaluate --data "<local-seg_train-directory>" --run runs/comparison
```

Run from the project folder. Use a fresh run directory for each experiment.
Training can download ResNet50V2 ImageNet weights if absent from the Keras cache.
Each run saves split/config JSON, TensorFlow version, model files and history JSON.
Evaluation writes accuracy, per-class precision/recall/F1/support, confusion matrices,
and accuracy/loss curves for both models. Use validation for selection and reserve
the test set for final reporting. For exact environment recording, save
`python -m pip freeze` locally alongside the run; runtime/GPU versions can affect results.

For Jupyter, optionally install `requirements-notebook.txt`, set `SCENE_DATA_DIR`
in your environment, and open `Assignment3.ipynb`. Cells run in order; training
and evaluation flags default to False. Enable them explicitly only when intended.

## Results and limitations

Historical notebook reports show about 75% ResNet50V2 validation accuracy and 69%
custom CNN validation accuracy. These used the original flawed preprocessing/splits;
they are not fresh test results or evidence for this corrected pipeline. See
RESULTS.md for provenance and VERIFICATION.md for tests actually run. No corrected
model was retrained and no new Intel test accuracy is available.

64x64 inputs trade detail for speed and may disadvantage transfer learning. A frozen
base limits domain adaptation. Exact duplicate removal cannot detect near duplicates
or photos from the same scene, so group-aware splitting may be needed. Deterministic
seeds/op settings improve repeatability, not cross-hardware bitwise equality. Tests
cover split integrity, preprocessing, model construction and synthetic evaluation;
they do not establish real-data accuracy or prove a full training run succeeds.

## Attribution

This work derives from `Assignment3.ipynb` in the collaborative
[CNN_Image_Classification repository](https://github.com/conleypj17/CNN_Image_Classification).
Credit the original team rather than presenting the notebook as solely authored work.
Local Git history identifies Patrick Conley; see ATTRIBUTION.md for publication prerequisites.

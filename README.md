# Natural Scene Classification

A group project comparing a custom CNN with ImageNet-pretrained **ResNet50V2** on six Intel natural-scene classes: buildings, forest, glacier, mountain, sea, and street. We explored how a small convolutional network compares with transfer learning. The team was **Logan McKinney, Patrick Conley, and Greg Pologruto**.

## Original validation results

| Model | Saved validation accuracy |
| --- | ---: |
| ResNet50V2 with a frozen base | **74.86%** |
| Custom CNN | **About 69%** |

These are the original notebook's saved validation results, not held-out test accuracy or newly generated scores. Later revisions corrected ResNet50V2 preprocessing and improved splitting/evaluation; those revisions were **not retrained** and are not the source of these scores. See [results provenance](RESULTS.md).

## Approach

- **Custom CNN:** three convolutional blocks with 32/64/128 filters, pooling and batch normalization, followed by a 128-unit dense layer, dropout, and six-class softmax.
- **Transfer learning:** frozen ResNet50V2, global average pooling, and a dense classification head.
- **Current implementation:** 64×64 images, train-only augmentation, early stopping, and seeded, stratified 70/15/15 file splits shared by both models. Exact duplicates are removed and split contents are checked before evaluation.

The custom CNN scales RGB pixels to [0,1]; ResNet50V2 uses [-1,1]. Default training is up to 15 epochs with batch size 32. Small images make experiments more manageable but lose detail, and exact duplicate checks do not detect related or near-duplicate scenes.

## Dataset and setup

Uses Python, TensorFlow/Keras, NumPy, scikit-learn, Matplotlib, and Pillow. Obtain the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) separately. Point `--data` at the directory containing the six labeled class folders, often `seg_train` or `train/seg_train`. The script creates its own splits from that directory; it does not use the separate `seg_test` or `seg_pred` folders. Dataset images and model weights are excluded from Git.

From the repository root, using a Python version supported by TensorFlow:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scene.py train --data "<local-seg_train-directory>" --run runs/comparison --seed 123 --epochs 15 --size 64 --batch 32
python scene.py evaluate --data "<local-seg_train-directory>" --run runs/comparison
```

Training can download ResNet50V2 ImageNet weights if they are not cached. Use a fresh run directory for each experiment. Runs save split/config manifests, models, and training histories; evaluation writes test metrics, per-class reports, confusion matrices, and curves.

For the notebook workflow, install `requirements-notebook.txt`, set `SCENE_DATA_DIR` to your dataset directory, and open [Assignment3.ipynb](Assignment3.ipynb). It retains the notebook workflow with updated setup and evaluation cells. Training/evaluation flags default to `False`; enable them to run an experiment.

## Tests

```sh
python -m unittest discover -s tests -v
```

The saved verification run passed **6 tests** for split integrity, duplicate handling, notebook execution with training disabled, preprocessing/model construction, and synthetic evaluation. They do not establish real-data accuracy. See [testing notes](TESTING.md).

## Credits

This was a group project by Logan McKinney, Patrick Conley, and Greg Pologruto, originally developed in the collaborative [CNN_Image_Classification repository](https://github.com/conleypj17/CNN_Image_Classification). Later Codex-assisted revisions prepared the script, splitting/evaluation improvements, and tests. See [attribution and dataset credit](ATTRIBUTION.md).

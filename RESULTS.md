# Results provenance

No new Intel training or test evaluation was run during portfolio preparation.

The original Assignment3 notebook contains these saved outputs:

| Model | Original evaluation | Saved accuracy | Support |
| --- | --- | --- | --- |
| Frozen ResNet50V2 | Validation, cell 4 scalar evaluation | 0.7485795617103577 | Not printed by scalar evaluation |
| Frozen ResNet50V2 | Validation classification report, cell 5 | 0.75 (rounded) | 2,112 |
| Custom CNN | Validation classification report, cell 7 | 0.69 (rounded) | 2,112 |

Cell numbers above are zero-based source JSON cell indexes. Saved outputs are
historical evidence, not reruns. The original val/test source was shuffled before
take/skip, so stable held-out separation cannot be assumed. Original ResNet50V2
received [0,1] values instead of expected [-1,1]. The corrected preprocessing and
split change the experiment; old numbers cannot validate the new implementation.

After an explicitly authorized training run, evaluate the saved models on the
saved test manifest and report `test-results.json` with configuration, environment,
class support and plots. Synthetic test metrics must never be reported as Intel results.

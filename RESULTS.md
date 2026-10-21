# Original validation results

The original group notebook saved the following outputs. These are historical validation results, not reruns or held-out test scores.

| Model | Original saved output | Accuracy | Report support |
| --- | --- | ---: | ---: |
| Frozen ResNet50V2 | Scalar validation evaluation | 0.7485795617103577 (74.86%) | Not printed |
| Frozen ResNet50V2 | Validation classification report | 0.75 (rounded) | 2,112 |
| Custom CNN | Validation classification report | 0.69 (rounded) | 2,112 |

These came from source notebook cells 4, 5, and 7 respectively (zero-based JSON indexes), recorded before the notebook was reorganized and outputs cleared. The original notebook belongs to the [group source repository](https://github.com/conleypj17/CNN_Image_Classification).

The original ResNet50V2 input scaling was [0,1], and shuffled take/skip splitting did not guarantee stable held-out membership. The current implementation uses [-1,1] preprocessing and fixed file-level splits. It has not been retrained, so the original validation scores do not measure the revised pipeline. No new Intel test accuracy is available.

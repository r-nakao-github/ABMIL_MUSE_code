# ABMIL for Weakly Supervised Breast Cancer Classification on MUSE Images

## Overview

This repository contains the code used in our study on weakly supervised breast cancer classification using microscopy with ultraviolet surface excitation (MUSE) images and an attention-based multiple instance learning (ABMIL) framework.

The pipeline includes:

- patch extraction from original images
- model training, validation, and testing using ABMIL
- prediction evaluation using ROC curves and confusion matrices
- attention map visualization
- Dice score calculation by comparing attention maps with manual annotation maps

## Repository Structure

```text
ABMIL_MUSE_code/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ config.yaml
├─ scripts/
│  ├─ patch_images.py
│  ├─ train_abmil_model.py
│  ├─ evaluate_predictions.py
│  ├─ visualize_attention_maps.py
│  └─ calculate_dice_scores.py
├─ modules/
│  └─ cv_splits.py
├─ example_data/
│  ├─ train.csv
│  ├─ test.csv
│  └─ README.md
└─ outputs/
```

## Dataset

This repository does not include the original dataset used in the study because it contains non-public medical image data.

The expected dataset structure is as follows:

```text
datasets/
├─ train/              # original images for training and validation
├─ test/               # original images for independent testing
├─ train.csv           # metadata for images in train/
├─ test.csv            # metadata for images in test/
└─ test_labeled/       # manual annotation maps for Dice score calculation
```

In `fixed_blocks` mode, additional metadata CSV files may be provided as validation blocks.

### Metadata files

- `train.csv`: metadata for images in `train/`
- `test.csv`: metadata for images in `test/`

Each CSV file contains the following columns:

- `ID`: image file name
- `case`: anonymized case identifier
- `target`: class label

### Notes on the dataset

- In this study, the term **patch** is used for image tiles generated from the original images.
- The `train/` and `test/` folders contain the original images before patch extraction.
- Patch images are generated separately using `patch_images.py`.
- Manual annotation maps in `test_labeled/` are used only for comparison with attention maps in the Dice score analysis.

## Workflow

The analysis pipeline consists of the following five steps.

### 1. Patch extraction

`patch_images.py` extracts patches from the original images in `train/` and `test/` and saves them to user-specified output folders.

- **Input**: original images in `train/` and `test/`
- **Output**: patch images for ABMIL analysis

### 2. Model training, validation, and testing

`train_abmil_model.py` performs model training, validation, and testing using the extracted patch images together with metadata CSV files specified in `config.yaml`.

This script uses `cv_splits.py` as an external module for cross-validation splitting and supports two validation modes:

- **`groupkfold`**: automatically generates case-level cross-validation folds from a single training metadata CSV using `GroupKFold`
- **`fixed_blocks`**: uses pre-defined metadata CSV files as validation folds while preserving their original split structure

The input and output paths are specified in `config.yaml`, including:

- patch image folders for training/validation and testing
- metadata files for cross-validation and independent testing
- output directory for model checkpoints and prediction results

The `config.yaml` file used for each experiment is also saved in the corresponding result folder.

### 3. Prediction evaluation

`evaluate_predictions.py` evaluates the prediction results saved in the result folder and generates performance plots such as:

- ROC curves
- confusion matrices

### 4. Attention map visualization

`visualize_attention_maps.py` generates attention maps for the test images using:

- the original images in `test/`
- `test.csv`
- trained model weights saved in the result folder
- the corresponding `config.yaml`

This script reuses major functions from `train_abmil_model.py`.

### 5. Dice score calculation

`calculate_dice_scores.py` compares the generated attention maps with the manual annotation maps in `test_labeled/` and calculates Dice scores.

### Summary of the pipeline

For model development, validation folds can be created either automatically from a single training metadata file or from pre-defined metadata CSV blocks, depending on the cross-validation mode specified in `config.yaml`.

```text
Original images (train/, test/)
        ↓
patch_images.py
        ↓
Patch images
        ↓
train_abmil_model.py
        ↓
Trained model / predictions / config.yaml
        ↓
evaluate_predictions.py
        ↓
ROC curves / confusion matrices

Trained model + original test images
        ↓
visualize_attention_maps.py
        ↓
Attention maps
        ↓
calculate_dice_scores.py + test_labeled/
        ↓
Dice scores
```

## Scripts

### `scripts/patch_images.py`
Extracts patches from original images for ABMIL analysis.

### `modules/cv_splits.py`
Provides utility functions for cross-validation splitting. This module supports both automatic case-level splitting with `GroupKFold` and concatenation of pre-defined metadata CSV blocks into fixed validation folds. It is imported by `train_abmil_model.py` and is not intended to be run directly.

### `scripts/train_abmil_model.py`
Performs training, validation, and testing of the ABMIL model.

### `scripts/evaluate_predictions.py`
Evaluates prediction results and generates ROC curves and confusion matrices.

### `scripts/visualize_attention_maps.py`
Generates attention maps for test images using trained model outputs.

### `scripts/calculate_dice_scores.py`
Calculates Dice scores between attention maps and manual annotation maps.

## Configuration

The main experiment settings are specified in `config.yaml`.

Typical configuration items include:

- path to training patch images
- path to test patch images
- path to the independent test metadata file
- output directory for experiment results
- training hyperparameters
- cross-validation method and related metadata files

Cross-validation settings are defined under the `cv` section of `config.yaml`.

Two cross-validation modes are supported:

- **`cv.method: groupkfold`**  
  Requires `cv.groupkfold.train_csv`, which is used to generate case-level folds automatically with `GroupKFold`.

- **`cv.method: fixed_blocks`**  
  Requires `cv.fixed_blocks.block_csvs`, which should be a list of metadata CSV files corresponding to pre-defined validation blocks.

In both modes, `data.test_csv` is used as the independent outer test set.

In `fixed_blocks` mode, the configuration is designed for workflows that keep `data.test_csv` as an outer test set and use `cv.fixed_blocks.block_csvs` as inner validation folds.

An example configuration file can be provided separately if needed.

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Example Usage

### 1. Patch extraction

The following command extracts image patches from all subfolders under the input directory and saves them to the specified output directory. In this example, the images are resized to 50% of their original size before patch extraction, split into 224 × 224 pixel patches without overlap, and padded by reflection so that edge regions are also retained.

```bash
python scripts/patch_images.py \
  --input-root /path/to/datasets/original_images \
  --output-root /path/to/datasets/patched_images \
  --resize-scale 0.5 \
  --patch-size 224 \
  --overlap-step 0 \
  --mirroring
```

### 2. Model training, validation, and testing

The following command runs the full ABMIL training pipeline using the settings specified in `config.yaml`. Depending on the value of `cv.method`, the script either generates case-level cross-validation folds automatically from a single training metadata CSV (`groupkfold`) or uses pre-defined metadata CSV blocks as fixed validation folds (`fixed_blocks`). After out-of-fold validation, the trained models are applied to the independent test set specified by `data.test_csv`. The configuration file used for the experiment is also saved in the output directory together with the prediction results and trained model weights.

```bash
python scripts/train_abmil_model.py --config configs/config.yaml
```

#### Example `config.yaml` for `groupkfold`

```yaml
cv:
  method: groupkfold
  groupkfold:
    train_csv: /path/to/metadata/train.csv
```

#### Example `config.yaml` for `fixed_blocks`

```yaml
cv:
  method: fixed_blocks
  fixed_blocks:
    block_csvs:
      - /path/to/metadata/fold0.csv
      - /path/to/metadata/fold1.csv
      - /path/to/metadata/fold2.csv
      - /path/to/metadata/fold3.csv
```

In `fixed_blocks` mode, these files can represent predefined validation blocks prepared for workflows that keep an outer test set separate from inner validation folds.

### 3. Prediction evaluation

The following command evaluates the prediction results saved in the experiment output files. The decision threshold is determined from the primary out-of-fold predictions, then applied to the corresponding test predictions to calculate classification metrics and generate ROC curves and a confusion matrix. Additional test prediction files from repeated runs can also be included for visual comparison.

```bash
python scripts/evaluate_predictions.py \
  --primary-oof /path/to/results/oof.csv \
  --primary-test /path/to/results/test.csv \
  --extra-tests /path/to/results/test_seed23.csv /path/to/results/test_seed83.csv \
  --thr-mode youden \
  --out-png /path/to/results/roc_primary_vs_extra.png \
  --out-metrics-csv /path/to/results/primary_metrics.csv \
  --title "224 × 224-pixel model" \
  --positive-class-name "Cancer" \
  --negative-class-name "Non-cancer"
```

### 4. Attention map visualization

The following command generates attention maps for the independent test images using the trained model weights and the saved experiment configuration. Attention scores are aggregated across cross-validation models, normalized across the test set, and saved as heatmaps and overlay images. The heatmap color mode can be selected as `inferno` or `grayscale`. In this study, `grayscale` maps were also used for Dice score calculation.

```bash
python scripts/visualize_attention_maps.py \
  --test-csv /path/to/metadata/test.csv \
  --image-folder /path/to/original_images/test \
  --experiment-dir /path/to/results/experiment_001 \
  --config-name config_used_exp0.yaml \
  --model-pattern "fold*_0_best_auc_loss.pth" \
  --image-width 2736 \
  --image-height 1824 \
  --patch-size 224 \
  --stride 224 \
  --color grayscale \
  --mirroring
```

### 5. Dice score calculation
The following command calculates Dice scores by comparing the manually annotated masks in test_labeled/ with the corresponding grayscale attention maps. Dice scores are calculated only for positive test images, and the final result is summarized as the mean Dice across images after thresholding the grayscale attention maps.

```bash
python scripts/calculate_dice_scores.py \
  --label-dir /path/to/test_labeled \
  --heatmap-dir /path/to/attention_maps_grayscale \
  --output-csv /path/to/results/dice_scores.csv \
  --threshold-mapping 127.5
```

## Output

Depending on the script, the outputs may include:

- extracted patch images
- trained model weights
- prediction results
- saved `config.yaml`
- ROC curves
- confusion matrices
- attention maps
- Dice scores

## Notes

- The original medical image dataset is not publicly available.
- Any example files included in this repository are provided only to illustrate the expected input format.
- Before public release, please ensure that no private information, institution-specific paths, or authentication tokens remain in the code or configuration files.
- In `fixed_blocks` mode, all samples listed in `cv.fixed_blocks.block_csvs` must correspond to patch files stored under `data.train_dir`.
- `data.test_csv` and `data.test_dir` remain reserved for the independent outer test set in both cross-validation modes.

## Citation

If you use this code in your research, please cite the associated manuscript.

Citation details will be added after publication.

Code archive: https://doi.org/10.5281/zenodo.21257524

## License

Unless otherwise noted, the code in this repository is released under the [MIT License](LICENSE).

The software is provided "as is", without warranty of any kind. Users are responsible for ensuring that their use of the code and any required third-party dependencies complies with the applicable licenses.

## Contact

For questions regarding this repository, please contact:

- Ryuta Nakao
- Department of Pathology and Cell Regulation, Kyoto Prefectural University of Medicine
- E-mail: r-nakao@koto.kpu-m.ac.jp
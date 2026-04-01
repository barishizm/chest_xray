# Chest X-Ray Pneumonia Detection

This project compares two binary classification approaches for chest X-ray images:

- a classical machine learning pipeline built on handcrafted and pixel-based features
- a deep learning pipeline based on transfer learning with `ResNet18`

The target classes are:

- `NORMAL`
- `PNEUMONIA`

The repository also includes a combined runner that executes both pipelines and generates a final comparison chart.

This is an educational and experimental project, not a clinical product. It must not be used for medical decision-making.

## Current Project Status

The current codebase includes:

- `classical_pipeline.py`: trains and evaluates multiple classical models
- `deep_learning_pipeline.py`: trains and evaluates a transfer-learning model with Grad-CAM outputs
- `main.py`: runs both pipelines and builds a comparison report
- `results/`: sample artifacts produced by prior runs

`main.py` compiles in the current version of the project and is usable.

## Repository Layout

```text
chest_xray/
|-- classical_pipeline.py
|-- deep_learning_pipeline.py
|-- main.py
|-- README.md
|-- LICENSE
`-- results/
    |-- classical/
    |   |-- best_model.pkl
    |   |-- classical_metrics_comparison.png
    |   |-- classical_results.png
    |   `-- preprocessing_pipeline.png
    |-- deep_learning/
    |   |-- best_model.pth
    |   |-- dl_evaluation_results.png
    |   |-- gradcam_heatmaps.png
    |   `-- training_history.png
    `-- comparison_chart.png
```

Some additional artifacts may be present in `results/` from earlier experiments.

## Expected Dataset Structure

The scripts expect the common Kaggle-style chest X-ray folder layout:

```text
chest_xray/
|-- train/
|   |-- NORMAL/
|   `-- PNEUMONIA/
|-- val/
|   |-- NORMAL/
|   `-- PNEUMONIA/
`-- test/
    |-- NORMAL/
    `-- PNEUMONIA/
```

Notes:

- the deep learning pipeline uses `train`, `val`, and `test`
- the classical pipeline uses `train` and `test`
- class folder names must remain exactly `NORMAL` and `PNEUMONIA`

## Pipeline Summary

### Classical Pipeline

Implemented in `classical_pipeline.py`.

Workflow:

1. Load grayscale X-ray images
2. Apply preprocessing with resize, CLAHE, and Gaussian denoising
3. Extract combined handcrafted and pixel-based features
4. Standardize features and reduce dimensionality with PCA
5. Train multiple classifiers
6. Optimize the decision threshold
7. Evaluate on the test split and save plots

Preprocessing:

- grayscale loading with OpenCV
- resize to `256 x 256`
- CLAHE contrast enhancement
- Gaussian blur denoising

Feature set:

- GLCM texture features
- histogram and statistical moments
- LBP texture features
- HOG descriptors
- spatial and edge statistics
- flattened pixel features from `128 x 128` images

Models trained:

- `SVM (RBF)`
- `RandomForest`
- `Ensemble (SVM + RandomForest + GradientBoosting)` with soft voting

Dimensionality reduction:

- `StandardScaler`
- `PCA(n_components=300)`

Threshold logic:

- first selects a threshold from 5-fold cross-validation on the training set
- may fall back to a test-set operating point if sensitivity or specificity stays below `0.80`

Saved outputs:

- `results/classical/best_model.pkl`
- `results/classical/classical_results.png`
- `results/classical/classical_metrics_comparison.png`
- `results/classical/preprocessing_pipeline.png`

### Deep Learning Pipeline

Implemented in `deep_learning_pipeline.py`.

Workflow:

1. Build `ImageFolder` datasets and dataloaders
2. Apply augmentation and normalization
3. Fine-tune a pretrained `ResNet18`
4. Save the best checkpoint by validation loss
5. Optimize the decision threshold
6. Evaluate on the test split
7. Generate training plots and Grad-CAM heatmaps

Data pipeline:

- resize to `224 x 224`
- training augmentations:
  - random horizontal flip
  - random rotation
  - random affine translation
  - brightness and contrast jitter
- ImageNet normalization
- `WeightedRandomSampler` on the training set for class imbalance handling

Model and training:

- backbone: `ResNet18`
- pretrained weights: `IMAGENET1K_V1`
- classifier head:
  - `Dropout(0.4)`
  - `Linear(..., 1)`
- loss: `BCEWithLogitsLoss`
- optimizer: `Adam`
- scheduler: `ReduceLROnPlateau`
- default epochs: `20`
- batch size: `32`
- learning rate: `5e-5`
- device: CUDA when available, otherwise CPU

Threshold logic:

- selects an operating threshold from training-set predictions
- tries to satisfy both sensitivity and specificity targets of `0.90`
- if no threshold meets both targets, falls back to a G-mean-optimal threshold

Interpretability:

- Grad-CAM uses the last block of `model.layer4`
- heatmaps are generated for sample test images from both classes

Saved outputs:

- `results/deep_learning/best_model.pth`
- `results/deep_learning/training_history.png`
- `results/deep_learning/dl_evaluation_results.png`
- `results/deep_learning/gradcam_heatmaps.png`

### Combined Runner

Implemented in `main.py`.

It:

1. runs the classical pipeline
2. runs the deep learning pipeline
3. picks the best classical model by F1 score
4. compares it against the deep learning model
5. saves `results/comparison_chart.png`

Comparison thresholds used by the report:

- classical metrics target: `0.80`
- deep learning metrics target: `0.90`

## Metrics Reported

Both pipelines report:

- accuracy
- sensitivity
- specificity
- F1-score
- AUC-ROC
- confusion matrix
- total inference time
- approximate per-image latency

## Installation

The repository does not currently include a `requirements.txt`, so dependencies need to be installed manually.

Suggested environment:

- Python `3.10+`
- `pip`

Install the core dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install numpy matplotlib opencv-python scikit-image scikit-learn seaborn pillow torch torchvision
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Configuration

The current scripts use hardcoded absolute Windows paths. Update these constants before running the project on your machine.

Current defaults in the source code:

- `classical_pipeline.py`
  - `DATA_DIR = c:\Users\baris\Downloads\chest_xray`
  - `OUTPUT_DIR = c:\Users\baris\Documents\Projects\chest_xray\results\classical`
- `deep_learning_pipeline.py`
  - `DATA_DIR = c:\Users\baris\Downloads\chest_xray`
  - `OUTPUT_DIR = c:\Users\baris\Documents\Projects\chest_xray\results\deep_learning`
- `main.py`
  - `RESULTS_DIR = c:\Users\baris\Documents\Projects\chest_xray\results`

If your dataset or project directory lives elsewhere, edit those values first.

## How To Run

Run the classical pipeline:

```bash
python3 classical_pipeline.py
```

Run the deep learning pipeline:

```bash
python3 deep_learning_pipeline.py
```

Run the full comparison workflow:

```bash
python3 main.py
```

If your system uses `python` instead of `python3`, use that command instead.

## Generated Artifacts

### Classical

- `preprocessing_pipeline.png`: preprocessing stages for a sample X-ray
- `classical_results.png`: ROC curves and confusion matrices
- `classical_metrics_comparison.png`: metric comparison across classical models
- `best_model.pkl`: serialized best classical model bundle with scaler, PCA, and threshold

### Deep Learning

- `training_history.png`: train and validation loss and accuracy curves
- `dl_evaluation_results.png`: ROC curve, confusion matrix, and metric bars
- `gradcam_heatmaps.png`: Grad-CAM overlays for representative test images
- `best_model.pth`: best validation-loss checkpoint

### Combined

- `comparison_chart.png`: side-by-side comparison of the best classical model and `ResNet18`

## Notes And Limitations

- paths are hardcoded and should be parameterized for portability
- threshold selection in both pipelines can use test-set information as a fallback, so reported test metrics should be interpreted as experimental rather than strict held-out benchmarks
- there is no dedicated inference script for predicting a single new image
- there is no dependency lockfile yet

## License

This project is released under the MIT License. See `LICENSE` for details.

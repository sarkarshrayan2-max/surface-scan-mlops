# SurfaceScan MLOps

End-to-end deep learning and MLOps project for industrial steel surface defect classification.

This project classifies steel surface defects into six classes using CNN and transfer-learning models. It includes experiment tracking with MLflow, data and model versioning with DVC, Grad-CAM explainability, a Streamlit inference app, Docker support, and GitHub Actions CI.

## Repository

```text
https://github.com/sarkarshrayan2-max/surface-scan-mlops
```

## Problem Statement

Manual inspection of industrial steel surfaces is slow, inconsistent, and difficult to scale. This project builds a computer vision pipeline that can classify steel surface defects from images and provide visual explanations for predictions.

The goal is not only to train a good model, but also to build a production-style ML workflow around it.

## Dataset

Dataset used:

```text
NEU Surface Defect Database
```

Task:

```text
6-class steel surface defect classification
```

Classes:

```python
[
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
```

Original dataset structure:

```text
data/raw/NEU-DET/
├── train/images/<class folders>
└── validation/images/<class folders>
```

Original counts:

```text
Train: 240 images per class = 1440 total
Validation/test: 60 images per class = 360 total
```

One exact duplicate was found and excluded:

```text
data/raw/NEU-DET/train/images/patches/patches_105.jpg
```

Final split:

```text
Train:       1223
Validation: 216
Test:       360
```

The original Kaggle validation folder was kept as the final untouched test set.

## Project Workflow

```text
Data understanding
        |
        v
Duplicate detection and split creation
        |
        v
Baseline CNN experiments
        |
        v
Transfer learning experiments
        |
        v
Champion model selection
        |
        v
Final test evaluation
        |
        v
Grad-CAM error analysis
        |
        v
Streamlit inference app
        |
        v
Docker and CI
```

## Tech Stack

```text
Python 3.11
TensorFlow / Keras
EfficientNetB0
MobileNetV3Small
Custom CNN
MLflow
DVC
Streamlit
Docker
GitHub Actions
uv
pandas
scikit-learn
matplotlib
Pillow
```

## Model Experiments

### Custom CNN Baseline

Run name:

```text
custom-cnn-lr-0.001-dropout-0.40
```

Result:

```text
Validation accuracy: 27.78%
Validation macro F1: 14.16%
```

Conclusion:

```text
The first custom CNN severely overfit and was kept as a failed baseline.
```

### Regularized Custom CNN

Run name:

```text
custom-cnn-lr-0.0001-dropout-0.50
```

Result:

```text
Validation accuracy: 85.19%
Validation macro precision: 87.38%
Validation macro recall: 85.19%
Validation macro F1: 84.61%
```

Conclusion:

```text
Regularization, lower learning rate, reduced capacity, and milder augmentation improved generalization.
```

### MobileNetV3Small Transfer Learning

Run name:

```text
mobilenetv3small-frozen-lr-0.001
```

Result:

```text
Validation accuracy: 99.07%
Validation macro precision: 99.12%
Validation macro recall: 99.07%
Validation macro F1: 99.07%
```

### EfficientNetB0 Transfer Learning

Run name:

```text
efficientnetb0-frozen-lr-0.001
```

Result:

```text
Validation accuracy: 99.54%
Validation macro precision: 99.55%
Validation macro recall: 99.54%
Validation macro F1: 99.54%
```

EfficientNetB0 was selected as the champion model.

## Final Test Results

Champion model:

```text
EfficientNetB0 frozen transfer-learning model
```

Final test performance:

```text
Test accuracy: 95.56%
Test macro precision: 95.57%
Test macro recall: 95.56%
Test macro F1: 95.50%
Total test images: 360
Correct predictions: 344
Wrong predictions: 16
Total inference seconds: 2.0154
Average inference: 5.60 ms/image on Colab GPU
```

## Model Comparison

| Model | Validation Accuracy | Validation Macro F1 | Notes |
|---|---:|---:|---|
| Custom CNN initial | 27.78% | 14.16% | Severe overfitting |
| Custom CNN regularized | 85.19% | 84.61% | Good baseline |
| MobileNetV3Small frozen | 99.07% | 99.07% | Strong transfer-learning model |
| EfficientNetB0 frozen | 99.54% | 99.54% | Champion model |

## Error Analysis

Grad-CAM and prediction-level error analysis were performed on the champion EfficientNetB0 model.

Wrong predictions:

```text
Total test images: 360
Correct predictions: 344
Wrong predictions: 16
```

Main confusion pairs:

```text
inclusion -> scratches:              7
scratches -> rolled-in_scale:        3
scratches -> inclusion:              3
inclusion -> pitted_surface:         2
pitted_surface -> rolled-in_scale:   1
```

Interpretation:

```text
Most errors occurred between visually similar defect classes.
Inclusion defects with line-like regions were often classified as scratches.
Some scratch samples were confused with rolled-in scale when the model focused on broader texture regions.
```

Grad-CAM layer used:

```text
top_conv
```

Generated reports:

```text
reports/gradcam_error_analysis/test_error_predictions.csv
reports/gradcam_error_analysis/confusion_pairs.csv
reports/gradcam_error_analysis/error_analysis_summary.json
reports/gradcam_error_analysis/top_error_gradcams.png
```

## MLflow Tracking

MLflow was used to track:

```text
Parameters
Training metrics
Validation metrics
Final test metrics
Confusion matrices
Training curves
Model artifacts
```

Tracking backend:

```text
SQLite backend: mlflow.db
```

The MLflow database is ignored by Git:

```text
mlflow.db
mlruns/
mlartifacts/
```

To open MLflow locally:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

## DVC

DVC is used to track dataset splits and the champion model artifact.

DVC-tracked items:

```text
data/interim/
artifacts/efficientnetb0/efficientnetb0_frozen_best.keras
```

Important DVC files:

```text
data/interim.dvc
artifacts/efficientnetb0/efficientnetb0_frozen_best.keras.dvc
dvc.yaml
dvc.lock
params.yaml
```

The raw dataset and actual model binary are not committed directly to Git.

Ignored files include:

```text
data/raw/
artifacts/**/*.keras
artifacts/**/*.h5
mlflow.db
mlruns/
mlartifacts/
```

If you have access to the configured DVC remote:

```powershell
uv run dvc pull
```

Check DVC status:

```powershell
uv run dvc status
```

Run validation and champion evaluation stages:

```powershell
uv run dvc repro validate_data
uv run dvc repro evaluate_champion
```

View DVC metrics:

```powershell
uv run dvc metrics show
```

## Streamlit App

The Streamlit app allows users to upload a steel surface image and get:

```text
Predicted defect class
Confidence score
Class probabilities
Grad-CAM explanation
```

Run locally:

```powershell
uv run streamlit run app/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

Example test image:

```text
data/raw/NEU-DET/validation/images/scratches/scratches_241.jpg
```

## Docker

The project includes Docker support for running the Streamlit inference app.

Build the image:

```powershell
docker build -t surface-scan-mlops .
```

Run the container:

```powershell
docker run -p 8501:8501 surface-scan-mlops
```

Open:

```text
http://localhost:8501
```

The Docker image expects the champion model to exist at:

```text
artifacts/efficientnetb0/efficientnetb0_frozen_best.keras
```

If the model is missing, restore it through DVC or place the artifact manually at the expected path.

## GitHub Actions CI

The project includes GitHub Actions CI for basic project checks.

CI checks:

```text
Dependency installation with uv
Python syntax compilation
Core module import checks
Model factory check
```

Workflow path:

```text
.github/workflows/ci.yml
```

## Project Structure

```text
surface-scan-mlops/
├── app/
│   └── streamlit_app.py
├── artifacts/
│   └── efficientnetb0/
│       └── efficientnetb0_frozen_best.keras
├── configs/
├── data/
│   ├── .gitignore
│   └── interim/
│       ├── excluded_duplicates.csv
│       └── splits/
│           ├── train.csv
│           ├── validation.csv
│           └── test.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_baseline_cnn_mlflow.ipynb
│   ├── 03_transfer_learning_mlflow.ipynb
│   └── 04_gradcam_error_analysis.ipynb
├── reports/
│   ├── baseline_cnn/
│   ├── baseline_cnn_regularized/
│   ├── efficientnetb0/
│   ├── mobilenetv3small/
│   ├── champion_efficientnetb0/
│   ├── gradcam_error_analysis/
│   ├── evaluation/
│   └── model_comparison_validation.csv
├── src/
│   ├── __init__.py
│   ├── check_mlflow.py
│   ├── data_validation.py
│   ├── evaluate.py
│   ├── gradcam.py
│   ├── model_factory.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── tests/
├── .dvc/
├── .dvcignore
├── .dockerignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── MODEL_CARD.md
├── params.yaml
├── pyproject.toml
├── README.md
└── uv.lock
```

## Important Source Files

### `src/predict.py`

Loads the champion EfficientNetB0 model, preprocesses uploaded images, and returns:

```text
Predicted class
Confidence score
Class probabilities
```

### `src/gradcam.py`

Generates Grad-CAM heatmaps using the champion model's `top_conv` layer.

### `src/data_validation.py`

Validates:

```text
Raw dataset structure
Split CSV files
Class labels
Duplicate image paths
Missing files
Champion model artifact
```

### `src/preprocess.py`

Builds TensorFlow datasets from split CSV files.

The model was trained without manual `/255` scaling, so images are kept in the `0-255` pixel range.

### `src/model_factory.py`

Builds supported model architectures:

```text
custom_cnn
efficientnetb0
mobilenetv3small
```

### `src/train.py`

Training script with MLflow logging.

Example smoke test:

```powershell
uv run python src/train.py --model-name custom_cnn --epochs 1 --batch-size 16
```

### `src/evaluate.py`

Evaluates a trained model and generates:

```text
Metrics JSON
Predictions CSV
Classification report
Confusion matrix CSV
Confusion matrix PNG
Error predictions CSV
```

Example:

```powershell
uv run python src/evaluate.py `
  --model-path artifacts/efficientnetb0/efficientnetb0_frozen_best.keras `
  --split-csv data/interim/splits/test.csv `
  --split-name test `
  --output-dir reports/evaluation
```

## Setup

Clone the repository:

```powershell
git clone https://github.com/sarkarshrayan2-max/surface-scan-mlops.git
cd surface-scan-mlops
```

Install dependencies:

```powershell
uv sync
```

Restore DVC-tracked artifacts if remote access is available:

```powershell
uv run dvc pull
```

Check project status:

```powershell
git status
uv run dvc status
```

## Run Inference Locally

Start Streamlit:

```powershell
uv run streamlit run app/streamlit_app.py
```

Upload a test image from:

```text
data/raw/NEU-DET/validation/images/
```

The app will display the predicted class, confidence score, probabilities, and Grad-CAM explanation.

## Run Evaluation

```powershell
uv run python src/evaluate.py `
  --model-path artifacts/efficientnetb0/efficientnetb0_frozen_best.keras `
  --split-csv data/interim/splits/test.csv `
  --split-name test `
  --output-dir reports/evaluation
```

## Run DVC Pipeline

Run data validation:

```powershell
uv run dvc repro validate_data
```

Run champion evaluation:

```powershell
uv run dvc repro evaluate_champion
```

Show metrics:

```powershell
uv run dvc metrics show
```

## Run Docker App

Build:

```powershell
docker build -t surface-scan-mlops .
```

Run:

```powershell
docker run -p 8501:8501 surface-scan-mlops
```

Open:

```text
http://localhost:8501
```

## Files Not Pushed to Git

The following are intentionally not committed:

```text
data/raw/
data/processed/
artifacts/**/*.keras
artifacts/**/*.h5
artifacts/**/*.joblib
mlflow.db
mlruns/
mlartifacts/
kaggle.json
.venv/
__pycache__/
```

The champion model is handled by DVC, not normal Git.

## Current Status

Completed:

```text
Data understanding
Dataset split creation
Duplicate detection
Custom CNN experiments
Transfer-learning experiments
Champion model selection
Final test evaluation
Grad-CAM error analysis
Prediction utility
Streamlit app
Grad-CAM app integration
Data validation utility
Preprocessing utility
Model factory
Training script
Evaluation script
DVC pipeline
Docker support
GitHub Actions CI
```

Remaining possible improvements:

```text
Deploy Streamlit app
Add Docker image publishing
Add full DVC remote accessible to collaborators
Add unit tests
Add model monitoring simulation
Add batch inference mode
Add API endpoint using FastAPI
Add more robust CI with DVC pull and Docker build
```

## Summary

SurfaceScan MLOps demonstrates a complete machine learning project lifecycle:

```text
Computer vision modeling
Experiment tracking
Artifact versioning
Model evaluation
Explainability
Interactive inference
Containerization
CI integration
```

The final champion model is an EfficientNetB0 transfer-learning model with 95.56% test accuracy and Grad-CAM explanations for model interpretability.

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocess import (  # noqa: E402
    BATCH_SIZE,
    CLASS_NAMES,
    DEFAULT_TEST_CSV,
    create_dataset_from_dataframe,
    load_split_dataframe,
)


DEFAULT_MODEL_PATH = Path("artifacts/efficientnetb0/efficientnetb0_frozen_best.keras")
DEFAULT_OUTPUT_DIR = Path("reports/evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SurfaceScan classification model."
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained .keras model.",
    )

    parser.add_argument(
        "--split-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
        help="Path to split CSV for evaluation.",
    )

    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="Name of evaluated split. Example: validation or test.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for evaluation.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where evaluation reports will be saved.",
    )

    return parser.parse_args()


def load_evaluation_model(model_path: Path) -> tf.keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at: {model_path}. "
            "Run `dvc pull` or place the trained model artifact locally."
        )

    model = tf.keras.models.load_model(model_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def predict_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, float]:
    true_labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    start_time = time.perf_counter()

    for image_batch, label_batch in dataset:
        batch_probabilities = model.predict(
            image_batch,
            verbose=0,
        )

        probabilities.append(batch_probabilities)
        true_labels.append(label_batch.numpy())

    total_inference_seconds = time.perf_counter() - start_time

    y_true = np.concatenate(true_labels).astype(int)
    y_probabilities = np.concatenate(probabilities)

    return y_true, y_probabilities, total_inference_seconds


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    total_inference_seconds: float,
) -> dict[str, float]:
    total_images = len(y_true)

    average_inference_seconds = (
        total_inference_seconds / total_images if total_images > 0 else 0.0
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "total_images": int(total_images),
        "total_inference_seconds": float(total_inference_seconds),
        "average_inference_seconds_per_image": float(average_inference_seconds),
        "average_inference_ms_per_image": float(average_inference_seconds * 1000),
    }

    return metrics


def save_json(
    data: dict,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def save_predictions(
    dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probabilities: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    confidence = np.max(y_probabilities, axis=1)

    predictions_dataframe = pd.DataFrame(
        {
            "image_path": dataframe["image_path"].astype(str).to_numpy(),
            "true_label_index": y_true,
            "true_label": [CLASS_NAMES[index] for index in y_true],
            "predicted_label_index": y_pred,
            "predicted_label": [CLASS_NAMES[index] for index in y_pred],
            "confidence": confidence,
            "correct": y_true == y_pred,
        }
    )

    probabilities_dataframe = pd.DataFrame(
        y_probabilities,
        columns=[f"prob_{class_name}" for class_name in CLASS_NAMES],
    )

    predictions_dataframe = pd.concat(
        [
            predictions_dataframe,
            probabilities_dataframe,
        ],
        axis=1,
    )

    predictions_dataframe.to_csv(
        output_path,
        index=False,
    )

    return predictions_dataframe


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_dataframe = pd.DataFrame(report).transpose()
    report_dataframe.to_csv(output_path)

    return report_dataframe


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_csv_path: Path,
    output_png_path: Path,
) -> pd.DataFrame:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    matrix_dataframe.to_csv(output_csv_path)

    figure, axis = plt.subplots(
        figsize=(10, 8),
    )

    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)

    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")

    axis.set_xticks(range(len(CLASS_NAMES)))
    axis.set_yticks(range(len(CLASS_NAMES)))

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )
    axis.set_yticklabels(CLASS_NAMES)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(output_png_path, dpi=200)
    plt.close(figure)

    return matrix_dataframe


def save_error_predictions(
    predictions_dataframe: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    errors_dataframe = predictions_dataframe[
        predictions_dataframe["correct"] == False  # noqa: E712
    ].copy()

    errors_dataframe = errors_dataframe.sort_values(
        by="confidence",
        ascending=False,
    )

    errors_dataframe.to_csv(
        output_path,
        index=False,
    )

    return errors_dataframe


def evaluate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_evaluation_model(args.model_path)

    dataframe = load_split_dataframe(args.split_csv)

    dataset = create_dataset_from_dataframe(
        dataframe=dataframe,
        batch_size=args.batch_size,
        shuffle=False,
    )

    y_true, y_probabilities, total_inference_seconds = predict_dataset(
        model=model,
        dataset=dataset,
    )

    y_pred = np.argmax(
        y_probabilities,
        axis=1,
    )

    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        total_inference_seconds=total_inference_seconds,
    )

    metrics["split_name"] = args.split_name
    metrics["model_path"] = str(args.model_path)
    metrics["split_csv"] = str(args.split_csv)

    metrics_path = args.output_dir / f"{args.split_name}_metrics.json"
    predictions_path = args.output_dir / f"{args.split_name}_predictions.csv"
    classification_report_path = (
        args.output_dir / f"{args.split_name}_classification_report.csv"
    )
    confusion_matrix_csv_path = args.output_dir / f"{args.split_name}_confusion_matrix.csv"
    confusion_matrix_png_path = args.output_dir / f"{args.split_name}_confusion_matrix.png"
    errors_path = args.output_dir / f"{args.split_name}_error_predictions.csv"

    save_json(
        data=metrics,
        output_path=metrics_path,
    )

    predictions_dataframe = save_predictions(
        dataframe=dataframe,
        y_true=y_true,
        y_pred=y_pred,
        y_probabilities=y_probabilities,
        output_path=predictions_path,
    )

    save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_path=classification_report_path,
    )

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        output_csv_path=confusion_matrix_csv_path,
        output_png_path=confusion_matrix_png_path,
    )

    errors_dataframe = save_error_predictions(
        predictions_dataframe=predictions_dataframe,
        output_path=errors_path,
    )

    print("=" * 80)
    print("SurfaceScan evaluation complete")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Split: {args.split_name}")
    print(f"Images: {metrics['total_images']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Total inference seconds: {metrics['total_inference_seconds']:.4f}")
    print(
        "Average inference ms/image: "
        f"{metrics['average_inference_ms_per_image']:.2f}"
    )
    print(f"Wrong predictions: {len(errors_dataframe)}")
    print("=" * 80)
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")
    print(f"Classification report saved to: {classification_report_path}")
    print(f"Confusion matrix CSV saved to: {confusion_matrix_csv_path}")
    print(f"Confusion matrix PNG saved to: {confusion_matrix_png_path}")
    print(f"Error predictions saved to: {errors_path}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate(parse_args())
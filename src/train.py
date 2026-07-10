from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import pandas as pd
import tensorflow as tf


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model_factory import build_model, compile_model  # noqa: E402
from src.preprocess import (  # noqa: E402
    BATCH_SIZE,
    CLASS_NAMES,
    DEFAULT_TEST_CSV,
    DEFAULT_TRAIN_CSV,
    DEFAULT_VALIDATION_CSV,
    build_datasets_from_splits,
    load_split_dataframe,
)


DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "surface-scan-training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a SurfaceScan defect-classification model."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="efficientnetb0",
        choices=["custom_cnn", "efficientnetb0", "mobilenetv3small"],
        help="Model architecture to train.",
    )

    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Unfreeze the pretrained backbone for transfer-learning models.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help="Path to train split CSV.",
    )

    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=DEFAULT_VALIDATION_CSV,
        help="Path to validation split CSV.",
    )

    parser.add_argument(
        "--test-csv",
        type=Path,
        default=DEFAULT_TEST_CSV,
        help="Path to test split CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training"),
        help="Directory for trained model artifacts.",
    )

    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports/training"),
        help="Directory for training reports.",
    )

    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional MLflow run name.",
    )

    return parser.parse_args()


def get_split_sizes(
    train_csv: Path,
    validation_csv: Path,
    test_csv: Path,
) -> dict[str, int]:
    return {
        "train_size": len(load_split_dataframe(train_csv)),
        "validation_size": len(load_split_dataframe(validation_csv)),
        "test_size": len(load_split_dataframe(test_csv)),
    }


def make_run_name(
    model_name: str,
    learning_rate: float,
    train_backbone: bool,
) -> str:
    backbone_status = "finetuned" if train_backbone else "frozen"

    return f"{model_name}-{backbone_status}-lr-{learning_rate}"


def save_training_history(
    history: tf.keras.callbacks.History,
    output_path: Path,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    history_dataframe = pd.DataFrame(history.history)
    history_dataframe.to_csv(output_path, index=False)

    return history_dataframe


def save_class_names(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "class_names": CLASS_NAMES,
            },
            file,
            indent=2,
        )


def train(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    split_sizes = get_split_sizes(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        test_csv=args.test_csv,
    )

    datasets = build_datasets_from_splits(
        train_csv=args.train_csv,
        validation_csv=args.validation_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
    )

    model = build_model(
        model_name=args.model_name,
        train_backbone=args.train_backbone,
    )

    model = compile_model(
        model=model,
        learning_rate=args.learning_rate,
    )

    run_name = args.run_name or make_run_name(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        train_backbone=args.train_backbone,
    )

    model_output_path = args.output_dir / f"{run_name}_best.keras"
    history_output_path = args.reports_dir / f"{run_name}_history.csv"
    class_names_output_path = args.output_dir / "class_names.json"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_output_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_name": args.model_name,
                "train_backbone": args.train_backbone,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "num_classes": len(CLASS_NAMES),
                **split_sizes,
            }
        )

        history = model.fit(
            datasets.train,
            validation_data=datasets.validation,
            epochs=args.epochs,
            callbacks=callbacks,
        )

        history_dataframe = save_training_history(
            history=history,
            output_path=history_output_path,
        )

        save_class_names(class_names_output_path)

        validation_loss, validation_accuracy = model.evaluate(
            datasets.validation,
            verbose=0,
        )

        test_loss, test_accuracy = model.evaluate(
            datasets.test,
            verbose=0,
        )

        mlflow.log_metrics(
            {
                "validation_loss": float(validation_loss),
                "validation_accuracy": float(validation_accuracy),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "best_validation_accuracy": float(
                    history_dataframe["val_accuracy"].max()
                ),
            }
        )

        mlflow.log_artifact(str(history_output_path))
        mlflow.log_artifact(str(class_names_output_path))

        if model_output_path.exists():
            mlflow.log_artifact(str(model_output_path))

        print("=" * 80)
        print("Training complete")
        print("=" * 80)
        print(f"Run name: {run_name}")
        print(f"Model saved to: {model_output_path}")
        print(f"History saved to: {history_output_path}")
        print(f"Validation accuracy: {validation_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    train(parse_args())
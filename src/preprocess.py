from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tensorflow as tf


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_INDEX = {
    class_name: index for index, class_name in enumerate(CLASS_NAMES)
}

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

SPLITS_DIR = Path("data/interim/splits")

DEFAULT_TRAIN_CSV = SPLITS_DIR / "train.csv"
DEFAULT_VALIDATION_CSV = SPLITS_DIR / "validation.csv"
DEFAULT_TEST_CSV = SPLITS_DIR / "test.csv"


@dataclass
class DatasetBundle:
    train: tf.data.Dataset
    validation: tf.data.Dataset
    test: tf.data.Dataset


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str:
    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not find any of these columns: {candidates}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def resolve_image_path(raw_path: str | Path) -> str:
    image_path = Path(raw_path)

    if image_path.exists():
        return str(image_path)

    project_relative_path = ROOT_DIR / image_path

    if project_relative_path.exists():
        return str(project_relative_path)

    raise FileNotFoundError(
        f"Image path not found: {raw_path}. "
        "Check whether data/raw/NEU-DET exists locally."
    )


def load_split_dataframe(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)

    if dataframe.empty:
        raise ValueError(f"Split CSV is empty: {csv_path}")

    path_column = find_column(
        dataframe,
        ["image_path", "path", "filepath", "file_path", "filename"],
    )

    label_column = find_column(
        dataframe,
        ["label", "class", "class_name", "target"],
    )

    dataframe = dataframe[[path_column, label_column]].copy()
    dataframe.columns = ["image_path", "label"]

    dataframe["image_path"] = dataframe["image_path"].apply(resolve_image_path)
    dataframe["label"] = dataframe["label"].astype(str)

    invalid_labels = sorted(set(dataframe["label"]) - set(CLASS_NAMES))

    if invalid_labels:
        raise ValueError(
            f"Invalid labels found in {csv_path}: {invalid_labels}. "
            f"Expected labels: {CLASS_NAMES}"
        )

    dataframe["label_index"] = dataframe["label"].map(CLASS_TO_INDEX)

    return dataframe


def decode_and_resize_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
    )

    image = tf.cast(image, tf.float32)

    # Important:
    # The EfficientNetB0 model was trained without manual /255 scaling.
    # Keep image pixels in 0-255 range.
    return image, label


def create_dataset_from_dataframe(
    dataframe: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> tf.data.Dataset:
    image_paths = dataframe["image_path"].astype(str).to_numpy()
    labels = dataframe["label_index"].astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            labels,
        )
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        decode_and_resize_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def create_dataset_from_csv(
    csv_path: str | Path,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
) -> tf.data.Dataset:
    dataframe = load_split_dataframe(csv_path)

    return create_dataset_from_dataframe(
        dataframe=dataframe,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def build_datasets_from_splits(
    train_csv: str | Path = DEFAULT_TRAIN_CSV,
    validation_csv: str | Path = DEFAULT_VALIDATION_CSV,
    test_csv: str | Path = DEFAULT_TEST_CSV,
    batch_size: int = BATCH_SIZE,
) -> DatasetBundle:
    train_dataframe = load_split_dataframe(train_csv)
    validation_dataframe = load_split_dataframe(validation_csv)
    test_dataframe = load_split_dataframe(test_csv)

    train_dataset = create_dataset_from_dataframe(
        dataframe=train_dataframe,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_dataset = create_dataset_from_dataframe(
        dataframe=validation_dataframe,
        batch_size=batch_size,
        shuffle=False,
    )

    test_dataset = create_dataset_from_dataframe(
        dataframe=test_dataframe,
        batch_size=batch_size,
        shuffle=False,
    )

    return DatasetBundle(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset,
    )


def inspect_dataset(
    dataset: tf.data.Dataset,
    name: str,
) -> None:
    for image_batch, label_batch in dataset.take(1):
        print(f"{name} image batch shape:", image_batch.shape)
        print(f"{name} label batch shape:", label_batch.shape)
        print(f"{name} image dtype:", image_batch.dtype)
        print(f"{name} label dtype:", label_batch.dtype)
        print(f"{name} min pixel:", float(tf.reduce_min(image_batch).numpy()))
        print(f"{name} max pixel:", float(tf.reduce_max(image_batch).numpy()))
        print(f"{name} labels:", label_batch.numpy()[:10])


if __name__ == "__main__":
    datasets = build_datasets_from_splits(
        batch_size=32,
    )

    inspect_dataset(
        dataset=datasets.train,
        name="train",
    )

    inspect_dataset(
        dataset=datasets.validation,
        name="validation",
    )

    inspect_dataset(
        dataset=datasets.test,
        name="test",
    )
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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


RAW_DATA_DIR = Path("data/raw/NEU-DET")
INTERIM_DATA_DIR = Path("data/interim")
SPLITS_DIR = Path("data/interim/splits")
MODEL_PATH = Path("artifacts/efficientnetb0/efficientnetb0_frozen_best.keras")

SPLIT_FILES = {
    "train": SPLITS_DIR / "train.csv",
    "validation": SPLITS_DIR / "validation.csv",
    "test": SPLITS_DIR / "test.csv",
}

EXCLUDED_DUPLICATES_PATH = INTERIM_DATA_DIR / "excluded_duplicates.csv"


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str


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


def validate_path_exists(path: Path, name: str) -> ValidationResult:
    if path.exists():
        return ValidationResult(
            name=name,
            passed=True,
            message=f"Found: {path}",
        )

    return ValidationResult(
        name=name,
        passed=False,
        message=f"Missing: {path}",
    )


def validate_raw_data_structure() -> list[ValidationResult]:
    results: list[ValidationResult] = []

    results.append(
        validate_path_exists(
            RAW_DATA_DIR,
            "raw_data_root",
        )
    )

    expected_dirs = [
        RAW_DATA_DIR / "train" / "images",
        RAW_DATA_DIR / "validation" / "images",
    ]

    for directory in expected_dirs:
        results.append(
            validate_path_exists(
                directory,
                f"raw_directory_{directory.as_posix()}",
            )
        )

    for split_name in ["train", "validation"]:
        for class_name in CLASS_NAMES:
            class_dir = RAW_DATA_DIR / split_name / "images" / class_name

            if class_dir.exists():
                image_count = len(
                    list(class_dir.glob("*.jpg"))
                    + list(class_dir.glob("*.jpeg"))
                    + list(class_dir.glob("*.png"))
                )

                passed = image_count > 0
                message = f"{class_dir}: {image_count} images"
            else:
                passed = False
                message = f"Missing class folder: {class_dir}"

            results.append(
                ValidationResult(
                    name=f"raw_{split_name}_{class_name}",
                    passed=passed,
                    message=message,
                )
            )

    return results


def validate_split_file(
    split_name: str,
    split_path: Path,
) -> list[ValidationResult]:
    results: list[ValidationResult] = []

    if not split_path.exists():
        return [
            ValidationResult(
                name=f"{split_name}_split_exists",
                passed=False,
                message=f"Missing split file: {split_path}",
            )
        ]

    dataframe = pd.read_csv(split_path)

    results.append(
        ValidationResult(
            name=f"{split_name}_split_exists",
            passed=True,
            message=f"Found {split_path} with {len(dataframe)} rows",
        )
    )

    if dataframe.empty:
        results.append(
            ValidationResult(
                name=f"{split_name}_split_not_empty",
                passed=False,
                message=f"{split_path} is empty",
            )
        )
        return results

    try:
        path_column = find_column(
            dataframe,
            ["image_path", "path", "filepath", "file_path", "filename"],
        )
        label_column = find_column(
            dataframe,
            ["label", "class", "class_name", "target"],
        )
    except ValueError as error:
        results.append(
            ValidationResult(
                name=f"{split_name}_required_columns",
                passed=False,
                message=str(error),
            )
        )
        return results

    labels = set(dataframe[label_column].astype(str).unique())
    expected_labels = set(CLASS_NAMES)

    invalid_labels = sorted(labels - expected_labels)
    missing_labels = sorted(expected_labels - labels)

    results.append(
        ValidationResult(
            name=f"{split_name}_labels_valid",
            passed=len(invalid_labels) == 0,
            message=(
                "All labels are valid"
                if len(invalid_labels) == 0
                else f"Invalid labels found: {invalid_labels}"
            ),
        )
    )

    results.append(
        ValidationResult(
            name=f"{split_name}_all_classes_present",
            passed=len(missing_labels) == 0,
            message=(
                "All classes are present"
                if len(missing_labels) == 0
                else f"Missing classes: {missing_labels}"
            ),
        )
    )

    duplicate_count = int(dataframe[path_column].duplicated().sum())

    results.append(
        ValidationResult(
            name=f"{split_name}_no_duplicate_paths",
            passed=duplicate_count == 0,
            message=f"Duplicate image paths: {duplicate_count}",
        )
    )

    missing_files = []

    for raw_path in dataframe[path_column].astype(str).tolist():
        image_path = Path(raw_path)

        if not image_path.is_absolute():
            image_path = Path(raw_path)

        if not image_path.exists():
            missing_files.append(raw_path)

    results.append(
        ValidationResult(
            name=f"{split_name}_files_exist",
            passed=len(missing_files) == 0,
            message=(
                "All image paths exist locally"
                if len(missing_files) == 0
                else f"{len(missing_files)} files missing locally. First missing: {missing_files[:3]}"
            ),
        )
    )

    class_counts = dataframe[label_column].value_counts().to_dict()

    results.append(
        ValidationResult(
            name=f"{split_name}_class_distribution",
            passed=True,
            message=f"Class counts: {class_counts}",
        )
    )

    return results


def validate_splits() -> list[ValidationResult]:
    results: list[ValidationResult] = []

    for split_name, split_path in SPLIT_FILES.items():
        results.extend(
            validate_split_file(
                split_name=split_name,
                split_path=split_path,
            )
        )

    return results


def validate_excluded_duplicates() -> list[ValidationResult]:
    results: list[ValidationResult] = []

    if not EXCLUDED_DUPLICATES_PATH.exists():
        return [
            ValidationResult(
                name="excluded_duplicates_exists",
                passed=False,
                message=f"Missing file: {EXCLUDED_DUPLICATES_PATH}",
            )
        ]

    dataframe = pd.read_csv(EXCLUDED_DUPLICATES_PATH)

    results.append(
        ValidationResult(
            name="excluded_duplicates_exists",
            passed=True,
            message=f"Found {EXCLUDED_DUPLICATES_PATH} with {len(dataframe)} rows",
        )
    )

    return results


def validate_model_artifact() -> list[ValidationResult]:
    return [
        validate_path_exists(
            MODEL_PATH,
            "champion_model_artifact",
        )
    ]


def print_results(results: list[ValidationResult]) -> None:
    passed_count = sum(result.passed for result in results)
    failed_count = len(results) - passed_count

    print("=" * 80)
    print("SurfaceScan data validation report")
    print("=" * 80)
    print(f"Total checks: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("=" * 80)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        print(f"       {result.message}")

    print("=" * 80)

    if failed_count > 0:
        raise SystemExit(
            f"Data validation failed with {failed_count} failed checks."
        )


def run_validation() -> list[ValidationResult]:
    results: list[ValidationResult] = []

    results.extend(validate_raw_data_structure())
    results.extend(validate_splits())
    results.extend(validate_excluded_duplicates())
    results.extend(validate_model_artifact())

    return results


if __name__ == "__main__":
    validation_results = run_validation()
    print_results(validation_results)
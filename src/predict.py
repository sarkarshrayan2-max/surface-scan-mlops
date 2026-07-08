from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IMAGE_SIZE = (224, 224)

DEFAULT_MODEL_PATH = Path(
    "artifacts/efficientnetb0/efficientnetb0_frozen_best.keras"
)


@dataclass
class PredictionResult:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. "
            "Run `dvc pull` or place the champion model artifact locally."
        )

    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)

    image_batch = np.expand_dims(image_array, axis=0)

    return image_batch


def predict_image(
    image: Image.Image,
    model: tf.keras.Model,
) -> PredictionResult:
    image_batch = preprocess_image(image)

    probabilities = model.predict(image_batch, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    class_probabilities = {
        class_name: float(probabilities[index])
        for index, class_name in enumerate(CLASS_NAMES)
    }

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=class_probabilities,
    )


def predict_from_path(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> PredictionResult:
    model = load_model(model_path)

    image = Image.open(image_path)

    return predict_image(
        image=image,
        model=model,
    )


if __name__ == "__main__":
    test_image_path = Path(
        "data/raw/NEU-DET/validation/images/scratches/scratches_241.jpg"
    )

    result = predict_from_path(test_image_path)

    print("Predicted class:", result.predicted_class)
    print("Confidence:", round(result.confidence, 4))
    print("Class probabilities:")

    for class_name, probability in result.class_probabilities.items():
        print(f"  {class_name}: {probability:.4f}")
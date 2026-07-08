from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    IMAGE_SIZE,
    load_model,
    preprocess_image,
)


DEFAULT_GRADCAM_LAYER = "top_conv"


def call_layer_safely(
    layer: tf.keras.layers.Layer,
    x: tf.Tensor,
) -> tf.Tensor:
    """
    Some Keras layers accept training=False and some do not.
    This helper prevents TypeError when manually passing tensors through layers.
    """
    if isinstance(layer, tf.keras.layers.InputLayer):
        return x

    try:
        return layer(x, training=False)
    except TypeError:
        return layer(x)


def find_backbone_with_layer(
    model: tf.keras.Model,
    target_layer_name: str = DEFAULT_GRADCAM_LAYER,
) -> tuple[int, tf.keras.Model]:
    """
    Finds the nested transfer-learning backbone that contains the Grad-CAM layer.
    For this project, the EfficientNetB0 backbone contains the `top_conv` layer.
    """
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                layer.get_layer(target_layer_name)
                return index, layer
            except ValueError:
                continue

    raise ValueError(
        f"Could not find a nested backbone containing layer `{target_layer_name}`. "
        "Check your model architecture with `model.summary()`."
    )


def make_gradcam_heatmap(
    image: Image.Image,
    model: tf.keras.Model,
    target_layer_name: str = DEFAULT_GRADCAM_LAYER,
    class_index: int | None = None,
) -> tuple[np.ndarray, int]:
    """
    Creates a normalized Grad-CAM heatmap for a PIL image.

    Returns:
        heatmap: 2D numpy array in range 0-1.
        class_index: class index used for Grad-CAM.
    """
    image_batch = preprocess_image(image)

    backbone_index, backbone = find_backbone_with_layer(
        model=model,
        target_layer_name=target_layer_name,
    )

    x = image_batch

    # Pass through layers before the EfficientNetB0 backbone.
    for layer in model.layers[:backbone_index]:
        x = call_layer_safely(layer, x)

    target_layer = backbone.get_layer(target_layer_name)

    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[
            target_layer.output,
            backbone.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, backbone_output = backbone_grad_model(
            x,
            training=False,
        )

        predictions = backbone_output

        # Pass through layers after the EfficientNetB0 backbone.
        for layer in model.layers[backbone_index + 1 :]:
            predictions = call_layer_safely(layer, predictions)

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]).numpy())

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Could not compute gradients for Grad-CAM. "
            "Check whether the selected Grad-CAM layer is connected to the output."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = float(tf.reduce_max(heatmap).numpy())

    if max_value <= 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / max_value

    return heatmap.numpy(), class_index


def overlay_heatmap_on_image(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> Image.Image:
    """
    Overlays the Grad-CAM heatmap on the original image.
    """
    image = image.convert("RGB").resize(IMAGE_SIZE)

    heatmap_uint8 = np.uint8(255 * heatmap)

    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap_uint8)

    colored_heatmap = np.uint8(colored_heatmap[:, :, :3] * 255)

    heatmap_image = Image.fromarray(colored_heatmap)
    heatmap_image = heatmap_image.resize(image.size)

    overlay = Image.blend(
        image,
        heatmap_image,
        alpha=alpha,
    )

    return overlay


def generate_gradcam(
    image: Image.Image,
    model: tf.keras.Model,
    target_layer_name: str = DEFAULT_GRADCAM_LAYER,
    class_index: int | None = None,
) -> tuple[Image.Image, np.ndarray, int]:
    """
    Generates Grad-CAM overlay, raw heatmap, and class index.
    """
    heatmap, class_index = make_gradcam_heatmap(
        image=image,
        model=model,
        target_layer_name=target_layer_name,
        class_index=class_index,
    )

    overlay = overlay_heatmap_on_image(
        image=image,
        heatmap=heatmap,
    )

    return overlay, heatmap, class_index


if __name__ == "__main__":
    model = load_model(DEFAULT_MODEL_PATH)

    test_image_path = Path(
        "data/raw/NEU-DET/validation/images/scratches/scratches_241.jpg"
    )

    if not test_image_path.exists():
        raise FileNotFoundError(
            f"Test image not found at: {test_image_path}. "
            "Use an existing image from data/raw/NEU-DET/validation/images/."
        )

    image = Image.open(test_image_path)

    overlay, _, predicted_index = generate_gradcam(
        image=image,
        model=model,
    )

    predicted_class = CLASS_NAMES[predicted_index]

    output_path = Path("reports/gradcam_error_analysis/test_gradcam_overlay.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay.save(output_path)

    print("Grad-CAM saved to:", output_path)
    print("Predicted class index:", predicted_index)
    print("Predicted class:", predicted_class)
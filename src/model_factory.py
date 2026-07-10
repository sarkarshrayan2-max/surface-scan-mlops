from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocess import CLASS_NAMES, IMAGE_SIZE  


NUM_CLASSES = len(CLASS_NAMES)


def build_custom_cnn(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.50,
    l2_strength: float = 1e-4,
) -> tf.keras.Model:
    regularizer = tf.keras.regularizers.l2(l2_strength)

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomZoom(0.05)(x)

    x = tf.keras.layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizer,
    )(x)
    x = tf.keras.layers.Dropout(0.30)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="custom_cnn",
    )

    return model


def build_efficientnetb0(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.30,
    train_backbone: bool = False,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomZoom(0.05)(x)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = train_backbone

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="efficientnetb0_frozen" if not train_backbone else "efficientnetb0_finetuned",
    )

    return model


def build_mobilenetv3small(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.30,
    train_backbone: bool = False,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomZoom(0.05)(x)

    backbone = tf.keras.applications.MobileNetV3Small(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = train_backbone

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mobilenetv3small_frozen"
        if not train_backbone
        else "mobilenetv3small_finetuned",
    )

    return model


def build_model(
    model_name: str,
    train_backbone: bool = False,
) -> tf.keras.Model:
    model_name = model_name.lower()

    if model_name == "custom_cnn":
        return build_custom_cnn(
            input_shape=(*IMAGE_SIZE, 3),
        )

    if model_name == "efficientnetb0":
        return build_efficientnetb0(
            input_shape=(*IMAGE_SIZE, 3),
            train_backbone=train_backbone,
        )

    if model_name == "mobilenetv3small":
        return build_mobilenetv3small(
            input_shape=(*IMAGE_SIZE, 3),
            train_backbone=train_backbone,
        )

    raise ValueError(
        f"Unknown model name: {model_name}. "
        "Use one of: custom_cnn, efficientnetb0, mobilenetv3small."
    )


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_model(
        model_name="efficientnetb0",
        train_backbone=False,
    )

    model = compile_model(
        model=model,
        learning_rate=1e-3,
    )

    model.summary()
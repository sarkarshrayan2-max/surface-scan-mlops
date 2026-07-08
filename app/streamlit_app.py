from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.gradcam import generate_gradcam  # noqa: E402
from src.predict import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    load_model,
    predict_image,
)


st.set_page_config(
    page_title="SurfaceScan MLOps",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def get_model():
    return load_model(DEFAULT_MODEL_PATH)


def main() -> None:
    st.title("SurfaceScan MLOps")
    st.subheader("Industrial Steel Surface Defect Classification")

    st.write(
        "Upload a steel surface image. The champion EfficientNetB0 model "
        "will classify the defect type and generate a Grad-CAM explanation."
    )

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            f"Model file not found at `{DEFAULT_MODEL_PATH}`.\n\n"
            "Make sure the DVC-tracked champion model exists locally."
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload surface defect image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Upload an image to run prediction.")
        return

    image = Image.open(uploaded_file)

    model = get_model()

    with st.spinner("Running inference..."):
        result = predict_image(image=image, model=model)

    probabilities_df = pd.DataFrame(
        {
            "class": list(result.class_probabilities.keys()),
            "probability": list(result.class_probabilities.values()),
        }
    ).sort_values("probability", ascending=False)

    predicted_index = CLASS_NAMES.index(result.predicted_class)

    with st.spinner("Generating Grad-CAM explanation..."):
        gradcam_overlay, _, _ = generate_gradcam(
            image=image,
            model=model,
            class_index=predicted_index,
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

    with col2:
        st.image(
            gradcam_overlay,
            caption="Grad-CAM explanation",
            use_container_width=True,
        )

    st.divider()

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric("Predicted defect", result.predicted_class)

    with metric_col2:
        st.metric("Confidence", f"{result.confidence * 100:.2f}%")

    st.subheader("Class probabilities")

    st.dataframe(
        probabilities_df,
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        probabilities_df.set_index("class"),
        y="probability",
    )

    st.divider()

    st.subheader("Supported classes")
    st.code("\n".join(CLASS_NAMES))


if __name__ == "__main__":
    main()
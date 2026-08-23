import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference import ModelInference
from reports.report_generator import generate_report


st.set_page_config(
    page_title="RealTrace AI",
    page_icon="🔍",
    layout="wide"
)


st.title("🔍 RealTrace AI")
st.subheader("AI-Generated Image Authenticity Verification")

st.write(
    "Upload an image to analyze whether it is real or AI-generated."
)

st.info(
    "Model integration is currently pending. "
    "The trained CNN/ResNet/EfficientNet models will be integrated later."
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width="stretch"
    )

    st.write(f"File name: `{uploaded_file.name}`")

    if st.button("🔎 Analyze Image"):
        
        with st.spinner("Analyzing image..."):
            
            try:
                image = Image.open(uploaded_file)

                model = ModelInference()
                result = model.predict(image)

                st.subheader("Analysis Result")

                if result.prediction is None:
                    st.warning(
                        "Analysis is not available yet. "
                        "The trained AI model is still being integrated."
                    )

                else:
                    if result.prediction == 1:
                        prediction_label = "AI-Generated Image"
                    else:
                        prediction_label = "Real Image"

                    st.success(
                        f"Prediction: **{prediction_label}**"
                    )

                    if result.confidence is not None:
                        st.metric(
                            "Confidence",
                            f"{result.confidence * 100:.2f}%"
                        )

                    if result.probabilities:
                        st.subheader("Probability Distribution")

                        for label, probability in result.probabilities.items():
                            st.write(
                                f"**{label}**: "
                                f"{probability * 100:.2f}%"
                            )

                    if result.model_name:
                        st.caption(
                            f"Model: {result.model_name}"
                        )

                    report_path = generate_report(
                        filename=uploaded_file.name,
                        prediction=result.prediction,
                        confidence=result.confidence,
                        probabilities=result.probabilities,
                        model_name=result.model_name
                    )

                    with open(report_path, "rb") as report_file:
                        st.download_button(
                            label="📄 Download Analysis Report",
                            data=report_file,
                            file_name=report_path.name,
                            mime="text/plain"
                        )

            except NotImplementedError:
                st.warning(
                    "Analysis is not available yet. "
                    "The trained AI model is still being integrated."
                )

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
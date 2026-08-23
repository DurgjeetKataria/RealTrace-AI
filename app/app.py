import streamlit as st


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

        st.warning(
            "Analysis is not available yet. "
            "Model integration is pending."
        )
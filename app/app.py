import os
import sys

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# Allow imports from project root
sys.path.append(os.path.abspath("."))

from src.inference.analyze_image import analyze_image
from src.reporting.report_generator import generate_report


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RealTrace AI | Forensic Verification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Model registry (checkpoint mapping preserved exactly)
# ---------------------------------------------------------------------------
MODEL_OPTIONS = {
    "Baseline CNN": "checkpoints/baseline_best.pt",
    "ResNet18": "checkpoints/resnet18_best.pt",
    "EfficientNet-B0 Original": "checkpoints/efficientnet_b0_best.pt",
    "EfficientNet-B0 Generalized": "checkpoints/efficientnet_b0_generalized.pt",
}


# ---------------------------------------------------------------------------
# Design system (dark forensic-workstation theme)
# ---------------------------------------------------------------------------
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --bg: #15171c;
            --panel: #1f232b;
            --panel-2: #262b34;
            --line: #353b46;
            --line-strong: #454c5a;
            --text: #e6e9ef;
            --muted: #9aa5b8;
            --blue: #d4a72c;
            --blue-soft: #2b2618;
            --teal: #34d399;
            --teal-soft: #122a24;
            --red: #f87171;
            --red-soft: #2a1818;
            --amber-line: #5a4a25;
            --amber-bg: #23201a;
        }

        .stApp { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1380px; padding: 2.5rem 3rem 4rem; }

        h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.03em; color: var(--text) !important; }
        h1 { font-size: 2.5rem !important; margin-bottom: .15rem !important; }
        h2 { font-size: 1.35rem !important; margin: 0 !important; }
        h3 { font-size: 1.05rem !important; }
        p, label, .stCaption { color: var(--muted); }

        /* Sidebar */
        [data-testid="stSidebar"] { background: #1a1d23; border-right: 1px solid var(--line); }
        [data-testid="stSidebar"] > div:first-child { padding: 2rem 1.25rem; }
        .brand { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.3rem; color: var(--text); display: flex; align-items: center; }
        .brand-mark { display: inline-flex; width: 30px; height: 30px; align-items: center; justify-content: center;
            border: 1px solid var(--blue); background: var(--blue-soft); border-radius: 9px; color: var(--blue);
            margin-right: 10px; font-weight: 700; font-size: .95rem; }
        .brand-sub { color: var(--muted); font-size: .78rem; margin-top: .45rem; line-height: 1.5; }
        .sidebar-label { color: #7a8aa1; text-transform: uppercase; letter-spacing: .12em; font-size: .68rem; font-weight: 700; margin: 1.8rem 0 .55rem; }
        .pipeline { color: var(--muted); font-size: .83rem; line-height: 1.95; }
        .pipeline .step-no { display: inline-block; width: 22px; color: var(--blue); font-weight: 700; }
        .pipeline b { color: #c4ccd9; font-weight: 500; }
        .sidebar-note { border: 1px solid var(--line); background: var(--panel-2); border-radius: 12px;
            padding: .8rem .9rem; color: var(--muted); font-size: .78rem; line-height: 1.55; }

        /* Hero */
        .eyebrow { color: var(--blue); font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .55rem; }
        .subtitle { font-size: 1.02rem; color: #b6c0d0; margin: .35rem 0 0; }
        .subtitle-2 { font-size: .9rem; color: var(--muted); margin: .35rem 0 0; }
        .status { display: inline-flex; align-items: center; gap: 8px; padding: 7px 13px; border: 1px solid #5a4a25;
            border-radius: 999px; color: var(--blue); background: var(--blue-soft); font-size: .8rem; font-weight: 600; white-space: nowrap; }
        .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--teal); box-shadow: 0 0 0 4px rgba(52,211,153,.15); }
        .hero { display: flex; align-items: flex-end; justify-content: space-between; gap: 2rem;
            margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--line); }

        /* Cards */
        .card { background: var(--panel); border: 1px solid var(--line); border-radius: 16px;
            padding: 1.25rem 1.35rem; margin-bottom: 1rem; box-shadow: 0 1px 2px rgba(0,0,0,.25); }
        .card-title { color: var(--text); font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.02rem; margin-bottom: .25rem; }
        .card-copy { color: var(--muted); font-size: .87rem; line-height: 1.55; }

        .metric-card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
            padding: 1rem 1.05rem; min-height: 104px; margin-bottom: .75rem; box-shadow: 0 1px 2px rgba(0,0,0,.2); }
        .metric-label { color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 600; }
        .metric-value { color: var(--text); font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; margin-top: .45rem; overflow-wrap: anywhere; }
        .real { color: var(--teal) !important; }
        .generated { color: var(--red) !important; }
        .pill { display: inline-flex; align-items: center; gap: 7px; margin-top: .5rem; padding: 5px 11px;
            border-radius: 999px; font-size: .74rem; font-weight: 700; letter-spacing: .04em; }
        .pill.real { background: var(--teal-soft); border: 1px solid #1f4a3f; }
        .pill.generated { background: var(--red-soft); border: 1px solid #4a2828; }

        /* Sections */
        .section { margin-top: 2.1rem; margin-bottom: .8rem; display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
        .section-note { color: var(--muted); font-size: .8rem; }

        /* Controls */
        .stButton > button, .stButton > button p, .stButton > button span,
        .stDownloadButton > button, .stDownloadButton > button p, .stDownloadButton > button span {
            color: #ffffff !important; }
        .stButton > button { border-radius: 10px; border: 1px solid var(--blue); background: var(--blue);
            color: #ffffff !important; font-weight: 700; min-height: 44px; }
        .stButton > button:hover { border-color: #b8921f; background: #b8921f; color: #ffffff !important; }
        .stDownloadButton > button { border-radius: 10px; border: 1px solid var(--blue); background: var(--blue);
            color: #ffffff !important; font-weight: 700; min-height: 44px; }
        .stDownloadButton > button:hover { border-color: #b8921f; background: #b8921f; color: #ffffff !important; }
        [data-testid="stFileUploader"] { background: var(--panel-2); border: 1px dashed var(--line-strong); border-radius: 12px; padding: .5rem; }
        [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); padding: 1rem; border-radius: 12px; }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--text); }
        [data-testid="stImage"] img { border-radius: 14px; border: 1px solid var(--line); }

        .disclaimer { border: 1px solid var(--amber-line); background: var(--amber-bg); border-radius: 14px;
            padding: 1rem 1.15rem; color: #d8c89a; font-size: .86rem; line-height: 1.65; }
        .footer { border-top: 1px solid var(--line); margin-top: 2.5rem; padding-top: 1rem; color: #6b7689; font-size: .75rem; }

        /* Streamlit native widget theming for dark mode */
        .stSelectbox > div > div { background-color: var(--panel-2); border-color: var(--line); color: var(--text); }
        .stSelectbox svg { color: var(--muted); }
        .stProgress > div > div { background-color: transparent; }
        .stProgress > div { background-color: var(--line); }
        .stProgress, .stProgress p, .stProgress span, [data-testid="stProgress"] { color: #ffffff !important; }
        div[data-testid="stAlert"] { background-color: var(--panel-2); color: var(--text); }
        .stSpinner > div { color: var(--blue); }

        @media (max-width: 900px) {
            .block-container { padding: 1.5rem 1rem 3rem; }
            .hero { display: block; }
            .hero .status { margin-top: 1rem; }
            h1 { font-size: 2rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def metric_card(label: str, value, class_name: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {class_name}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="section">
            <h2>{title}</h2>
            <div class="section-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_open(title: str, copy: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Layout blocks
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-mark">R</span>RealTrace AI</div>'
            '<div class="brand-sub">Generalizable forensic intelligence framework for authenticity '
            "verification of AI-generated visual content.</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Detection Configuration</div>', unsafe_allow_html=True)
        selected = st.selectbox("Select Detection Model", list(MODEL_OPTIONS.keys()), index=3)
        st.caption(f"Checkpoint: `{MODEL_OPTIONS[selected]}`")

        st.markdown('<div class="sidebar-label">Analysis Pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="pipeline">
                <span class="step-no">01</span><b>Visual classification</b><br/>
                <span class="step-no">02</span><b>Confidence analysis</b><br/>
                <span class="step-no">03</span><b>Grad-CAM explanation</b><br/>
                <span class="step-no">04</span><b>FFT analysis</b><br/>
                <span class="step-no">05</span><b>DCT analysis</b><br/>
                <span class="step-no">06</span><b>Forensic report</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">System Note</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-note">Model-based analysis for research and demonstration purposes. '
            "Results should be interpreted with scientific caution.</div>",
            unsafe_allow_html=True,
        )
    return selected


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div>
                <div class="eyebrow">Digital forensics workstation</div>
                <h1>RealTrace AI</h1>
                <p class="subtitle">AI-Generated Visual Content Authenticity Verification</p>
                <p class="subtitle-2">Deep-learning classification with explainable AI and frequency-domain forensic analysis.</p>
            </div>
            <div class="status"><span class="status-dot"></span>Forensic Analysis System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(result: dict, selected_model: str, prediction_class: str) -> None:
    section("Classification Result", "Model output")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Prediction</div>
                <div class="metric-value {prediction_class}">{result["prediction"]}</div>
                <div class="pill {prediction_class}">
                    {"Classified as authentic capture" if prediction_class == "real" else "Classified as AI-generated"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        metric_card("Model confidence", f'{result["confidence"] * 100:.2f}%')
    with c3:
        metric_card("Selected model", selected_model)


def render_probability_section(result: dict) -> None:
    section("Prediction Probabilities", "Softmax output")
    p1, p2 = st.columns(2)
    with p1:
        card_open("Real Probability", "Likelihood assigned to the authentic-capture class.")
        st.progress(float(result["real_probability"]), text=f'{result["real_probability"] * 100:.2f}%')
        card_close()
    with p2:
        card_open("AI-Generated Probability", "Likelihood assigned to the AI-generated class.")
        st.progress(float(result["ai_probability"]), text=f'{result["ai_probability"] * 100:.2f}%')
        card_close()
    st.info(
        "Model confidence represents the model's certainty in its classification. It is not the "
        "probability that the image is genuinely authentic. Images from unseen generators may be "
        "misclassified with high confidence."
    )


def render_gradcam_section(result: dict) -> None:
    section("Explainable AI — Grad-CAM", "Class-activation overlay")
    st.image(
        result["gradcam_overlay"],
        caption="Regions contributing strongly to the model prediction",
        use_container_width=True,
    )
    st.info(
        "Grad-CAM indicates image regions that contributed strongly to the model's prediction. "
        "A highlighted region is not definitive proof that the region itself is fake."
    )


def render_frequency_section(result: dict) -> None:
    section("Frequency-Domain Forensics", "Spectral representations")
    f1, f2 = st.columns(2)
    with f1:
        card_open(
            "FFT Magnitude Spectrum",
            "Frequency representation generated using the Fast Fourier Transform.",
        )
        fft_fig = plt.figure(figsize=(6, 5))
        plt.imshow(result["fft"], cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fft_fig)
        plt.close(fft_fig)
        card_close()
    with f2:
        card_open(
            "DCT Magnitude Spectrum",
            "Spatial-frequency representation generated using the Discrete Cosine Transform.",
        )
        dct_fig = plt.figure(figsize=(6, 5))
        plt.imshow(result["dct"], cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(dct_fig)
        plt.close(dct_fig)
        card_close()


def render_summary_section(result: dict, selected_model: str, prediction_class: str) -> None:
    section("Analysis Summary", "Values returned by the inference pipeline")
    s1, s2, s3 = st.columns(3)
    with s1:
        metric_card("Selected model", selected_model)
        metric_card("Internal model name", result["model_name"])
    with s2:
        metric_card("Predicted class", result["prediction"], prediction_class)
        metric_card("Model confidence", f'{result["confidence"] * 100:.2f}%')
    with s3:
        metric_card("Real probability", f'{result["real_probability"] * 100:.2f}%')
        metric_card("AI-generated probability", f'{result["ai_probability"] * 100:.2f}%')


def render_report_section(image: Image.Image, result: dict) -> None:
    section("Forensic Analysis Report", "PDF export")
    card_open(
        "Downloadable report",
        "Download a PDF containing the classification result, explainability visualization, "
        "and frequency-domain analysis.",
    )
    report = generate_report(image, result)
    st.download_button(
        "Download Forensic Analysis Report",
        data=report.getvalue(),
        file_name="RealTrace_AI_Forensic_Report.pdf",
        mime="application/pdf",
    )
    card_close()


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            <b>Scientific disclaimer</b><br/>
            RealTrace AI provides model-based forensic analysis. The classification, confidence score,
            Grad-CAM visualization and frequency-domain representations should not be treated as
            definitive forensic proof. Results may be less reliable for AI generators or image
            distributions not represented during model training.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        '<div class="footer">RealTrace AI · University final-year project demonstration · '
        "Explainable, evidence-aware visual verification</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
inject_styles()

selected_model = render_sidebar()
checkpoint_path = MODEL_OPTIONS[selected_model]

render_header()

section("Image Evidence", "JPG · JPEG · PNG · WEBP")
card_open(
    "Upload an image for examination",
    "Evidence is analyzed only after you initiate the forensic workflow.",
)
uploaded_file = st.file_uploader(
    "Upload image evidence",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)
card_close()

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        left, right = st.columns([1.45, 1], gap="large")
        with left:
            st.image(image, caption="Evidence preview", use_container_width=True)
        with right:
            st.markdown('<div class="card-title">Evidence metadata</div>', unsafe_allow_html=True)
            metric_card("File name", uploaded_file.name)
            metric_card("Dimensions", f"{image.width} × {image.height} px")
            metric_card("Format", image.format or "RGB image")
            analyze = st.button("Analyze Image", type="primary", use_container_width=True)

        if analyze:
            with st.spinner("Running forensic analysis..."):
                result = analyze_image(image, checkpoint_path)
            st.success("Analysis completed successfully.")

            prediction_class = "real" if str(result["prediction"]).upper() == "REAL" else "generated"

            render_result_card(result, selected_model, prediction_class)
            render_probability_section(result)
            render_gradcam_section(result)
            render_frequency_section(result)
            render_summary_section(result, selected_model, prediction_class)
            render_report_section(image, result)
            render_disclaimer()
    except Exception as error:
        st.error(f"Analysis failed: {error}")

render_footer()

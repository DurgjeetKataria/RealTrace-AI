import os
import sys

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# Allow imports from project root
sys.path.append(os.path.abspath("."))

from src.inference.analyze_image import analyze_image
from src.reporting.report_generator import generate_report


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RealTrace AI",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# MODEL OPTIONS
# (unchanged mappings — do not edit paths, names, or order)
# ============================================================

MODEL_OPTIONS = {
    "Baseline CNN": "checkpoints/baseline_best.pt",
    "ResNet18": "checkpoints/resnet18_best.pt",
    "EfficientNet-B0 Original": "checkpoints/efficientnet_b0_best.pt",
    "EfficientNet-B0 Generalized": "checkpoints/efficientnet_b0_generalized.pt",
}


# ============================================================
# GLOBAL STYLES
# ============================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg:            #0A0F1A;
            --bg-elevated:   #0F1626;
            --surface:       #121A2B;
            --surface-alt:   #16203350;
            --border:        #223049;
            --border-soft:   #1A2338;
            --text-primary:  #E7ECF5;
            --text-secondary:#95A2BC;
            --text-muted:    #5E6B85;
            --accent:        #4C8DF5;
            --accent-soft:   #4C8DF522;
            --real:          #34C77B;
            --real-soft:     #34C77B1A;
            --ai:            #F0A23A;
            --ai-soft:       #F0A23A1A;
            --danger:        #EF5A5A;
            --mono: 'JetBrains Mono', 'SFMono-Regular', Consolas, monospace;
            --sans: 'Inter', -apple-system, 'Segoe UI', sans-serif;
        }

        html, body, [class*="css"] {
            font-family: var(--sans);
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, #101B30 0%, transparent 45%),
                var(--bg);
            color: var(--text-primary);
        }

        .main .block-container {
            max-width: 900px;
            margin: 0 auto;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        #MainMenu, footer, header[data-testid="stHeader"] {
            background: transparent;
        }

        /* ---------- Hero header ---------- */
        .rt-hero {
            position: relative;
            text-align: center;
            padding: 10px 0 34px 0;
            margin-bottom: 26px;
        }
        .rt-hero::before {
            content: "";
            position: absolute;
            top: -120px;
            left: 50%;
            transform: translateX(-50%);
            width: 640px;
            height: 320px;
            background: radial-gradient(circle, #2E5FD955 0%, transparent 68%);
            pointer-events: none;
            z-index: 0;
        }
        .rt-badge {
            position: relative;
            z-index: 1;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 16px;
            border-radius: 100px;
            background: var(--surface);
            border: 1px solid var(--border);
            font-size: 12.5px;
            color: var(--text-secondary);
            margin-bottom: 18px;
        }
        .rt-dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: var(--real);
        }
        .rt-title {
            position: relative;
            z-index: 1;
            font-size: 40px;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: var(--text-primary);
            margin: 0 0 10px 0;
        }
        .rt-title span { color: var(--accent); }
        .rt-subtitle {
            position: relative;
            z-index: 1;
            font-size: 15px;
            color: var(--text-secondary);
            margin: 0 0 6px 0;
        }
        .rt-tagline {
            position: relative;
            z-index: 1;
            font-size: 13px;
            color: var(--text-muted);
            margin: 0;
        }

        /* ---------- Cards ---------- */
        .rt-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 22px 24px;
            margin-bottom: 18px;
        }
        .rt-card-title {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            margin: 0 0 4px 0;
        }
        .rt-card-desc {
            font-size: 13px;
            color: var(--text-muted);
            margin: 0 0 16px 0;
        }

        .rt-section-label {
            font-size: 11.5px;
            color: var(--text-muted);
            font-weight: 600;
            letter-spacing: 0.02em;
            margin: 0 0 10px 0;
        }
        /* ---------- Verdict badge ---------- */
        .rt-verdict {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 10px;
            font-weight: 700;
            font-size: 20px;
            letter-spacing: -0.01em;
        }
        .rt-verdict.real {
            background: var(--real-soft);
            color: var(--real);
            border: 1px solid #34C77B44;
        }
        .rt-verdict.ai {
            background: var(--ai-soft);
            color: var(--ai);
            border: 1px solid #F0A23A44;
        }
        .rt-stat-label {
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 4px;
        }
        .rt-stat-value {
            font-size: 22px;
            font-weight: 700;
            color: var(--text-primary);
        }

        /* ---------- Probability bars ---------- */
        .rt-prob-row { margin-bottom: 14px; }
        .rt-prob-label {
            display: flex;
            justify-content: space-between;
            font-size: 13.5px;
            color: var(--text-secondary);
            margin-bottom: 6px;
        }
        .rt-prob-value { font-family: var(--mono); color: var(--text-primary); }
        .rt-prob-track {
            width: 100%;
            height: 9px;
            border-radius: 100px;
            background: var(--bg);
            border: 1px solid var(--border-soft);
            overflow: hidden;
        }
        .rt-prob-fill { height: 100%; border-radius: 100px; }
        .rt-prob-fill.real { background: var(--real); }
        .rt-prob-fill.ai { background: var(--ai); }

        /* ---------- Info footnotes ---------- */
        .rt-note {
            font-size: 12.5px;
            color: var(--text-muted);
            line-height: 1.55;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid var(--border-soft);
        }

        /* ---------- Summary table ---------- */
        .rt-summary-row {
            display: flex;
            justify-content: space-between;
            padding: 9px 0;
            border-bottom: 1px solid var(--border-soft);
            font-size: 13.5px;
        }
        .rt-summary-row:last-child { border-bottom: none; }
        .rt-summary-key { color: var(--text-muted); }
        .rt-summary-val {
            color: var(--text-primary);
            font-family: var(--mono);
            font-size: 13px;
            text-align: right;
        }

        /* ---------- Disclaimer panel ---------- */
        .rt-disclaimer-panel {
            background: linear-gradient(180deg, #16203D 0%, #121A2B 100%);
            border: 1px solid #2C3A5A;
            border-left: 3px solid var(--accent);
            border-radius: 12px;
            padding: 18px 22px;
            font-size: 13px;
            color: var(--text-secondary);
            line-height: 1.6;
        }
        .rt-disclaimer-title {
            font-size: 13px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 6px;
        }

        /* Buttons */
        div.stButton > button {
            background: var(--accent);
            color: #0A0F1A;
            border: none;
            border-radius: 100px;
            font-weight: 600;
            padding: 0.6em 1.4em;
        }
        div.stButton > button:hover {
            background: #6BA1F8;
            color: #0A0F1A;
        }
        div.stDownloadButton > button {
            background: var(--surface);
            color: var(--text-primary);
            border: 1px solid var(--border);
            border-radius: 100px;
            font-weight: 600;
            padding: 0.6em 1.4em;
        }
        div.stDownloadButton > button:hover {
            border-color: var(--accent);
            color: var(--accent);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--bg);
            border: 1px dashed var(--border);
            border-radius: 12px;
        }

        hr { border-color: var(--border-soft); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UI HELPER FUNCTIONS
# ============================================================

def render_header() -> None:
    st.markdown(
        """
        <div class="rt-hero">
            <div class="rt-badge"><span class="rt-dot"></span> Forensic Analysis System</div>
            <p class="rt-title">🔍 RealTrace <span>AI</span></p>
            <p class="rt-subtitle">AI-Generated Visual Content Authenticity Verification</p>
            <p class="rt-tagline">
                Deep-learning classification with explainable AI and frequency-domain forensic analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_selector():
    st.markdown(
        """
        <div class="rt-card">
            <p class="rt-section-label">DETECTION CONFIGURATION</p>
            <p class="rt-card-title">Select Detection Model</p>
        """,
        unsafe_allow_html=True,
    )

    selected_model = st.selectbox(
        "Detection model",
        list(MODEL_OPTIONS.keys()),
        index=3,
        label_visibility="collapsed",
    )

    checkpoint_path = MODEL_OPTIONS[selected_model]

    st.markdown("</div>", unsafe_allow_html=True)
    return selected_model, checkpoint_path


def render_upload_card():
    st.markdown(
        """
        <div class="rt-card">
            <p class="rt-card-title">Image Evidence</p>
            <p class="rt-card-desc">Upload an image to submit it for forensic authenticity analysis.</p>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_file


def render_image_preview(image: Image.Image, uploaded_file) -> None:
    st.markdown(
        """
        <div class="rt-card">
            <p class="rt-card-title">Uploaded Image</p>
        """,
        unsafe_allow_html=True,
    )

    preview_col, info_col = st.columns([2, 1])

    with preview_col:
        st.image(image, caption="Image selected for analysis", use_container_width=True)

    with info_col:
        st.markdown('<div class="rt-summary-row"><span class="rt-summary-key">File name</span>'
                     f'<span class="rt-summary-val">{uploaded_file.name}</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="rt-summary-row"><span class="rt-summary-key">Dimensions</span>'
                     f'<span class="rt-summary-val">{image.size[0]} × {image.size[1]}</span></div>', unsafe_allow_html=True)
        fmt = uploaded_file.type or (image.format or "Unknown")
        st.markdown('<div class="rt-summary-row"><span class="rt-summary-key">Format</span>'
                     f'<span class="rt-summary-val">{fmt}</span></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_result_card(result: dict, selected_model: str) -> None:
    prediction = result["prediction"]
    is_real = str(prediction).strip().upper() == "REAL"
    verdict_class = "real" if is_real else "ai"

    st.markdown('<div class="rt-card">', unsafe_allow_html=True)
    st.markdown('<p class="rt-card-title">Classification Result</p>', unsafe_allow_html=True)

    verdict_col, conf_col, model_col = st.columns([1.3, 1, 1])

    with verdict_col:
        st.markdown('<div class="rt-stat-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="rt-verdict {verdict_class}">{prediction}</span>',
            unsafe_allow_html=True,
        )

    with conf_col:
        st.markdown('<div class="rt-stat-label">Model Confidence</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="rt-stat-value">{result["confidence"] * 100:.2f}%</div>',
            unsafe_allow_html=True,
        )

    with model_col:
        st.markdown('<div class="rt-stat-label">Selected Model</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rt-stat-value" style="font-size:15px;">{selected_model}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_probability_section(result: dict) -> None:
    real_pct = result["real_probability"] * 100
    ai_pct = result["ai_probability"] * 100

    st.markdown(
        """
        <div class="rt-card">
            <p class="rt-card-title">Prediction Probabilities</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="rt-prob-row">
            <div class="rt-prob-label"><span>Real</span><span class="rt-prob-value">{real_pct:.2f}%</span></div>
            <div class="rt-prob-track"><div class="rt-prob-fill real" style="width:{real_pct:.2f}%;"></div></div>
        </div>
        <div class="rt-prob-row">
            <div class="rt-prob-label"><span>AI-Generated</span><span class="rt-prob-value">{ai_pct:.2f}%</span></div>
            <div class="rt-prob-track"><div class="rt-prob-fill ai" style="width:{ai_pct:.2f}%;"></div></div>
        </div>
        <div class="rt-note">
            Model confidence represents the model's certainty in its classification.
            It is not the probability that the image is genuinely authentic. Images
            from unseen generators may be misclassified with high confidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_gradcam_section(result: dict) -> None:
    st.markdown(
        """
        <div class="rt-card">
            <p class="rt-card-title">Explainable AI — Grad-CAM</p>
            <p class="rt-card-desc">Visual regions that most influenced the model's decision.</p>
        """,
        unsafe_allow_html=True,
    )

    st.image(
        result["gradcam_overlay"],
        caption="Grad-CAM overlay showing regions that contributed strongly to the prediction.",
        use_container_width=False,
        width=520,
    )

    st.markdown(
        """
        <div class="rt-note">
            Grad-CAM indicates image regions that contributed strongly to the model's
            prediction. A highlighted region is not definitive proof that the region
            itself is fake.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_frequency_section(result: dict) -> None:
    st.markdown('<p class="rt-card-title" style="margin-top:4px;">Frequency-Domain Forensics</p>', unsafe_allow_html=True)

    fft_col, dct_col = st.columns(2)

    with fft_col:
        st.markdown(
            """
            <div class="rt-card">
                <p class="rt-card-title" style="font-size:14px;">FFT Magnitude Spectrum</p>
            """,
            unsafe_allow_html=True,
        )

        fft_fig = plt.figure(figsize=(5, 5))
        plt.imshow(result["fft"], cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fft_fig)
        plt.close(fft_fig)

        st.markdown(
            '<div class="rt-note">Frequency representation generated using the Fast Fourier Transform.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with dct_col:
        st.markdown(
            """
            <div class="rt-card">
                <p class="rt-card-title" style="font-size:14px;">DCT Magnitude Spectrum</p>
            """,
            unsafe_allow_html=True,
        )

        dct_fig = plt.figure(figsize=(5, 5))
        plt.imshow(result["dct"], cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(dct_fig)
        plt.close(dct_fig)

        st.markdown(
            '<div class="rt-note">Spatial-frequency representation generated using the Discrete Cosine Transform.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="rt-note" style="margin-top:-6px;">
            These visualizations support forensic interpretation and do not, on their
            own, prove that an image is AI-generated.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_section(result: dict, selected_model: str) -> None:
    st.markdown(
        """
        <div class="rt-card">
            <p class="rt-card-title">Analysis Summary</p>
        """,
        unsafe_allow_html=True,
    )

    rows = [
        ("Selected model", selected_model),
        ("Internal model name", result["model_name"]),
        ("Predicted class", result["prediction"]),
        ("Model confidence", f"{result['confidence'] * 100:.2f}%"),
        ("Real probability", f"{result['real_probability'] * 100:.2f}%"),
        ("AI-generated probability", f"{result['ai_probability'] * 100:.2f}%"),
    ]

    rows_html = "".join(
        f'<div class="rt-summary-row"><span class="rt-summary-key">{k}</span>'
        f'<span class="rt-summary-val">{v}</span></div>'
        for k, v in rows
    )
    st.markdown(rows_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_report_section(image: Image.Image, result: dict) -> None:
    st.markdown(
        """
        <div class="rt-card">
            <p class="rt-card-title">Forensic Analysis Report</p>
            <p class="rt-card-desc">
                Download a PDF containing the classification result, explainability
                visualization, and frequency-domain analysis.
            </p>
        """,
        unsafe_allow_html=True,
    )

    report = generate_report(image, result)

    st.download_button(
        label="📄 Download Forensic Analysis Report",
        data=report.getvalue(),
        file_name="RealTrace_AI_Forensic_Report.pdf",
        mime="application/pdf",
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_disclaimer() -> None:
    st.markdown(
        """
        <div class="rt-disclaimer-panel">
            <div class="rt-disclaimer-title">Scientific Disclaimer</div>
            RealTrace AI provides model-based forensic analysis. The classification,
            confidence score, Grad-CAM visualization and frequency-domain
            representations should not be treated as definitive forensic proof.
            Results may be less reliable for AI generators or image distributions
            not represented during model training.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APP
# ============================================================

inject_css()
render_header()

selected_model, checkpoint_path = render_model_selector()

uploaded_file = render_upload_card()

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        render_image_preview(image, uploaded_file)

        analyze_clicked = st.button("🔎  Analyze Image", type="primary")

        if analyze_clicked:

            with st.spinner("Running forensic analysis..."):

                result = analyze_image(image, checkpoint_path)

            st.success("Analysis completed successfully.")

            render_result_card(result, selected_model)
            render_probability_section(result)
            render_gradcam_section(result)
            render_frequency_section(result)
            render_summary_section(result, selected_model)
            render_report_section(image, result)
            render_disclaimer()

    except Exception as error:

        st.error(f"Analysis failed: {error}")
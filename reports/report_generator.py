from datetime import datetime
from pathlib import Path


def generate_report(
    filename,
    prediction=None,
    confidence=None,
    probabilities=None,
    model_name=None
):
    """
    Generate a text report for a RealTrace AI analysis.

    The report does not create or assume a prediction.
    If the trained model is not integrated, the result
    is clearly marked as unavailable.
    """

    report_dir = Path("reports/generated")
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = []

    report.append("RealTrace AI - Image Analysis Report")
    report.append("=" * 45)
    report.append(f"File Name: {filename}")
    report.append(f"Analysis Time: {timestamp}")
    report.append("")

    report.append("Analysis Result")
    report.append("-" * 20)

    if prediction is None:
        report.append(
            "Prediction: Not available"
        )
        report.append(
            "Reason: Trained AI model is not integrated yet."
        )
    else:
        if prediction == 1:
            label = "AI-Generated Image"
        else:
            label = "Real Image"

        report.append(f"Prediction: {label}")

        if confidence is not None:
            report.append(
                f"Confidence: {confidence * 100:.2f}%"
            )

    if probabilities:
        report.append("")
        report.append("Probability Distribution")
        report.append("-" * 25)

        for label, probability in probabilities.items():
            report.append(
                f"{label}: {probability * 100:.2f}%"
            )

    if model_name:
        report.append("")
        report.append(f"Model: {model_name}")

    report.append("")
    report.append(
        "Note: This report contains only results "
        "provided by the integrated model."
    )

    report_path = (
        report_dir /
        f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    return report_path
from dataclasses import dataclass
from typing import Any, Optional

from PIL import Image


@dataclass
class InferenceResult:
    """
    Standard output format for RealTrace AI model inference.

    This interface allows Member 2's trained models to be
    integrated later without changing the Streamlit application.
    """

    prediction: Optional[int]
    confidence: Optional[float]
    probabilities: Optional[dict]
    model_name: Optional[str]


class ModelInference:
    """
    Model-independent inference interface.

    The actual trained CNN / ResNet / EfficientNet model
    will be connected later.
    """

    def __init__(self, model: Any = None, model_name: Optional[str] = None):
        self.model = model
        self.model_name = model_name

    def preprocess(self, image: Image.Image) -> Image.Image:
        """
        Basic image preprocessing.

        Actual preprocessing required by the trained model
        will be added during model integration.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def predict(self, image: Image.Image) -> InferenceResult:
        """
        Run inference.

        Currently returns no prediction because the trained
        model has not yet been integrated.

        This prevents fabricated predictions.
        """

        image = self.preprocess(image)

        if self.model is None:
            return InferenceResult(
                prediction=None,
                confidence=None,
                probabilities=None,
                model_name=self.model_name
            )

        # Actual model inference will be implemented
        # when Member 2 provides the trained model.
        raise NotImplementedError(
            "Trained model integration is pending."
        )
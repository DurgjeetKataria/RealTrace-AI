from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from PIL import Image


@dataclass
class InferenceResult:
    """
    Standard output format for RealTrace AI model inference.
    """

    prediction: Optional[int]
    confidence: Optional[float]
    probabilities: Optional[dict]
    model_name: Optional[str]


class TinyCNN(nn.Module):
    """
    Same TinyCNN architecture used during training.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Linear(
            16 * 7 * 7,
            2
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x


class ModelInference:
    """
    Handles loading the trained TinyCNN model
    and performing image inference.
    """

    def __init__(
        self,
        model: Any = None,
        model_name: Optional[str] = None
    ):
        self.model = model
        self.model_name = model_name

        if self.model is None:
            self._load_model()

    def _load_model(self):
        """
        Load the trained TinyCNN model from tiny_cnn.pth.
        """

        project_root = Path(__file__).resolve().parent.parent

        model_path = project_root / "tiny_cnn.pth"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}"
            )

        # Create the same architecture used during training
        model = TinyCNN()

        # Load trained weights
        checkpoint = torch.load(
            model_path,
            map_location=torch.device("cpu")
        )

        # Handle state_dict checkpoints
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                model.load_state_dict(
                    checkpoint["state_dict"]
                )
            else:
                model.load_state_dict(checkpoint)

        # Handle a complete saved model
        elif isinstance(checkpoint, nn.Module):
            model = checkpoint

        else:
            raise RuntimeError(
                "Unsupported model file format."
            )

        model.eval()

        self.model = model
        self.model_name = "TinyCNN"

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Convert uploaded image into the format expected
        by the TinyCNN model.

        Input:
            PIL Image

        Output:
            Tensor with shape [1, 1, 28, 28]
        """

        # Convert to grayscale
        image = image.convert("L")

        # Resize to the same size used during training
        image = image.resize(
            (28, 28),
            Image.Resampling.LANCZOS
        )

        # Convert image bytes into a tensor
        image_bytes = bytearray(image.tobytes())

        tensor = torch.tensor(
            image_bytes,
            dtype=torch.float32
        )

        # Reshape to:
        # batch = 1
        # channel = 1
        # height = 28
        # width = 28
        tensor = tensor.reshape(
            1,
            1,
            28,
            28
        )

        # Normalize pixel values from 0-255 to 0-1
        tensor = tensor / 255.0

        return tensor

    def predict(
        self,
        image: Image.Image
    ) -> InferenceResult:
        """
        Run inference on an uploaded image.
        """

        if self.model is None:
            self._load_model()

        # Preprocess image
        input_tensor = self.preprocess(image)

        # Disable gradient calculation during inference
        with torch.no_grad():

            # Get raw model output
            output = self.model(input_tensor)

            # Convert raw output into probabilities
            probabilities = torch.softmax(
                output,
                dim=1
            )

            # Get predicted class
            prediction = torch.argmax(
                probabilities,
                dim=1
            ).item()

            # Get confidence
            confidence = probabilities[
                0,
                prediction
            ].item()

        # Class mapping used by the training script:
        #
        # 0 = vertical pattern
        # 1 = horizontal pattern
        #
        # NOTE:
        # These are NOT currently "Real Image" and
        # "AI-Generated Image" classes because the
        # current training data contains patterns.

        probability_dict = {
            "Class 0": probabilities[0, 0].item(),
            "Class 1": probabilities[0, 1].item()
        }

        return InferenceResult(
            prediction=prediction,
            confidence=confidence,
            probabilities=probability_dict,
            model_name=self.model_name
        )
"""
ml_food_classifier.py — Food Image Classification using MobileNetV2

Loads a trained MobileNetV2 model (from Google Colab) and classifies food images.
Used by the Food Logger agent for image-based food logging.

Usage:
    from ml_food_classifier import FoodClassifier

    classifier = FoodClassifier()               # loads model once
    result = classifier.classify(image_bytes)    # classify raw image bytes
    result = classifier.classify_base64(b64str)  # classify base64-encoded image
"""

import io
import json
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# Default paths (relative to project root)
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "food_classifier.pth"
DEFAULT_TRACED_MODEL_PATH = Path(__file__).parent / "models" / "food_classifier_traced.pt"
DEFAULT_CLASSES_PATH = Path(__file__).parent / "models" / "food_classes.json"


class FoodClassifier:
    """
    Food image classifier using MobileNetV2 trained on Food-101.

    Loads the model once on init and provides fast inference.
    Supports both raw bytes and base64-encoded image input.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        classes_path: Optional[str] = None,
        use_traced: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the classifier.

        Args:
            model_path:   Path to .pth or .pt model file
            classes_path: Path to food_classes.json
            use_traced:   If True, try to load TorchScript model first (faster CPU inference)
            device:       "cpu" or "cuda" (auto-detected if None)
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.class_names: List[str] = []
        self.num_classes = 101
        self.input_size = 224
        self.confidence_threshold = 0.1
        self._loaded = False

        # Image preprocessing (must match training transforms)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load class info
        classes_file = Path(classes_path) if classes_path else DEFAULT_CLASSES_PATH
        if classes_file.exists():
            with open(classes_file) as f:
                info = json.load(f)
            self.class_names = info.get("classes", [])
            self.num_classes = info.get("num_classes", 101)
            self.input_size = info.get("input_size", 224)

        # Try loading model
        if use_traced:
            traced_path = Path(model_path).with_suffix(".pt") if model_path else DEFAULT_TRACED_MODEL_PATH
            if traced_path.exists():
                self._load_traced(str(traced_path))
                return

        pth_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if pth_path.exists():
            self._load_weights(str(pth_path))

    def _load_traced(self, path: str):
        """Load TorchScript traced model."""
        self.model = torch.jit.load(path, map_location=self.device)
        self.model.eval()
        self._loaded = True
        print(f"[ML] Loaded TorchScript model from {path} on {self.device}")

    def _load_weights(self, path: str):
        """Load model weights into MobileNetV2 architecture."""
        model = models.mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, self.num_classes),
        )
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        self.model = model
        self._loaded = True
        print(f"[ML] Loaded MobileNetV2 weights from {path} on {self.device}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Convert raw image bytes to model input tensor."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(img).unsqueeze(0)
        return tensor.to(self.device)

    def classify(
        self,
        image_bytes: bytes,
        top_k: int = 5,
    ) -> Dict:
        """
        Classify a food image from raw bytes.

        Args:
            image_bytes: Raw image file bytes (JPEG/PNG)
            top_k:       Number of top predictions to return

        Returns:
            dict with keys:
                - predicted_class: str (top prediction)
                - confidence: float (0-1)
                - top_k: list of {class, confidence} dicts
                - inference_time_ms: float
                - image_size_bytes: int
        """
        if not self._loaded:
            return {
                "error": True,
                "text": "ML model not loaded. Place model files in models/ directory.",
            }

        # Preprocess
        tensor = self._preprocess(image_bytes)

        # Inference
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
        inference_ms = (time.perf_counter() - start) * 1000

        # Top-k predictions
        top_probs, top_indices = probs.topk(top_k)
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_idx = idx.item()
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f"class_{class_idx}"
            predictions.append({
                "class": class_name.replace("_", " ").title(),
                "class_raw": class_name,
                "confidence": round(prob.item(), 4),
            })

        top = predictions[0]
        return {
            "predicted_class": top["class"],
            "predicted_class_raw": top["class_raw"],
            "confidence": top["confidence"],
            "top_k": predictions,
            "inference_time_ms": round(inference_ms, 2),
            "image_size_bytes": len(image_bytes),
        }

    def classify_base64(self, b64_string: str, top_k: int = 5) -> Dict:
        """
        Classify a food image from a base64-encoded string.

        Args:
            b64_string: Base64-encoded image data (with or without data URI prefix)
            top_k:      Number of top predictions to return
        """
        # Strip data URI prefix if present (e.g., "data:image/jpeg;base64,...")
        if "," in b64_string and b64_string.startswith("data:"):
            b64_string = b64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(b64_string)
        return self.classify(image_bytes, top_k)

    def get_image_as_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes as base64 string."""
        return base64.b64encode(image_bytes).decode("utf-8")


# ---------------------------------------------------------------------------
# Global singleton (loaded once when imported)
# ---------------------------------------------------------------------------

_classifier: Optional[FoodClassifier] = None


def get_classifier() -> FoodClassifier:
    """Get or create the global FoodClassifier singleton."""
    global _classifier
    if _classifier is None:
        _classifier = FoodClassifier()
    return _classifier

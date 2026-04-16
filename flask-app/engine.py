import json
import importlib
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw


def _load_boto3():
    try:
        return importlib.import_module("boto3")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("API mode requires boto3 to be installed") from exc


@dataclass
class ProcessorConfig:
    mode: str = "SERVER"
    manifest_dir: Optional[str] = None
    rekognition_region: str = "us-east-1"


class UniversalProcessor:
    """Route image processing to LOCAL, SERVER, or API lanes."""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.mode = (config.mode or "SERVER").upper()
        self._rekognition = None

        if self.mode == "API":
            boto3 = _load_boto3()
            self._rekognition = boto3.client("rekognition", region_name=config.rekognition_region)

    @staticmethod
    def _load_rgb(image_path: str, max_edge: int = 1600) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        longest = max(h, w)
        if longest > max_edge:
            scale = max_edge / float(longest)
            rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        return rgb

    @staticmethod
    def _match_embeddings(query: np.ndarray, known: np.ndarray, threshold: float = 0.6) -> Dict[str, Any]:
        """Fast NumPy-vectorized Euclidean matching."""
        if known.size == 0:
            return {"matches": [], "distances": []}
        diffs = known - query
        distances = np.sqrt(np.sum(diffs * diffs, axis=1))
        idx = np.where(distances <= threshold)[0]
        return {
            "matches": idx.tolist(),
            "distances": distances.tolist(),
        }

    def _process_local(self, image_path: str) -> Dict[str, Any]:
        manifest_dir = self.config.manifest_dir or os.path.dirname(image_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        manifest_path = os.path.join(manifest_dir, f"{stem}.json")

        if not os.path.exists(manifest_path):
            return {
                "mode": "LOCAL",
                "status": "manifest_wait",
                "manifest_path": manifest_path,
                "embeddings": [],
            }

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "mode": "LOCAL",
            "status": "manifest_ready",
            "manifest_path": manifest_path,
            "manifest": data,
            "embeddings": data.get("embeddings", []),
        }

    def _process_server(self, image_path: str, known_embeddings: Optional[np.ndarray]) -> Dict[str, Any]:
        rgb = self._load_rgb(image_path)
        encodings = face_recognition.face_encodings(rgb)
        vectors = np.array(encodings, dtype=np.float32) if encodings else np.empty((0, 128), dtype=np.float32)

        result: Dict[str, Any] = {
            "mode": "SERVER",
            "status": "ok",
            "face_count": int(vectors.shape[0]),
            "embeddings": vectors,
        }

        if vectors.shape[0] and known_embeddings is not None and known_embeddings.size:
            result["match"] = self._match_embeddings(vectors[0], known_embeddings)

        return result

    def _process_api(self, image_path: str) -> Dict[str, Any]:
        if self._rekognition is None:
            boto3 = _load_boto3()
            self._rekognition = boto3.client("rekognition", region_name=self.config.rekognition_region)

        with open(image_path, "rb") as f:
            payload = f.read()

        response = self._rekognition.detect_faces(Image={"Bytes": payload}, Attributes=["DEFAULT"])
        details: List[Dict[str, Any]] = response.get("FaceDetails", [])

        return {
            "mode": "API",
            "status": "ok",
            "face_count": len(details),
            "rekognition": response,
            "embeddings": [],
        }

    def process_image(self, image_path: str, known_embeddings: Optional[np.ndarray] = None) -> Dict[str, Any]:
        mode = self.mode
        if mode == "LOCAL":
            return self._process_local(image_path)
        if mode == "API":
            return self._process_api(image_path)
        return self._process_server(image_path, known_embeddings)


def create_highlight_preview(
    image_path: str,
    face_location: tuple[int, int, int, int],
    cache_dir: str,
    key: Optional[str] = None,
) -> str:
    """Draw a semi-transparent purple face rectangle and save a cached preview."""
    os.makedirs(cache_dir, exist_ok=True)

    top, right, bottom, left = face_location
    top = int(max(0, top))
    right = int(max(0, right))
    bottom = int(max(0, bottom))
    left = int(max(0, left))

    with Image.open(image_path).convert("RGBA") as base:
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        draw.rectangle(
            [(left, top), (right, bottom)],
            fill=(128, 0, 255, 75),
            outline=(128, 0, 255, 220),
            width=5,
        )

        composed = Image.alpha_composite(base, overlay).convert("RGB")
        filename = f"verify_{key or uuid.uuid4().hex[:12]}.jpg"
        out_path = os.path.join(cache_dir, filename)
        composed.save(out_path, format="JPEG", quality=90)

    return filename

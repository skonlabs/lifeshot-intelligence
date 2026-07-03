"""Pydantic v2 schemas for the face feature."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Detector = Literal["yunet", "opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8", "centerface"]
Model = Literal["Facenet512", "Facenet", "VGG-Face", "ArcFace", "Dlib", "SFace", "GhostFaceNet", "OpenFace", "DeepFace", "DeepID"]
Metric = Literal["cosine", "euclidean", "euclidean_l2"]


class Eye(BaseModel):
    x: Optional[int] = None
    y: Optional[int] = None


class FacialArea(BaseModel):
    x: int
    y: int
    w: int
    h: int
    left_eye: Optional[List[int]] = None
    right_eye: Optional[List[int]] = None


class Camera(BaseModel):
    make: Optional[str] = None
    model: Optional[str] = None
    lens: Optional[str] = None
    iso: Optional[int] = None
    f_number: Optional[float] = None
    exposure_time: Optional[str] = None
    focal_length: Optional[float] = None


class GPS(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    altitude: Optional[float] = None


class ImageMeta(BaseModel):
    format: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    orientation: Optional[str] = None
    megapixels: Optional[float] = None
    has_alpha: Optional[bool] = None
    dpi: Optional[List[int]] = None
    taken_at: Optional[str] = None
    gps: Optional[GPS] = None
    camera: Optional[Camera] = None
    exif_present: bool = False


class DetectedFace(BaseModel):
    facial_area: FacialArea
    confidence: float
    face: Optional[str] = Field(None, description="base64 raw cv2 crop; only when return_faces=true")


class DetectResponse(BaseModel):
    image: ImageMeta
    count: int
    faces: List[DetectedFace]
    detector: str
    request_id: Optional[str] = None
    timing_ms: Optional[dict] = None


class Attributes(BaseModel):
    age: Optional[int] = None
    gender: Optional[dict] = None
    dominant_gender: Optional[str] = None
    emotion: Optional[dict] = None
    dominant_emotion: Optional[str] = None
    race: Optional[dict] = None
    dominant_race: Optional[str] = None


class AnalyzedFace(BaseModel):
    facial_area: FacialArea
    confidence: float
    attributes: Attributes


class AnalyzeResponse(BaseModel):
    image: ImageMeta
    count: int
    faces: List[AnalyzedFace]
    detector: str
    actions: List[str]
    request_id: Optional[str] = None
    timing_ms: Optional[dict] = None


class Embedding(BaseModel):
    embedding: List[float]
    facial_area: FacialArea
    face_confidence: float


class RepresentResponse(BaseModel):
    count: int
    dimension: int
    model: str
    detector: str
    embeddings: List[Embedding]
    request_id: Optional[str] = None
    timing_ms: Optional[dict] = None


class ReferenceInfo(BaseModel):
    facial_area: FacialArea
    confidence: float


class VerifyItem(BaseModel):
    index: int
    verified: Optional[bool] = None
    distance: Optional[float] = None
    threshold: Optional[float] = None
    error: Optional[str] = None


class VerifyResponse(BaseModel):
    model: str
    detector: str
    distance_metric: str
    reference: ReferenceInfo
    results: List[VerifyItem]
    request_id: Optional[str] = None
    timing_ms: Optional[dict] = None

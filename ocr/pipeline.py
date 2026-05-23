from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def decode_jpeg_to_gray(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    if not jpeg_bytes:
        return None
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    if buf.size == 0:
        return None
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def preprocess_gray(gray: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return gray
    denoised = cv2.bilateralFilter(gray, 5, 40, 40)
    return cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str, paragraph_min_chars: int = 6) -> List[str]:
    clean = normalize_text(text)
    if not clean:
        return []

    blocks = [blk.strip() for blk in clean.split("\n\n") if blk.strip()]
    if not blocks:
        blocks = [clean]

    paragraphs: List[str] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        merged = " ".join(lines)
        if len(merged) >= int(paragraph_min_chars):
            paragraphs.append(merged)

    return paragraphs

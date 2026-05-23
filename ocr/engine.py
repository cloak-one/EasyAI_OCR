from __future__ import annotations

import concurrent.futures
import time
from typing import Callable, Optional

import numpy as np

from ocr.config import LocalOCRConfig
from ocr.pipeline import decode_jpeg_to_gray, preprocess_gray, split_paragraphs
from ocr.types import OCRRecognizeOutput, OCRResult

Recognizer = Callable[[np.ndarray], OCRRecognizeOutput]


def _rapidocr_fallback_recognize(jpeg_bytes: bytes) -> OCRRecognizeOutput:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:
        raise RuntimeError("rapidocr_onnxruntime is not available") from exc

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("cv2 is not available for rapidocr fallback") from exc

    arr = np.frombuffer(jpeg_bytes or b"", dtype=np.uint8)
    if arr.size == 0:
        return OCRRecognizeOutput(text="", confidence=0.0)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return OCRRecognizeOutput(text="", confidence=0.0)

    ocr = RapidOCR()
    result, _ = ocr(bgr)
    if not result:
        return OCRRecognizeOutput(text="", confidence=0.0)

    text_list = []
    for item in result:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text = str(item[1] or "").strip()
        if text:
            text_list.append(text)
    return OCRRecognizeOutput(text="\n".join(text_list).strip(), confidence=0.0)


def _default_tesseract_recognizer(gray_image: np.ndarray) -> OCRRecognizeOutput:
    try:
        import pytesseract
        from pytesseract import Output
    except Exception as exc:
        raise RuntimeError("pytesseract is not available") from exc

    data = pytesseract.image_to_data(gray_image, output_type=Output.DICT)
    text_chunks = []
    conf_values = []
    for idx, token in enumerate(data.get("text", [])):
        token = (token or "").strip()
        if not token:
            continue
        text_chunks.append(token)
        try:
            conf = float(data.get("conf", ["-1"])[idx])
        except Exception:
            conf = -1.0
        if conf >= 0:
            conf_values.append(conf / 100.0)

    text = " ".join(text_chunks).strip()
    confidence = sum(conf_values) / len(conf_values) if conf_values else 0.0
    return OCRRecognizeOutput(text=text, confidence=confidence)


class LocalOCREngine:
    def __init__(self, config: Optional[LocalOCRConfig] = None, recognizer: Optional[Recognizer] = None):
        self.config = config or LocalOCRConfig()
        self.recognizer = recognizer or _default_tesseract_recognizer

    def run_on_jpeg(self, jpeg_bytes: bytes) -> OCRResult:
        started = time.perf_counter()

        if not jpeg_bytes:
            return OCRResult(ok=False, error_code="empty_input", error_message="jpeg bytes are empty")

        gray = decode_jpeg_to_gray(jpeg_bytes)
        if gray is None:
            return OCRResult(ok=False, error_code="decode_failed", error_message="failed to decode jpeg")

        if self.config.enable_preprocess:
            gray = preprocess_gray(gray)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.recognizer, gray)
                recognized = future.result(timeout=max(0.1, float(self.config.timeout_sec)))
        except concurrent.futures.TimeoutError:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return OCRResult(
                ok=False,
                elapsed_ms=elapsed_ms,
                engine=self.config.default_engine_name,
                error_code="timeout",
                error_message="ocr recognize timeout",
            )
        except RuntimeError as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return OCRResult(
                ok=False,
                elapsed_ms=elapsed_ms,
                engine=self.config.default_engine_name,
                error_code="engine_unavailable",
                error_message=str(exc),
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            return OCRResult(
                ok=False,
                elapsed_ms=elapsed_ms,
                engine=self.config.default_engine_name,
                error_code="exception",
                error_message=str(exc),
            )

        if not (recognized.text or "").strip():
            try:
                recognized = _rapidocr_fallback_recognize(jpeg_bytes)
                if (recognized.text or "").strip():
                    self.config.default_engine_name = "rapidocr_onnxruntime"
            except Exception:
                pass

        text = (recognized.text or "").strip()
        paragraphs = split_paragraphs(text, paragraph_min_chars=self.config.paragraph_min_chars)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        if len(text) < int(self.config.min_text_chars):
            if "tesseract" in str(self.config.default_engine_name).lower() or not text:
                try:
                    fallback_recognized = _rapidocr_fallback_recognize(jpeg_bytes)
                    fallback_text = (fallback_recognized.text or "").strip()
                    if fallback_text:
                        text = fallback_text
                        paragraphs = split_paragraphs(text, paragraph_min_chars=self.config.paragraph_min_chars)
                        return OCRResult(
                            ok=True,
                            text=text,
                            paragraphs=paragraphs,
                            confidence=float(fallback_recognized.confidence or 0.0),
                            elapsed_ms=elapsed_ms,
                            engine="rapidocr_onnxruntime",
                        )
                except Exception:
                    pass
            return OCRResult(
                ok=False,
                text=text,
                paragraphs=paragraphs,
                confidence=float(recognized.confidence or 0.0),
                elapsed_ms=elapsed_ms,
                engine=self.config.default_engine_name,
                error_code="no_text",
                error_message="recognized text is too short",
            )

        return OCRResult(
            ok=True,
            text=text,
            paragraphs=paragraphs,
            confidence=float(recognized.confidence or 0.0),
            elapsed_ms=elapsed_ms,
            engine=self.config.default_engine_name,
        )

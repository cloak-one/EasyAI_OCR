from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalOCRConfig:
    timeout_sec: float = 3.0
    min_text_chars: int = 20
    enable_preprocess: bool = True
    paragraph_min_chars: int = 6
    default_engine_name: str = "local_tesseract"

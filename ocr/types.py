from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OCRResult:
    ok: bool
    text: str = ""
    paragraphs: List[str] = field(default_factory=list)
    confidence: float = 0.0
    elapsed_ms: int = 0
    engine: str = "local"
    error_code: str = ""
    error_message: str = ""


@dataclass
class OCRRecognizeOutput:
    text: str
    confidence: Optional[float] = None

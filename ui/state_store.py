#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading
import time
from collections import deque
from typing import Deque, Dict, List, Optional


class UIStateStore:
    def __init__(self, max_history: int = 50, max_errors: int = 20):
        self._lock = threading.Lock()
        self._max_history = max(1, int(max_history))
        self._max_errors = max(1, int(max_errors))

        self._frame_jpeg: Optional[bytes] = None
        self._frame_ts: float = 0.0

        self._system_state: str = "IDLE"
        self._partial_text: str = ""
        self._user_text: str = ""
        self._model_delta_text: str = ""

        self._history: Deque[Dict[str, str]] = deque(maxlen=self._max_history)
        self._errors: Deque[str] = deque(maxlen=self._max_errors)
        self._updated_at: float = time.time()

    def _touch(self):
        self._updated_at = time.time()

    def update_frame(self, jpeg_bytes: bytes):
        if not jpeg_bytes:
            return
        with self._lock:
            self._frame_jpeg = bytes(jpeg_bytes)
            self._frame_ts = time.time()
            self._touch()

    def set_system_state(self, state: str):
        with self._lock:
            self._system_state = str(state or "")
            self._touch()

    def set_partial_text(self, text: str):
        with self._lock:
            self._partial_text = str(text or "")
            self._touch()

    def set_user_text(self, text: str):
        with self._lock:
            self._user_text = str(text or "")
            self._model_delta_text = ""
            self._touch()

    def append_model_delta(self, text: str):
        if not text:
            return
        with self._lock:
            self._model_delta_text += str(text)
            self._touch()

    def set_model_final(self, text: str):
        final_text = str(text or "")
        with self._lock:
            self._model_delta_text = final_text
            self._history.appendleft({
                "user": self._user_text,
                "model": final_text,
            })
            self._touch()

    def push_error(self, message: str):
        msg = str(message or "")
        if not msg:
            return
        with self._lock:
            self._errors.appendleft(msg)
            self._touch()

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            if self._frame_jpeg is None:
                return None
            return bytes(self._frame_jpeg)

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            return {
                "updated_at": self._updated_at,
                "frame_ts": self._frame_ts,
                "system_state": self._system_state,
                "partial_text": self._partial_text,
                "user_text": self._user_text,
                "model_text_delta": self._model_delta_text,
                "history": list(self._history),
                "errors": list(self._errors),
            }

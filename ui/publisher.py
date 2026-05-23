#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from ui.state_store import UIStateStore


class UIPublisher:
    def __init__(self, state_store: UIStateStore):
        self._store = state_store

    def publish_frame(self, jpeg_bytes: bytes):
        try:
            self._store.update_frame(jpeg_bytes)
        except Exception:
            pass

    def publish_state(self, state: str):
        try:
            self._store.set_system_state(state)
        except Exception:
            pass

    def publish_partial_text(self, text: str):
        try:
            self._store.set_partial_text(text)
        except Exception:
            pass

    def publish_user_text(self, text: str):
        try:
            self._store.set_user_text(text)
        except Exception:
            pass

    def publish_model_delta(self, text: str):
        try:
            self._store.append_model_delta(text)
        except Exception:
            pass

    def publish_model_final(self, text: str):
        try:
            self._store.set_model_final(text)
        except Exception:
            pass

    def publish_error(self, message: str):
        try:
            self._store.push_error(message)
        except Exception:
            pass

    def snapshot(self) -> Optional[dict]:
        try:
            return self._store.snapshot()
        except Exception:
            return None

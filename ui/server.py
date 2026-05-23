#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse

from ui.state_store import UIStateStore


class _UIRequestHandler(BaseHTTPRequestHandler):
    assets_dir: str = ""
    state_store: Optional[UIStateStore] = None

    def log_message(self, fmt: str, *args):
        return

    def _serve_file(self, file_name: str, content_type: str):
        file_path = os.path.join(self.assets_dir, file_name)
        if not os.path.exists(file_path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            with open(file_path, "rb") as f:
                raw = f.read()
        except Exception:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_state(self):
        if self.state_store is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        payload = self.state_store.snapshot()
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_frame(self):
        if self.state_store is None:
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
            return
        jpeg = self.state_store.get_frame()
        if not jpeg:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(jpeg)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_file("visualizer.html", "text/html; charset=utf-8")
            return
        if path == "/visualizer.js":
            self._serve_file("visualizer.js", "application/javascript; charset=utf-8")
            return
        if path == "/visualizer.css":
            self._serve_file("visualizer.css", "text/css; charset=utf-8")
            return
        if path == "/placeholder.jpg":
            self._serve_file("placeholder.jpg", "image/jpeg")
            return
        if path == "/api/state":
            self._serve_state()
            return
        if path == "/api/frame.jpg":
            self._serve_frame()
            return

        self.send_error(HTTPStatus.NOT_FOUND)


class UIServer:
    def __init__(self, state_store: UIStateStore, host: str = "127.0.0.1", port: int = 8765):
        self._state_store = state_store
        self._host = host
        self._port = int(port)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        if self._httpd is not None:
            host, port = self._httpd.server_address[:2]
            return f"http://{host}:{port}"
        return f"http://{self._host}:{self._port}"

    def start(self):
        if self._httpd is not None:
            return

        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        if not os.path.isdir(assets_dir):
            raise RuntimeError(f"UI assets directory not found: {assets_dir}")

        class Handler(_UIRequestHandler):
            pass

        Handler.assets_dir = assets_dir
        Handler.state_store = self._state_store

        self._httpd = ThreadingHTTPServer((self._host, self._port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._httpd is None:
            return
        try:
            self._httpd.shutdown()
            self._httpd.server_close()
        finally:
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

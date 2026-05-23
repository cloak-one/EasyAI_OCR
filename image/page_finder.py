# page_finder.py
# -*- coding: utf-8 -*-

import base64
import json
import os
import re
import time
import importlib.util
from dataclasses import dataclass
from collections import deque
from typing import Optional, Deque, Dict, Any, Callable


_CN_NUM = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _resolve_stream_chat() -> Callable:
    try:
        mod = __import__("omni_client", fromlist=["stream_chat"])
        fn = getattr(mod, "stream_chat", None)
        if fn:
            return fn
    except Exception:
        pass

    module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audio", "omni_client.py"))
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location("_image_pagefinder_omni_client", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, "stream_chat", None)
            if fn:
                return fn

    raise ImportError("无法加载 stream_chat")


def cn_to_int(token: str) -> Optional[int]:
    s = (token or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)

    total = 0
    current = 0
    has_num = False

    for ch in s:
        if ch in _CN_NUM:
            current = _CN_NUM[ch]
            has_num = True
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
            has_num = True
        else:
            return None

    total += current
    if not has_num:
        return None
    return total if total > 0 else None


def parse_page_token(token: str) -> Optional[int]:
    t = (token or "").strip()
    if not t:
        return None

    val = cn_to_int(t)
    if val is not None:
        return val

    m = re.search(r"([0-9]+)", t)
    if m:
        return int(m.group(1))
    return None


@dataclass
class PageRecord:
    ts: float
    page_no: Optional[int]
    snippet: str
    raw_text: str


class BookPageFinder:
    """书页检索：当前页识别 + 历史缓存匹配 + 引导语生成。"""

    def __init__(self, max_records: int = 80):
        self.records: Deque[PageRecord] = deque(maxlen=max_records)

    async def _extract_page_info(self, jpeg_bytes: bytes) -> Dict[str, Any]:
        stream_chat = _resolve_stream_chat()
        img_b64 = base64.b64encode(jpeg_bytes).decode("ascii")

        prompt = (
            "你是OCR页码抽取器。请识别图中可见文字，并重点识别页码。"
            "严格输出JSON，不要输出其他文字。"
            "如果无法确认页码，page_no必须为null，禁止输出0或负数，禁止猜测。"
            "JSON格式：{\"page_no\": 正整数或null, \"snippet\": \"不超过30字\", \"confidence\": 0到1之间数字}"
        )

        content_list = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]

        chunks = []
        async for piece in stream_chat(
            content_list,
            audio_format="wav",
            modalities=["text"],
            # model="qwen-vl-ocr",
        ):
            if piece.text_delta:
                chunks.append(piece.text_delta)

        raw = "".join(chunks).strip()
        page_no = None
        snippet = ""

        try:
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                payload = json.loads(m.group(0))
                page_val = payload.get("page_no")
                if isinstance(page_val, str):
                    page_no = parse_page_token(page_val)
                elif isinstance(page_val, (int, float)):
                    page_no = int(page_val)
                snippet = str(payload.get("snippet", "") or "").strip()
        except Exception:
            page_no = None
            snippet = ""

        if page_no is None:
            m = re.search(r"第\s*([0-9一二三四五六七八九十百千万两]+)\s*页", raw)
            if m:
                page_no = parse_page_token(m.group(1))
            if page_no is None:
                m2 = re.search(r"\b([0-9]{1,4})\b", raw)
                if m2:
                    page_no = int(m2.group(1))

        return {"page_no": page_no, "snippet": snippet, "raw_text": raw}

    def _append_record(self, page_no: Optional[int], snippet: str, raw_text: str):
        if self.records and self.records[-1].page_no == page_no and self.records[-1].snippet == snippet:
            return
        self.records.append(PageRecord(ts=time.time(), page_no=page_no, snippet=snippet, raw_text=raw_text))

    def _latest_page(self) -> Optional[int]:
        for rec in reversed(self.records):
            if rec.page_no is not None and rec.page_no > 0:
                return rec.page_no
        return None

    def build_no_camera_guidance(self, target_page: int) -> str:
        return f"当前没有相机画面，无法定位第{target_page}页。请先对准书页拍摄。"

    def _build_guidance(self, target_page: int, current_page: Optional[int]) -> str:
        if current_page is None or current_page <= 0:
            return (
                f"暂未识别到页码，无法直接定位第{target_page}页。"
                "请将页码区域放到画面下方并保持稳定，我会继续识别。"
            )

        if current_page == target_page:
            return f"已定位到第{target_page}页。请保持当前页面，我可以开始阅读或总结。"

        delta = target_page - current_page
        if delta > 0:
            return f"当前大约在第{current_page}页，目标是第{target_page}页。请继续向后翻约{delta}页。"
        return f"当前大约在第{current_page}页，目标是第{target_page}页。请向前翻约{abs(delta)}页。"

    async def find_page(self, target_page: int, jpeg_bytes: bytes) -> Dict[str, Any]:
        info = await self._extract_page_info(jpeg_bytes)
        current_page = info.get("page_no")
        snippet = info.get("snippet", "")
        raw_text = info.get("raw_text", "")

        self._append_record(current_page, snippet, raw_text)
        # if current_page is None or current_page <= 0:
        #     current_page = self._latest_page()

        message = self._build_guidance(target_page, current_page)
        return {
            "target_page": target_page,
            "current_page": current_page,
            "snippet": snippet,
            "message": message,
            "history_size": len(self.records),
        }

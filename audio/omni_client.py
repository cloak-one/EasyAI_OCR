# omni_client.py
# -*- coding: utf-8 -*-
import asyncio
import os, base64
import threading
from typing import AsyncGenerator, Dict, Any, List, Optional, Tuple

from openai import OpenAI

# ===== OpenAI 兼容（达摩院 DashScope 兼容模式）=====
from config import DASHSCOPE_API_KEY as API_KEY

QWEN_MODEL = "qwen3.5-omni-flash"
DEFAULT_OMNI_VOICE = "Tina"

# 兼容模式
oai_client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

_warmed_up = False

class OmniStreamPiece:
    """对外的统一增量数据：text/audio 二选一或同时。"""
    def __init__(self, text_delta: Optional[str] = None, audio_b64: Optional[str] = None):
        self.text_delta = text_delta
        self.audio_b64  = audio_b64


def warmup_chat_client(model: Optional[str] = None) -> None:
    """后台预热首轮 Omni 请求，降低第一条用户指令的冷启动延迟。"""
    global _warmed_up
    if _warmed_up:
        return

    req: Dict[str, Any] = {
        "model": model or QWEN_MODEL,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "你好"}]}],
        "modalities": ["text"],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        oai_client.chat.completions.create(**req)
        _warmed_up = True
        print("[OMNI] warm-up complete")
    except Exception as e:
        print(f"[OMNI] warm-up failed: {e}")


async def warmup_chat_client_async(model: Optional[str] = None) -> None:
    await asyncio.to_thread(warmup_chat_client, model)

async def stream_chat(
    content_list: List[Dict[str, Any]],
    voice: str = DEFAULT_OMNI_VOICE,
    audio_format: str = "wav",
    modalities: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> AsyncGenerator[OmniStreamPiece, None]:
    """
    发起一轮 Omni-Turbo ChatCompletions 流式对话：
    - content_list: OpenAI chat 的 content，多模态（image_url/text）
    - 以 stream=True 返回
    - 增量产出：OmniStreamPiece(text_delta=?, audio_b64=?)
    """
    req_modalities = modalities or ["text", "audio"]
    req: Dict[str, Any] = {
        "model": model or QWEN_MODEL,
        "messages": [{"role": "user", "content": content_list}],
        "modalities": req_modalities,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if "audio" in req_modalities:
        req["audio"] = {"voice": voice, "format": audio_format}

    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Tuple[str, Optional[OmniStreamPiece], Optional[BaseException]]]" = asyncio.Queue()
    stop_event = threading.Event()
    stream_holder: Dict[str, Any] = {}

    def _enqueue(
        kind: str,
        piece: Optional[OmniStreamPiece] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, piece, error))

    def _worker() -> None:
        try:
            completion = oai_client.chat.completions.create(**req)
            stream_holder["completion"] = completion

            for chunk in completion:
                if stop_event.is_set():
                    break

                text_delta: Optional[str] = None
                audio_b64: Optional[str] = None

                if getattr(chunk, "choices", None):
                    c0 = chunk.choices[0]
                    delta = getattr(c0, "delta", None)
                    # 文本增量
                    if delta and getattr(delta, "content", None):
                        piece = delta.content
                        if piece:
                            text_delta = piece
                    # 音频分片
                    if delta and getattr(delta, "audio", None):
                        aud = delta.audio
                        audio_b64 = aud.get("data") if isinstance(aud, dict) else getattr(aud, "data", None)
                    if audio_b64 is None:
                        msg = getattr(c0, "message", None)
                        if msg and getattr(msg, "audio", None):
                            ma = msg.audio
                            audio_b64 = ma.get("data") if isinstance(ma, dict) else getattr(ma, "data", None)

                if (text_delta is not None) or (audio_b64 is not None):
                    _enqueue("piece", OmniStreamPiece(text_delta=text_delta, audio_b64=audio_b64))
        except BaseException as e:
            _enqueue("error", error=e)
        finally:
            completion = stream_holder.get("completion")
            if completion is not None and hasattr(completion, "close"):
                try:
                    completion.close()
                except Exception:
                    pass
            _enqueue("done")

    worker = threading.Thread(target=_worker, name="omni-stream-worker", daemon=True)
    worker.start()

    try:
        while True:
            kind, piece, error = await queue.get()
            if kind == "piece" and piece is not None:
                yield piece
            elif kind == "error" and error is not None:
                raise error
            elif kind == "done":
                break
    finally:
        stop_event.set()
        completion = stream_holder.get("completion")
        if completion is not None and hasattr(completion, "close"):
            try:
                completion.close()
            except Exception:
                pass

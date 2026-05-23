# app_main.py
# -*- coding: utf-8 -*-
import os
import sys
import re
import json
import io
import time
import wave
import atexit
import asyncio
import base64
import threading
import traceback
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import uvicorn
from dashscope import audio as dash_audio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

import bridge_io
from audio.asr_core import ASRCallback, set_current_recognition, stop_current_recognition
from audio.audio_player import (
    AUDIO_OUTPUT_MODE,
    initialize_audio_system,
    play_pcm16_audio_local,
    play_pcm8k_audio,
    play_voice_text,
    shutdown_audio_system,
    stop_audio_playback,
)
from audio.audio_stream import (
    BYTES_PER_20MS_16K,
    cancel_current_ai,
    hard_reset_audio,
    is_playing_now,
    register_stream_route,
)
from audio.omni_client import DEFAULT_OMNI_VOICE, stream_chat
from audio.audio_utils import PCMResampler, adjust_volume
from image.page_finder import BookPageFinder, parse_page_token
from intent_rules import is_visual_intent
from state_master import StateMaster

if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

from config import DASHSCOPE_API_KEY as API_KEY

MODEL = "paraformer-realtime-v2"
SAMPLE_RATE = 16000
AUDIO_FMT = "pcm"
CHUNK_MS = 20
BYTES_CHUNK = SAMPLE_RATE * CHUNK_MS // 1000 * 2
SILENCE_20MS = bytes(BYTES_CHUNK)

CAM_LOW_PROFILE = {
    "type": "camera_profile",
    "profile": "low",
    "framesize": "QVGA",
    "width": 320,
    "height": 240,
    "fps": 5,
    "jpeg_quality": 45,
}
CAM_HIGH_PROFILE = {
    "type": "camera_profile",
    "profile": "capture",
    "framesize": "XGA",
    "width": 1280,
    "height": 1024,
    "fps": 3,
    "jpeg_quality": 10,
}

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    main_loop = asyncio.get_event_loop()

    def _sender(jpeg_bytes: bytes):
        try:
            if main_loop.is_closed() or not camera_viewers:
                return
            async def _broadcast():
                dead = []
                for viewer_ws in list(camera_viewers):
                    try:
                        await viewer_ws.send_bytes(jpeg_bytes)
                    except Exception:
                        dead.append(viewer_ws)
                for d in dead:
                    camera_viewers.discard(d)
            asyncio.run_coroutine_threadsafe(_broadcast(), main_loop)
        except Exception as e:
            if "Event loop is closed" not in str(e):
                print(f"[BRIDGE] sender error: {e}")

    bridge_io.set_sender(_sender)
    threading.Thread(target=lambda: initialize_audio_system(), daemon=True).start()

    try:
        yield
    except asyncio.CancelledError:
        pass
    finally:
        print("[SHUTDOWN] 开始清理资源...")
        try:
            await hard_reset_audio("shutdown")
            recorder.close()
            shutdown_audio_system()

            # 取消所有剩余的 asyncio.Task（WebSocket handlers、heartbeat 等），等待其退出
            current_task = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current_task and not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                try:
                    await asyncio.gather(*pending, return_exceptions=True)
                except asyncio.CancelledError:
                    pass
        except asyncio.CancelledError:
            pass

        print("[SHUTDOWN] 资源清理完成")

app = FastAPI(lifespan=lifespan)
app.mount("/resources/static", StaticFiles(directory="resources/static"), name="static")

ui_clients: Dict[int, WebSocket] = {}
current_partial: str = ""
recent_finals: List[str] = []
recent_captures: List[str] = []

RECENT_MAX = 50

last_frames: Deque[Tuple[float, bytes]] = deque(maxlen=20)
camera_viewers: Set[WebSocket] = set()
command_camera_clients: Set[WebSocket] = set()
command_streaming_enabled: bool = False
camera_frame_event = asyncio.Event()

esp32_camera_ws: Optional[WebSocket] = None
esp32_audio_ws: Optional[WebSocket] = None

interrupt_lock = asyncio.Lock()
orchestrator = StateMaster()
page_finder = BookPageFinder()


class CommandCaptureRecorder:
    """命令触发式录制器：每次命令单独落盘，不做全程录像。"""

    def __init__(self, save_root: str):
        self.save_root = save_root
        self.session_dir = self._create_session_dir()
        self.turn_idx = 0
        self.lock = threading.Lock()
        self.mic_wf: Optional[wave.Wave_write] = None
        self.ai_wf: Optional[wave.Wave_write] = None
        self.turn_dir: Optional[str] = None

    def _create_session_dir(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        d = os.path.join(self.save_root, f"session_{ts}")
        os.makedirs(d, exist_ok=True)
        return d

    def _open_wav(self, path: str):
        wf = wave.open(path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        return wf

    def start_turn(self, user_text: str, intent: str, jpeg_bytes: Optional[bytes]):
        with self.lock:
            self.finish_turn_locked()
            self.turn_idx += 1
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.turn_dir = os.path.join(self.session_dir, f"turn_{self.turn_idx:04d}_{ts}")
            os.makedirs(self.turn_dir, exist_ok=True)

            meta = {
                "timestamp": ts,
                "intent": intent,
                "user_text": user_text,
            }
            with open(os.path.join(self.turn_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            if jpeg_bytes:
                with open(os.path.join(self.turn_dir, "command_image.jpg"), "wb") as f:
                    f.write(jpeg_bytes)

            self.mic_wf = self._open_wav(os.path.join(self.turn_dir, "mic_input.wav"))
            self.ai_wf = self._open_wav(os.path.join(self.turn_dir, "ai_output.wav"))

    def add_mic(self, pcm16: bytes):
        with self.lock:
            if self.mic_wf is not None and pcm16:
                self.mic_wf.writeframes(pcm16)

    def add_ai(self, pcm16: bytes):
        with self.lock:
            if self.ai_wf is not None and pcm16:
                self.ai_wf.writeframes(pcm16)

    def finish_turn_locked(self):
        if self.mic_wf is not None:
            try:
                self.mic_wf.close()
            except Exception:
                pass
            self.mic_wf = None

        if self.ai_wf is not None:
            try:
                self.ai_wf.close()
            except Exception:
                pass
            self.ai_wf = None

        self.turn_dir = None

    def finish_turn(self):
        with self.lock:
            self.finish_turn_locked()

    def close(self):
        self.finish_turn()


recorder = CommandCaptureRecorder(save_root=os.path.join(os.path.dirname(__file__), "save"))


def cleanup_on_exit():
    try:
        recorder.close()
    except Exception:
        pass
    try:
        shutdown_audio_system()
    except Exception:
        pass


atexit.register(cleanup_on_exit)


_device_state: dict = {
    "camera_connected": False,
    "audio_connected": False,
    "asr_active": False,
    "current_intent": "idle",
}

async def _broadcast_device_state():
    await ui_broadcast_raw("STATE:" + json.dumps(_device_state, ensure_ascii=False))


async def ui_broadcast_raw(msg: str):
    dead = []
    for k, ws in list(ui_clients.items()):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(k)
    for k in dead:
        ui_clients.pop(k, None)


async def ui_broadcast_partial(text: str):
    global current_partial
    current_partial = text
    await ui_broadcast_raw("PARTIAL:" + text)


async def ui_broadcast_final(text: str):
    global current_partial, recent_finals
    current_partial = ""
    recent_finals.append(text)
    if len(recent_finals) > RECENT_MAX:
        recent_finals = recent_finals[-RECENT_MAX:]
    await ui_broadcast_raw("FINAL:" + text)
    print(f"[ASR/AI FINAL] {text}", flush=True)


async def ui_broadcast_capture(b64: str):
    global recent_captures
    recent_captures.append(b64)
    if len(recent_captures) > 12:
        recent_captures = recent_captures[-8:]
    await ui_broadcast_raw("CAPTURE:" + b64)


def _extract_find_page_target(user_text: str) -> Optional[int]:
    m = re.search(r"第\s*([0-9一二三四五六七八九十百千万两\d]+)\s*页", user_text)
    if not m:
        return None
    return parse_page_token(m.group(1))


async def _request_camera_profile(profile: dict):
    if not esp32_camera_ws or esp32_camera_ws.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await esp32_camera_ws.send_text("CAMERA_PROFILE:" + json.dumps(profile, ensure_ascii=False))
        return True
    except Exception:
        return False


async def _broadcast_command_stream_state(reason: str = ""):
    payload = {
        "type": "command_stream_state",
        "enabled": command_streaming_enabled,
        "reason": reason,
    }
    text = "STATE:" + json.dumps(payload, ensure_ascii=False)
    dead = []
    for ws in list(command_camera_clients):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        command_camera_clients.discard(ws)


async def _set_command_streaming(enabled: bool, reason: str = ""):
    global command_streaming_enabled
    if command_streaming_enabled == bool(enabled):
        if reason:
            await _broadcast_command_stream_state(reason)
        return

    command_streaming_enabled = bool(enabled)

    # 向 ESP32 发控制指令（若固件未实现则忽略，不影响主流程）
    if esp32_camera_ws and esp32_camera_ws.client_state == WebSocketState.CONNECTED:
        try:
            ctrl = "START" if command_streaming_enabled else "STOP"
            await esp32_camera_ws.send_text(f"CAMERA_STREAM:{ctrl}")
        except Exception:
            pass

    await _broadcast_command_stream_state(reason)


def _pick_best_command_jpeg(max_frames: int = 8, min_ts: float = 0.0) -> Optional[bytes]:
    """仅在已到达的帧缓存里选最清晰帧，无法直接控制硬件抓拍。"""
    best_jpeg = None
    best_score = -1.0
    snapshot = list(last_frames)[-max_frames:]
    for ts, jpeg in snapshot:
        if ts < min_ts:
            continue
        try:
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if score > best_score:
                best_score = score
                best_jpeg = jpeg
        except Exception:
            continue
    return best_jpeg


def _resize_to_fit(bgr: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w <= 0 or h <= 0:
        return bgr
    scale = min(max_width / w, max_height / h, 1.0)
    if scale >= 1.0:
        return bgr
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)


async def _wait_for_camera_ready(timeout_ms: int = 1500) -> bool:
    deadline = time.time() + max(0.1, timeout_ms / 1000.0)
    while time.time() < deadline:
        if esp32_camera_ws and esp32_camera_ws.client_state == WebSocketState.CONNECTED:
            return True
        await asyncio.sleep(0.05)
    return bool(esp32_camera_ws and esp32_camera_ws.client_state == WebSocketState.CONNECTED)


async def _acquire_command_jpeg(wait_ms: int = 700) -> Optional[bytes]:
    start_ts = time.time()
    ready = await _wait_for_camera_ready(timeout_ms=1800)
    if not ready:
        print("[CAPTURE] camera websocket not ready before capture")
        return None

    try:
        last_frames.clear()
    except Exception:
        pass
    camera_frame_event.clear()
    await _set_command_streaming(True, reason="capture_begin")
    await _request_camera_profile(CAM_HIGH_PROFILE)
    await asyncio.sleep(0.45)
    try:
        last_frames.clear()
    except Exception:
        pass
    start_ts = time.time()
    camera_frame_event.clear()

    try:
        await asyncio.wait_for(camera_frame_event.wait(), timeout=max(0.8, wait_ms / 1000.0))
    except asyncio.TimeoutError:
        pass

    best = _pick_best_command_jpeg(max_frames=10, min_ts=start_ts)
    if best is None:
        print("[CAPTURE] no fresh frame on first attempt, retrying")
        start_ts = time.time()
        try:
            last_frames.clear()
        except Exception:
            pass
        camera_frame_event.clear()
        await _set_command_streaming(True, reason="capture_retry")
        await _request_camera_profile(CAM_HIGH_PROFILE)
        await asyncio.sleep(0.35)
        try:
            last_frames.clear()
        except Exception:
            pass
        start_ts = time.time()
        camera_frame_event.clear()
        try:
            await asyncio.wait_for(camera_frame_event.wait(), timeout=0.9)
        except asyncio.TimeoutError:
            pass
        best = _pick_best_command_jpeg(max_frames=12, min_ts=start_ts)
    if best is None:
        best = _pick_best_command_jpeg(max_frames=10)

    if best:
        print(f"[CAPTURE] captured jpeg bytes={len(best)}")
        try:
            arr = np.frombuffer(best, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                bgr = _resize_to_fit(bgr, 960, 720)
                ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    b64 = base64.b64encode(enc.tobytes()).decode("ascii")
                    asyncio.get_event_loop().create_task(ui_broadcast_capture(b64))
        except Exception:
            pass
    else:
        print("[CAPTURE] failed: no jpeg available")

    if esp32_camera_ws and esp32_camera_ws.client_state == WebSocketState.CONNECTED:
        try:
            await esp32_camera_ws.send_text("CAMERA_STREAM:STOP")
            await _request_camera_profile(CAM_LOW_PROFILE)
        except Exception:
            pass
    return best


def _decode_omni_audio_chunk(audio_b64: str) -> Tuple[bytes, int]:
    raw = base64.b64decode(audio_b64)
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(raw), "rb") as wf:
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                channels = wf.getnchannels()
                frames = wf.readframes(wf.getnframes())
            if sample_width == 2 and channels == 1:
                return frames, sample_rate
            if sample_width == 2 and channels > 1:
                samples = np.frombuffer(frames, dtype=np.int16)
                usable = (len(samples) // channels) * channels
                if usable > 0:
                    samples = samples[:usable].reshape(-1, channels).mean(axis=1)
                    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes(), sample_rate
        except Exception as e:
            print(f"[AI AUDIO] wav decode failed, fallback raw pcm: {e}")
    return raw, 24000


async def _handle_find_page(user_text: str, target_page: int):
    orchestrator.set_intent_state("find_page")
    jpeg = await _acquire_command_jpeg()
    recorder.start_turn(user_text=user_text, intent="find_page", jpeg_bytes=jpeg)

    try:
        if not jpeg:
            msg = page_finder.build_no_camera_guidance(target_page)
            await ui_broadcast_final("[页码] " + msg)
            play_voice_text(msg)
            return

        report = await page_finder.find_page(target_page=target_page, jpeg_bytes=jpeg)
        current_page = report.get("current_page")

        if current_page is None or current_page <= 0:
            await asyncio.sleep(0.5)
            jpeg2 = await _acquire_command_jpeg()
            if jpeg2:
                report = await page_finder.find_page(target_page=target_page, jpeg_bytes=jpeg2)

        msg = str(report.get("message") or "").strip() or f"未能定位第{target_page}页。"
        await ui_broadcast_final("[页码] " + msg)
        play_voice_text(msg)
    except Exception as e:
        err = f"页码定位失败：{e}"
        await ui_broadcast_final("[页码] " + err)
        play_voice_text("页码定位失败，请重试。")
    finally:
        recorder.finish_turn()
        orchestrator.set_intent_state("chat")
        await _set_command_streaming(False, reason="find_page_done")


async def full_system_reset(
    reason: str = "",
    notify_audio_device: bool = True,
    stop_asr: bool = True,
):
    stop_audio_playback()
    await hard_reset_audio(reason or "full_system_reset")
    if stop_asr:
        await stop_current_recognition()

    global current_partial, recent_finals, recent_captures
    current_partial = ""
    recent_finals = []
    recent_captures = []

    try:
        last_frames.clear()
    except Exception:
        pass

    if notify_audio_device:
        try:
            if esp32_audio_ws and (esp32_audio_ws.client_state == WebSocketState.CONNECTED):
                await esp32_audio_ws.send_text("RESET")
        except Exception:
            pass

    if esp32_camera_ws and esp32_camera_ws.client_state == WebSocketState.CONNECTED:
        try:
            await esp32_camera_ws.send_text("CAMERA_STREAM:STOP")
        except Exception:
            pass
    recorder.finish_turn()
    print("[SYSTEM] full reset done.", flush=True)


async def start_ai_with_text_custom(user_text: str):
    text = (user_text or "").strip()
    if not text:
        return

    target_page = _extract_find_page_target(text)
    if target_page is not None and any(k in text for k in ["找", "定位", "翻到", "第"]):
        await _set_command_streaming(True, reason="asr_final_find_page")
        await _handle_find_page(text, target_page)
        return

    orchestrator.set_intent_state("chat")
    visual_intent = is_visual_intent(text)
    await start_ai_with_text(
        text,
        manage_command_stream=visual_intent,
        include_image=visual_intent,
    )


async def start_ai_with_text(
    user_text: str,
    manage_command_stream: bool = False,
    include_image: bool = True,
):
    if manage_command_stream:
        await _set_command_streaming(True, reason="asr_final_chat")

    jpeg = await _acquire_command_jpeg() if include_image else None
    if include_image and not jpeg:
        try:
            await ui_broadcast_final("[AI] 当前没有获取到摄像头画面，请稍后重试。")
        except Exception:
            pass
        if manage_command_stream:
            await _set_command_streaming(False, reason="capture_failed")
        return
    recorder.start_turn(user_text=user_text, intent="chat", jpeg_bytes=jpeg)

    async def _runner():
        txt_buf: List[str] = []
        audio_state = {
            "current_sr": 24000,
            "chunks_received": 0,
        }
        resamplers = {
            "to_8k": PCMResampler(24000, 8000),
            "to_16k": PCMResampler(24000, 16000),
        }

        content_list = []
        if jpeg:
            try:
                img_b64 = base64.b64encode(jpeg).decode("ascii")
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    }
                )
            except Exception:
                pass
        content_list.append({"type": "text", "text": user_text})

        async def _do_stream():
            async for piece in stream_chat(content_list, voice=DEFAULT_OMNI_VOICE, audio_format="wav"):
                if piece.text_delta:
                    txt_buf.append(piece.text_delta)
                    try:
                        await ui_broadcast_partial("[AI] " + "".join(txt_buf))
                    except Exception:
                        pass
                if piece.audio_b64:
                    try:
                        pcm24, chunk_sr = _decode_omni_audio_chunk(piece.audio_b64)
                    except Exception:
                        pcm24 = b""
                        chunk_sr = audio_state["current_sr"]
                    if not pcm24:
                        continue
                    if chunk_sr != audio_state["current_sr"]:
                        audio_state["current_sr"] = chunk_sr
                        resamplers["to_8k"] = PCMResampler(chunk_sr, 8000)
                        resamplers["to_16k"] = PCMResampler(chunk_sr, 16000)
                        print(f"[AI AUDIO] source sample rate={audio_state['current_sr']}")
                    audio_state["chunks_received"] += 1
                    if audio_state["chunks_received"] == 1:
                        print(f"[AI AUDIO] first chunk bytes={len(pcm24)} sr={audio_state['current_sr']}")
                    pcm16k = resamplers["to_16k"].process(pcm24)
                    if pcm16k:
                        recorder.add_ai(pcm16k)
                    if AUDIO_OUTPUT_MODE == "local":
                        play_pcm16_audio_local(pcm24, chunk_sr)
                        continue
                    pcm8k = resamplers["to_8k"].process(pcm24)
                    pcm8k = adjust_volume(pcm8k, 0.60)
                    if pcm8k:
                        play_pcm8k_audio(pcm8k)

        try:
            await _do_stream()
        except asyncio.CancelledError:
            raise
        except Exception:
            try:
                await ui_broadcast_partial("[AI] 连接中断，正在重试...")
            except Exception:
                pass
            txt_buf.clear()
            resamplers["to_8k"].reset()
            resamplers["to_16k"].reset()
            await asyncio.sleep(1)
            try:
                await _do_stream()
            except asyncio.CancelledError:
                raise
            except Exception as e2:
                print("[AI ERROR] retry failed")
                traceback.print_exc()
                try:
                    await ui_broadcast_final(f"[AI] 发生错误：{e2}")
                except Exception:
                    pass
        finally:
            try:
                from audio.audio_stream import stream_clients

                for sc in list(stream_clients):
                    if not sc.abort_event.is_set():
                        try:
                            sc.q.put_nowait(b"\x00" * BYTES_PER_20MS_16K)
                        except Exception:
                            pass
                        try:
                            sc.q.put_nowait(None)
                        except Exception:
                            pass
            except Exception:
                pass

            final_text = ("".join(txt_buf)).strip() or "（空响应）"
            if audio_state["chunks_received"] == 0:
                print("[AI AUDIO] no audio chunks received")
            try:
                await ui_broadcast_final("[AI] " + final_text)
            except Exception:
                pass

            recorder.finish_turn()
            if manage_command_stream:
                await _set_command_streaming(False, reason="chat_done")

    stop_audio_playback()
    await cancel_current_ai()
    loop = asyncio.get_running_loop()
    from audio.audio_stream import __dict__ as _as_dict

    task = loop.create_task(_runner())
    _as_dict["current_ai_task"] = task


@app.get("/", response_class=HTMLResponse)
def root():
    with open(os.path.join("resources", "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(
            f.read(),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )


@app.get("/api/health", response_class=PlainTextResponse)
def health():
    return "OK"


register_stream_route(app)


@app.websocket("/ws_ui")
async def ws_ui(ws: WebSocket):
    await ws.accept()
    ui_clients[id(ws)] = ws
    try:
        init = {
            "partial": current_partial,
            "finals": recent_finals[-10:],
            "captures": recent_captures[-6:],
            "device_state": _device_state,
        }
        await ws.send_text("INIT:" + json.dumps(init, ensure_ascii=False))
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=60)
                if msg.startswith("PROMPT:"):
                    text = msg[7:].strip()
                    if text:
                        asyncio.get_event_loop().create_task(start_ai_with_text_custom(text))
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        ui_clients.pop(id(ws), None)


@app.websocket("/ws_audio")
async def ws_audio(ws: WebSocket):
    global esp32_audio_ws
    esp32_audio_ws = ws
    await ws.accept()
    _device_state["audio_connected"] = True
    asyncio.get_event_loop().create_task(_broadcast_device_state())
    print("\n[AUDIO] client connected")

    recognition = None
    streaming = False
    last_ts = time.monotonic()
    keepalive_task: Optional[asyncio.Task] = None

    async def stop_rec(send_notice: Optional[str] = None):
        nonlocal recognition, streaming, keepalive_task
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        keepalive_task = None

        if recognition:
            try:
                recognition.stop()
            except Exception:
                pass
            recognition = None

        await set_current_recognition(None)
        streaming = False
        if send_notice:
            try:
                await ws.send_text(send_notice)
            except Exception:
                pass

    async def start_recognition():
        nonlocal recognition, streaming, keepalive_task, last_ts
        if streaming and recognition is not None:
            return

        await stop_rec()
        loop = asyncio.get_running_loop()

        def post(coro):
            asyncio.run_coroutine_threadsafe(coro, loop)

        async def session_full_reset(reason: str = ""):
            await stop_rec()
            await full_system_reset(
                reason,
                notify_audio_device=False,
                stop_asr=False,
            )
            await start_recognition()

        cb = ASRCallback(
            on_sdk_error=lambda s: post(on_sdk_error(s)),
            post=post,
            ui_broadcast_partial=ui_broadcast_partial,
            ui_broadcast_final=ui_broadcast_final,
            is_playing_now_fn=is_playing_now,
            start_ai_with_text_fn=start_ai_with_text_custom,
            full_system_reset_fn=session_full_reset,
            interrupt_lock=interrupt_lock,
        )

        recognition = dash_audio.asr.Recognition(
            api_key=API_KEY,
            model=MODEL,
            format=AUDIO_FMT,
            sample_rate=SAMPLE_RATE,
            callback=cb,
        )
        recognition.start()
        await set_current_recognition(recognition)
        streaming = True
        last_ts = time.monotonic()
        keepalive_task = asyncio.create_task(keepalive_loop())
        await ui_broadcast_partial("（已开始接收音频…）")
        await ws.send_text("OK:STARTED")

    async def on_sdk_error(_msg: str):
        await stop_rec(send_notice="RESTART")

    async def keepalive_loop():
        nonlocal last_ts, recognition, streaming
        try:
            while streaming and recognition is not None:
                idle = time.monotonic() - last_ts
                if idle > 0.35:
                    try:
                        for _ in range(30):
                            recognition.send_audio_frame(SILENCE_20MS)
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("keepalive send failed")
                        return
                await asyncio.sleep(0.10)
        except asyncio.CancelledError:
            return

    try:
        while True:
            if WebSocketState and ws.client_state != WebSocketState.CONNECTED:
                break

            try:
                msg = await ws.receive()
            except WebSocketDisconnect:
                break
            except RuntimeError as e:
                if 'Cannot call "receive"' in str(e):
                    break
                raise

            if "text" in msg and msg["text"] is not None:
                raw = (msg["text"] or "").strip()
                cmd = raw.upper()

                if cmd == "START":
                    if streaming and recognition is not None:
                        await ws.send_text("OK:STARTED")
                        continue
                    print("[AUDIO] START received")
                    await start_recognition()

                elif cmd == "STOP":
                    if recognition:
                        for _ in range(15):
                            try:
                                recognition.send_audio_frame(SILENCE_20MS)
                            except Exception:
                                break
                    await stop_rec(send_notice="OK:STOPPED")

                elif raw.startswith("PROMPT:"):
                    text = raw[len("PROMPT:") :].strip()
                    if text:
                        await start_ai_with_text_custom(text)
                        await ws.send_text("OK:PROMPT_ACCEPTED")
                    else:
                        await ws.send_text("ERR:EMPTY_PROMPT")

            elif "bytes" in msg and msg["bytes"] is not None:
                if streaming and recognition:
                    payload = msg["bytes"]
                    try:
                        recognition.send_audio_frame(payload)
                        recorder.add_mic(payload)
                        last_ts = time.monotonic()
                    except Exception:
                        await on_sdk_error("send_audio_frame failed")

    except Exception as e:
        print(f"\n[WS ERROR] {e}")
    finally:
        await stop_rec()
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        if esp32_audio_ws is ws:
            esp32_audio_ws = None
            _device_state["audio_connected"] = False
            asyncio.get_event_loop().create_task(_broadcast_device_state())
        print("[WS] connection closed")


@app.websocket("/ws/camera")
async def ws_camera_esp(ws: WebSocket):
    global esp32_camera_ws
    if esp32_camera_ws is not None:
        await ws.close(code=1013)
        return
    esp32_camera_ws = ws
    await ws.accept()
    _device_state["camera_connected"] = True
    asyncio.get_event_loop().create_task(_broadcast_device_state())
    print("[CAMERA] ESP32 connected")
    if esp32_camera_ws.client_state == WebSocketState.CONNECTED:
        try:
            await esp32_camera_ws.send_text("CAMERA_STREAM:STOP")
        except Exception:
            pass

    frame_idx = 0
    last_preview_ts = 0.0
    last_cmd_preview_ts = 0.0

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                frame_idx += 1
                data = msg["bytes"]
                camera_frame_event.set()

                try:
                    last_frames.append((time.time(), data))
                except Exception:
                    pass

                bridge_io.push_raw_jpeg(data)

                if command_streaming_enabled and command_camera_clients and (time.time() - last_cmd_preview_ts) >= 0.12:
                    last_cmd_preview_ts = time.time()
                    cmd_preview_jpeg = data
                    try:
                        arr = np.frombuffer(data, dtype=np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            bgr = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_AREA)
                            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                            if ok:
                                cmd_preview_jpeg = enc.tobytes()
                    except Exception:
                        pass

                    dead_cmd = []
                    for cmd_ws in list(command_camera_clients):
                        try:
                            await cmd_ws.send_bytes(cmd_preview_jpeg)
                        except Exception:
                            dead_cmd.append(cmd_ws)
                    for d in dead_cmd:
                        command_camera_clients.discard(d)

                if camera_viewers and (time.time() - last_preview_ts) >= 0.25:
                    last_preview_ts = time.time()
                    preview_jpeg = data
                    try:
                        arr = np.frombuffer(data, dtype=np.uint8)
                        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if bgr is not None:
                            bgr = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_AREA)
                            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
                            if ok:
                                preview_jpeg = enc.tobytes()
                    except Exception:
                        pass

                    dead = []
                    for viewer_ws in list(camera_viewers):
                        try:
                            await viewer_ws.send_bytes(preview_jpeg)
                        except Exception:
                            dead.append(viewer_ws)
                    for d in dead:
                        camera_viewers.discard(d)

            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[CAMERA ERROR] {e}")
    finally:
        try:
            if WebSocketState is None or ws.client_state == WebSocketState.CONNECTED:
                await ws.close(code=1000)
        except Exception:
            pass
        esp32_camera_ws = None
        _device_state["camera_connected"] = False
        asyncio.get_event_loop().create_task(_broadcast_device_state())
        print("[CAMERA] ESP32 disconnected")


@app.websocket("/ws/camera/command")
async def ws_camera_command(ws: WebSocket):
    await ws.accept()
    command_camera_clients.add(ws)
    await _broadcast_command_stream_state("client_connected")
    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"] is not None:
                cmd = (msg["text"] or "").strip().upper()
                if cmd == "START":
                    await _set_command_streaming(True, reason="manual_start")
                    await ws.send_text("OK:STARTED")
                elif cmd == "STOP":
                    await _set_command_streaming(False, reason="manual_stop")
                    await ws.send_text("OK:STOPPED")
                elif cmd == "CAPTURE":
                    jpeg = await _acquire_command_jpeg()
                    if jpeg:
                        await ws.send_bytes(jpeg)
                    else:
                        await ws.send_text("ERR:NO_FRAME")
                elif cmd == "LOW":
                    await _request_camera_profile(CAM_LOW_PROFILE)
                    await ws.send_text("OK:LOW")
                elif cmd == "HIGH":
                    await _request_camera_profile(CAM_HIGH_PROFILE)
                    await ws.send_text("OK:HIGH")
                else:
                    await ws.send_text("ERR:UNKNOWN_COMMAND")
            elif "type" in msg and msg["type"] in ("websocket.close", "websocket.disconnect"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        command_camera_clients.discard(ws)


@app.websocket("/ws/viewer")
async def ws_viewer(ws: WebSocket):
    await ws.accept()
    camera_viewers.add(ws)
    print(f"[VIEWER] Browser connected. Total viewers: {len(camera_viewers)}", flush=True)
    try:
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        print("[VIEWER] Browser disconnected", flush=True)
    finally:
        try:
            camera_viewers.remove(ws)
        except Exception:
            pass
        print(f"[VIEWER] Removed. Total viewers: {len(camera_viewers)}", flush=True)




def get_last_frames():
    return last_frames


def get_camera_ws():
    return esp32_camera_ws


if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8765,
            log_level="warning",
            access_log=False,
            loop="asyncio",
            workers=1,
            reload=False,
        )
    except asyncio.CancelledError:
        pass

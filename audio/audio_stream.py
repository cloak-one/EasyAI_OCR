# audio_stream.py
# -*- coding: utf-8 -*-
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Optional, Set, List, Tuple, Any, Dict, Callable
from fastapi import Request
from fastapi.responses import StreamingResponse

# ===== 录制器动态导入（优先 image，兼容旧 video） =====
_recorder_module = None
_recorder_imported = False

def _get_recorder_module():
    global _recorder_module, _recorder_imported
    if _recorder_imported:
        return _recorder_module

    _recorder_imported = True
    for mod in ("image.image_recorder", "image_recorder", "sync_recorder"):
        try:
            _recorder_module = __import__(mod, fromlist=["*"])
            print(f"[AUDIO-STREAM] 使用录制器模块: {mod}")
            return _recorder_module
        except Exception:
            continue

    _recorder_module = None
    print("[AUDIO-STREAM] 未找到录制器模块，跳过录制")
    return None

# ===== 下行 WAV 流基础参数 =====
STREAM_SR = 8000  # 改为8kHz，ESP32支持
STREAM_CH = 1
STREAM_SW = 2
BYTES_PER_20MS_16K = STREAM_SR * STREAM_SW * 20 // 1000  # 320B (8kHz)

# ===== AI 播放任务总闸 =====
current_ai_task: Optional[asyncio.Task] = None

async def cancel_current_ai():
    """取消当前大模型语音任务，并等待其退出。"""
    global current_ai_task
    task = current_ai_task
    current_ai_task = None
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

def is_playing_now() -> bool:
    t = current_ai_task
    return (t is not None) and (not t.done())

# ===== /stream.wav 连接管理 =====
@dataclass(eq=False)
class StreamClient:
    q: asyncio.Queue
    abort_event: asyncio.Event
    done_future: "Optional[asyncio.Future[None]]" = None

stream_clients: "Set[StreamClient]" = set()
STREAM_QUEUE_MAX = 96  # 小缓冲，避免积压
STREAM_BACKLOG_MAX_BYTES = STREAM_SR * STREAM_SW * 3  # 保留最近 3 秒音频，处理客户端晚连
_stream_backlog: "deque[bytes]" = deque()
_stream_backlog_bytes = 0


def _append_stream_backlog(chunk: bytes) -> None:
    global _stream_backlog_bytes
    if not chunk:
        return
    _stream_backlog.append(chunk)
    _stream_backlog_bytes += len(chunk)
    while _stream_backlog and _stream_backlog_bytes > STREAM_BACKLOG_MAX_BYTES:
        dropped = _stream_backlog.popleft()
        _stream_backlog_bytes -= len(dropped)


def _take_stream_backlog() -> List[bytes]:
    global _stream_backlog_bytes
    if not _stream_backlog:
        return []
    chunks = list(_stream_backlog)
    _stream_backlog.clear()
    _stream_backlog_bytes = 0
    return chunks

def _wav_header_unknown_size(sr=16000, ch=1, sw=2) -> bytes:
    import struct
    byte_rate = sr * ch * sw
    block_align = ch * sw
    data_size = 0x7FFFFFF0
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,
        1, ch, sr, byte_rate, block_align, sw * 8,
        b"data", data_size
    )

async def hard_reset_audio(reason: str = ""):
    """
    **一键清场**：丢弃所有播放器连接（abort_event置位）+ 取消当前AI任务。
    等待所有流任务退出后再返回，确保 shutdown 不卡住。
    """
    # 1) 断开所有正在播放的 HTTP 连接
    for sc in list(stream_clients):
        try:
            sc.abort_event.set()
        except Exception:
            pass

    # 2) 收集所有生成器任务
    tasks = [sc.done_future for sc in list(stream_clients) if sc.done_future is not None]
    stream_clients.clear()
    _take_stream_backlog()

    # 3) 取消当前AI任务
    await cancel_current_ai()

    # 4) 等待所有流任务自然退出（abort_event 已置位，循环会 break）
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # 5) 日志
    if reason:
        print(f"[HARD-RESET] {reason}")

async def broadcast_pcm16_realtime(
    pcm16: bytes,
    should_abort: "Optional[Callable[[], bool]]" = None,
):
    """以 20ms 节拍把 pcm16 发送给所有仍存活的连接；队列满丢尾，保持实时。"""
    # 【新增】录制音频（在分发之前整体录制，避免分片）
    try:
        # import sync_recorder
        # sync_recorder.record_audio(pcm16, text="[Omni对话]")
        recorder = _get_recorder_module()
        if recorder and hasattr(recorder, "record_audio"):
            recorder.record_audio(pcm16, text="[Omni对话]")
    except Exception:
        pass  # 静默失败，不影响播放
    
    loop = asyncio.get_event_loop()
    next_tick = loop.time()
    off = 0
    while off < len(pcm16):
        if should_abort and should_abort():
            break
        take = min(BYTES_PER_20MS_16K, len(pcm16) - off)
        piece = pcm16[off:off + take]

        dead: List[StreamClient] = []
        active_clients = [sc for sc in list(stream_clients) if not sc.abort_event.is_set()]
        if not active_clients:
            _append_stream_backlog(piece)

        for sc in active_clients:
            if sc.abort_event.is_set():
                dead.append(sc)
                continue
            try:
                if sc.q.full():
                    try: sc.q.get_nowait()
                    except Exception: pass
                sc.q.put_nowait(piece)
            except Exception:
                dead.append(sc)
        for sc in dead:
            try: stream_clients.discard(sc)
            except Exception: pass

        next_tick += 0.020
        now = loop.time()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
        else:
            next_tick = now
        off += take

# ===== FastAPI 路由注册器 =====
def register_stream_route(app):
    @app.get("/stream.wav")
    async def stream_wav(_: Request):
        # —— 强制单连接（或少数连接），先拉闸所有旧连接 ——
        for sc in list(stream_clients):
            try: sc.abort_event.set()
            except Exception: pass
        stream_clients.clear()

        q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
        abort_event = asyncio.Event()
        done_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        sc = StreamClient(q=q, abort_event=abort_event, done_future=done_future)
        stream_clients.add(sc)
        backlog = _take_stream_backlog()

        async def gen():
            yield _wav_header_unknown_size(STREAM_SR, STREAM_CH, STREAM_SW)
            for chunk in backlog:
                if abort_event.is_set():
                    break
                if chunk:
                    yield chunk
            try:
                while True:
                    if abort_event.is_set():
                        break
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # 保持 HTTP 音频流不断开，避免 ESP32 端在空闲期频繁重连
                        yield b"\x00" * BYTES_PER_20MS_16K
                        continue
                    if abort_event.is_set():
                        break
                    if chunk is None:
                        break
                    if chunk:
                        yield chunk
            finally:
                stream_clients.discard(sc)
                if not done_future.done():
                    done_future.set_result(None)

        return StreamingResponse(gen(), media_type="audio/wav")

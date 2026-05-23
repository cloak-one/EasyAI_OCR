# audio_player.py
# 仅负责将语音内容转成PCM并通过ESP32音频流下发，不做本地扬声器播放。

import os
import audioop
import asyncio
import time
import tempfile
import threading
import queue
from typing import Dict

try:
    from audio.audio_stream import broadcast_pcm16_realtime
except Exception:
    from audio_stream import broadcast_pcm16_realtime

try:
    from audio.tts_core import TTSOptions, load_wav_to_pcm16k_mono, synthesize_tts_wav
except Exception:
    from tts_core import TTSOptions, load_wav_to_pcm16k_mono, synthesize_tts_wav

PCM8K_CACHE: Dict[str, bytes] = {}

_initialized = False
_init_lock = threading.Lock()

_audio_queue = queue.PriorityQueue(maxsize=10)
_audio_priority = 0
_worker_thread = None
_worker_loop = None
_is_playing = False
_playing_lock = threading.Lock()
_playback_generation = 0
_shutdown_requested = False

_last_voice_time = 0.0
_last_voice_text = ""
_voice_cooldown = 0.8


def _ensure_even_bytes(data: bytes) -> bytes:
    if len(data) % 2 == 1:
        return data[:-1]
    return data


def initialize_audio_system():
    global _initialized, _worker_thread, _shutdown_requested
    with _init_lock:
        if _initialized:
            return
        _shutdown_requested = False
        _worker_thread = threading.Thread(target=_audio_worker, daemon=True)
        _worker_thread.start()
        PCM8K_CACHE.clear()
        _initialized = True
        print("[AUDIO] 初始化完成（TTS远端播报模式）")


def _is_generation_stale(generation: int) -> bool:
    with _playing_lock:
        return generation != _playback_generation


async def _stream_pcm8k(pcm8k: bytes, generation: int):
    if not pcm8k:
        return
    lead = b"\x00" * (8000 * 2 * 60 // 1000)
    tail = b"\x00" * (8000 * 2 * 40 // 1000)
    await broadcast_pcm16_realtime(lead + pcm8k + tail, should_abort=lambda: _is_generation_stale(generation))


async def _broadcast_audio_optimized(pcm_data: bytes, generation: int):
    global _is_playing
    try:
        with _playing_lock:
            _is_playing = True
        await _stream_pcm8k(pcm_data or b"", generation)
    finally:
        with _playing_lock:
            _is_playing = False


def _clear_audio_queue():
    while True:
        try:
            _audio_queue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break


def stop_audio_playback():
    global _playback_generation
    with _playing_lock:
        _playback_generation += 1
    _clear_audio_queue()


def shutdown_audio_system():
    global _initialized, _worker_thread, _worker_loop, _shutdown_requested
    with _init_lock:
        if not _initialized:
            return
        _shutdown_requested = True
        stop_audio_playback()
        try:
            _audio_queue.put_nowait(None)
        except queue.Full:
            _clear_audio_queue()
            try:
                _audio_queue.put_nowait(None)
            except Exception:
                pass

        worker = _worker_thread
        _worker_thread = None
        _worker_loop = None
        _initialized = False

    if worker and worker.is_alive():
        worker.join(timeout=1.0)


def _audio_worker():
    global _worker_loop
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)
    try:
        while True:
            try:
                priority_data = _audio_queue.get(timeout=0.2)
            except queue.Empty:
                if _shutdown_requested:
                    break
                continue
            except Exception:
                if _shutdown_requested:
                    break
                continue

            if priority_data is None:
                break

            if isinstance(priority_data, tuple) and len(priority_data) == 3:
                _, generation, pcm_data = priority_data
            elif isinstance(priority_data, tuple) and len(priority_data) == 2:
                _, pcm_data = priority_data
                with _playing_lock:
                    generation = _playback_generation
            else:
                pcm_data = priority_data
                with _playing_lock:
                    generation = _playback_generation

            if not pcm_data:
                continue

            if _is_generation_stale(generation):
                continue

            try:
                _worker_loop.run_until_complete(_broadcast_audio_optimized(pcm_data, generation))
            except Exception as e:
                print(f"[AUDIO] 队列播放失败: {e}")
    finally:
        try:
            _worker_loop.stop()
        except Exception:
            pass
        try:
            _worker_loop.close()
        except Exception:
            pass


def _submit_stream_task(pcm8k: bytes):
    global _audio_priority
    if not pcm8k:
        return

    queue_size = _audio_queue.qsize()
    with _playing_lock:
        currently_playing = _is_playing
        generation = _playback_generation

    if queue_size > 0 and not currently_playing:
        print(f"[AUDIO] 清空队列（当前{queue_size}个），播放最新语音")
        _clear_audio_queue()
    elif queue_size > 1 and currently_playing:
        print(f"[AUDIO] 队列积压({queue_size}个)，清空以保持实时")
        _clear_audio_queue()

    try:
        _audio_priority += 1
        _audio_queue.put_nowait((_audio_priority, generation, pcm8k))
        if queue_size >= 1:
            print(f"[AUDIO] 播放队列当前大小: {queue_size + 1}")
    except queue.Full:
        print("[AUDIO] 队列满，丢弃本次语音")


def _synthesize_text_to_pcm8k(text: str) -> bytes:
    text = (text or "").strip()
    if not text:
        return b""

    with tempfile.NamedTemporaryFile(prefix="tts_", suffix=".wav", delete=False) as tf:
        wav_path = tf.name

    try:
        ok = synthesize_tts_wav(
            text=text,
            wav_path=wav_path,
            opts=TTSOptions(rate=185, volume=1.0, voice_name=""),
        )
        if not ok:
            return b""

        pcm16k = load_wav_to_pcm16k_mono(wav_path)
        if not pcm16k:
            return b""

        pcm8k, _ = audioop.ratecv(pcm16k, 2, 1, 16000, 8000, None)
        return _ensure_even_bytes(pcm8k)
    except Exception as e:
        print(f"[AUDIO] TTS合成失败: {e}")
        return b""
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass


def play_voice_text(text: str):
    global _last_voice_time, _last_voice_text

    if not text:
        return
    if not _initialized:
        initialize_audio_system()

    now = time.monotonic()
    if text == _last_voice_text and (now - _last_voice_time) < _voice_cooldown:
        return

    pcm8k = _synthesize_text_to_pcm8k(text)
    if pcm8k:
        _submit_stream_task(pcm8k)
        _last_voice_text = text
        _last_voice_time = now
        return

    print(f"[AUDIO] TTS失败: {text}")


play_audio_on_esp32 = play_voice_text

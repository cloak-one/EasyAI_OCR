#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import asyncio
import audioop
import base64
import os
import queue
import re
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import sounddevice as sd
from dashscope import audio as dash_audio
import audio.asr_core as asr_core_mod
from audio.tts_core import TTSOptions, load_wav_to_pcm16k_mono, synthesize_tts_wav
from ui.publisher import UIPublisher
from ui.server import UIServer
from ui.state_store import UIStateStore

from audio.omni_client import stream_chat
from image.page_finder import BookPageFinder, parse_page_token


from config import DASHSCOPE_API_KEY as API_KEY

ASR_MODEL = "paraformer-realtime-v2"
ASR_SAMPLE_RATE = 16000
ASR_FORMAT = "pcm"
PCM16_16K_CHUNK_BYTES = 640
SILENCE_20MS = bytes(PCM16_16K_CHUNK_BYTES)

IDLE = "IDLE"
CHAT = "CHAT"
FIND_PAGE = "FIND_PAGE"


def _normalize_text(s: str) -> str:
	return re.sub(r"[\s\u3000\.,，。！？!?、；;:：\"'“”‘’`~…-]+", "", s or "")


def _is_interrupt_text(s: str) -> bool:
	norm = _normalize_text(s)
	if not norm:
		return False
	keywords = ["停下", "别说了", "停止", "停一下", "算了", "不说了", "打住"]
	for kw in keywords:
		if _normalize_text(kw) in norm:
			return True
	return False


def _ensure_even_bytes(data: bytes) -> bytes:
	if len(data) % 2 == 1:
		return data[:-1]
	return data


class SessionAudioRecorder:
	"""低延迟会话录制器：运行中异步写盘，退出时统一关闭。"""

	def __init__(self, output_dir: str):
		self.output_dir = output_dir
		self.mic_path = os.path.join(output_dir, "mic_input.wav")
		self.ai_path = os.path.join(output_dir, "ai_output.wav")
		self._queue: "queue.Queue[tuple[str, bytes]]" = queue.Queue(maxsize=4096)
		self._stop = threading.Event()
		self._worker: Optional[threading.Thread] = None
		self._mic_wf: Optional[wave.Wave_write] = None
		self._ai_wf: Optional[wave.Wave_write] = None
		self._mic_bytes = 0
		self._ai_bytes = 0
		self._dropped = 0

	def start(self):
		os.makedirs(self.output_dir, exist_ok=True)
		self._mic_wf = wave.open(self.mic_path, "wb")
		self._mic_wf.setnchannels(1)
		self._mic_wf.setsampwidth(2)
		self._mic_wf.setframerate(16000)
		self._ai_wf = wave.open(self.ai_path, "wb")
		self._ai_wf.setnchannels(1)
		self._ai_wf.setsampwidth(2)
		self._ai_wf.setframerate(16000)
		self._stop.clear()
		self._worker = threading.Thread(target=self._run, daemon=True)
		self._worker.start()
		print(f"[RECORDER] 会话录制已启动: {self.output_dir}")

	def _run(self):
		while not self._stop.is_set() or not self._queue.empty():
			try:
				stream_name, payload = self._queue.get(timeout=0.2)
			except queue.Empty:
				continue
			except Exception:
				continue

			if not payload:
				continue

			try:
				if stream_name == "mic" and self._mic_wf is not None:
					self._mic_wf.writeframes(payload)
					self._mic_bytes += len(payload)
				elif stream_name == "ai" and self._ai_wf is not None:
					self._ai_wf.writeframes(payload)
					self._ai_bytes += len(payload)
			except Exception as e:
				print(f"[RECORDER] 写入失败({stream_name}): {e}")

	def add_mic(self, pcm16: bytes):
		self._put("mic", pcm16)

	def add_ai(self, pcm16: bytes):
		self._put("ai", pcm16)

	def _put(self, stream_name: str, pcm16: bytes):
		if not pcm16:
			return
		try:
			self._queue.put_nowait((stream_name, pcm16))
		except queue.Full:
			self._dropped += 1
		except Exception:
			pass

	def stop(self):
		self._stop.set()
		if self._worker is not None:
			self._worker.join(timeout=3.0)
		self._worker = None

		try:
			if self._mic_wf is not None:
				self._mic_wf.close()
		except Exception:
			pass
		finally:
			self._mic_wf = None

		try:
			if self._ai_wf is not None:
				self._ai_wf.close()
		except Exception:
			pass
		finally:
			self._ai_wf = None

		mic_sec = self._mic_bytes / (16000 * 2) if self._mic_bytes else 0.0
		ai_sec = self._ai_bytes / (16000 * 2) if self._ai_bytes else 0.0
		print(
			f"[RECORDER] 会话录制已保存: mic={self.mic_path}({mic_sec:.2f}s), "
			f"ai={self.ai_path}({ai_sec:.2f}s), dropped={self._dropped}"
		)


@dataclass
class LocalState:
	state: str = IDLE
	partial: str = ""
	finals: List[str] = field(default_factory=list)
	latest_jpeg: Optional[bytes] = None
	latest_bgr: Optional[Any] = None
	ai_playing: bool = False
	lock: threading.Lock = field(default_factory=threading.Lock)


class LocalReaderController:
	def __init__(self, args):
		self.args = args
		self.state = LocalState()
		self.page_finder = BookPageFinder()
		self.stop_event = asyncio.Event()
		self.interrupt_lock = asyncio.Lock()

		self.mic_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=512)
		self.play_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=512)
		self.last_final_text: str = ""
		self.last_final_norm: str = ""
		self.last_final_ts: float = 0.0
		self.run_session_dir: Optional[str] = None
		self.turn_seq: int = 0
		self.turn_lock = threading.Lock()
		self.current_ai_task: Optional[asyncio.Task] = None
		self.play_epoch: int = 0

		self.camera_info: Dict[str, Any] = {}
		self.camera_lock = threading.Lock()
		self.camera_cap: Optional[cv2.VideoCapture] = None
		self.camera_stop = threading.Event()
		self.camera_thread: Optional[threading.Thread] = None
		self.command_capture_seq = 0
		self.ui_state_store: Optional[UIStateStore] = None
		self.ui_publisher: Optional[UIPublisher] = None
		self.ui_server: Optional[UIServer] = None
		self.session_recorder: Optional[SessionAudioRecorder] = None

	def _ui_publish_state(self, state: str):
		if self.ui_publisher is not None:
			self.ui_publisher.publish_state(state)

	def _ui_publish_error(self, message: str):
		if self.ui_publisher is not None:
			self.ui_publisher.publish_error(message)

	def _ui_publish_user_text(self, text: str):
		if self.ui_publisher is not None:
			self.ui_publisher.publish_user_text(text)

	def _ui_publish_partial_text(self, text: str):
		if self.ui_publisher is not None:
			self.ui_publisher.publish_partial_text(text)

	def _ui_publish_model_delta(self, text: str):
		if self.ui_publisher is not None:
			self.ui_publisher.publish_model_delta(text)

	def _ui_publish_model_final(self, text: str):
		if self.ui_publisher is not None:
			self.ui_publisher.publish_model_final(text)

	def _ui_publish_frame(self, jpeg: bytes):
		if self.ui_publisher is not None:
			self.ui_publisher.publish_frame(jpeg)

	def _start_run_session(self):
		ts = time.strftime("%Y%m%d_%H%M%S")
		save_root = self.args.response_save_dir
		base_name = f"session_{ts}"
		candidate = os.path.join(save_root, base_name)
		idx = 1
		while os.path.exists(candidate):
			candidate = os.path.join(save_root, f"{base_name}_{idx:02d}")
			idx += 1
		self.run_session_dir = candidate
		os.makedirs(self.run_session_dir, exist_ok=False)
		print(f"[SAVE] 本次运行目录: {self.run_session_dir}")
		try:
			self.session_recorder = SessionAudioRecorder(self.run_session_dir)
			self.session_recorder.start()
		except Exception as e:
			self.session_recorder = None
			print(f"[RECORDER] 会话录制器启动失败: {e}")

	def _create_turn_dir(self) -> str:
		with self.turn_lock:
			self.turn_seq += 1
			seq = self.turn_seq
		ts = time.strftime("%Y%m%d_%H%M%S")
		base_dir = self.run_session_dir or self.args.response_save_dir
		turn_dir = os.path.join(base_dir, f"turn_{seq:04d}_{ts}")
		os.makedirs(turn_dir, exist_ok=True)
		return turn_dir

	def _record_asr_final_meta(self, text: str):
		from image.image_recorder import record_asr_text
		record_asr_text(text)

	def _clear_play_queue(self):
		while True:
			try:
				self.play_queue.get_nowait()
			except queue.Empty:
				break
			except Exception:
				break

	def _bump_play_epoch(self) -> int:
		with self.state.lock:
			self.play_epoch += 1
			return self.play_epoch

	def _current_play_epoch(self) -> int:
		with self.state.lock:
			return self.play_epoch

	def _is_playing_now(self) -> bool:
		with self.state.lock:
			return bool(self.state.ai_playing)

	def _enqueue_play_chunk(self, reply_epoch: int, pcm16k: bytes):
		if not pcm16k:
			return
		try:
			self.play_queue.put_nowait((reply_epoch, pcm16k))
		except queue.Full:
			try:
				self.play_queue.get_nowait()
				self.play_queue.put_nowait((reply_epoch, pcm16k))
			except Exception:
				pass

	async def ui_partial_from_asr(self, text: str):
		with self.state.lock:
			self.state.partial = text
		self._ui_publish_partial_text(text)
		print(f"[ASR PARTIAL] {text}")
		# print(f"[识别-实时] {text}")

	async def ui_final_from_asr(self, text: str):
		self._append_final(text)
		self._record_asr_final_meta(text)
		with self.state.lock:
			self.state.partial = ""
		self._ui_publish_partial_text("")
		self._ui_publish_user_text(text)
		print(f"[ASR FINAL] {text}")
		# print(f"[识别-最终] {text}")

	async def full_system_reset(self, reason: str = ""):
		# 热词打断时，立即取消当前AI任务并清空待播音频队列。
		self._bump_play_epoch()
		if self.current_ai_task is not None and not self.current_ai_task.done():
			self.current_ai_task.cancel()
			try:
				await self.current_ai_task
			except asyncio.CancelledError:
				pass
			except Exception:
				pass

		self.current_ai_task = None

		self._clear_play_queue()

		with self.state.lock:
			self.state.state = IDLE
			self.state.partial = ""
			self.state.ai_playing = False
		if reason:
			self._append_final(f"[系统] 已停止：{reason}")
			self._ui_publish_error(reason)
		self._ui_publish_state(IDLE)
		print(f"[HOTWORD RESET] {reason}")

	async def start_ai_from_asr(self, text: str):
		# 兜底保护：即使ASR层热词短路失败，也不要把“停下”送到LLM。
		if _is_interrupt_text(text):
			await self.full_system_reset("Hotword interrupt (main fallback)")
			print(f"[CHAIN] interrupt text blocked from LLM: {text}")
			return

		if self._is_playing_now():
			print(f"[CHAIN] AI播放中，直接丢弃识别文本: {text}")
			return

		norm = _normalize_text(text)
		if not norm:
			print("[ASR FINAL] empty-after-normalize ignored")
			return
		if len(norm) < int(self.args.min_query_chars):
			print(f"[ASR FINAL] too short ignored: {text}")
			return

		now = time.time()
		if norm == self.last_final_norm and (now - self.last_final_ts) < float(self.args.duplicate_final_window):
			print("[ASR FINAL] duplicate ignored")
			return
		self.last_final_text = text
		self.last_final_norm = norm
		self.last_final_ts = now
		self._ui_publish_user_text(text)

		if bool(getattr(self.args, "asr_only", False)):
			turn_dir = self._create_turn_dir()
			print(f"[SAVE] 当前问答目录: {turn_dir}")
			# ASR-only 也执行抓拍，便于验证阅读任务的图像输入链路。
			await self.capture_command_image(turn_dir=turn_dir)
			self._append_final(f"[ASR-ONLY] {text}")
			self._ui_publish_model_final(f"[ASR-ONLY] {text}")
			print("[CHAIN] ASR-only模式：跳过Qwen请求")
			return

		print("[CHAIN] ASR final accepted -> handle_user_text")
		if self.current_ai_task is not None and not self.current_ai_task.done():
			print("[CHAIN] 已有AI任务在执行，忽略本次触发")
			return

		self.current_ai_task = asyncio.create_task(self.handle_user_text(text))

		def _on_done(task: asyncio.Task):
			if self.current_ai_task is task:
				self.current_ai_task = None
			try:
				exc = task.exception()
			except asyncio.CancelledError:
				exc = None
			except Exception:
				exc = None
			if exc:
				print(f"[CHAIN] AI任务异常结束: {exc}")

		self.current_ai_task.add_done_callback(_on_done)


	def _append_final(self, text: str):
		with self.state.lock:
			self.state.finals.append(text)
			if len(self.state.finals) > 200:
				self.state.finals = self.state.finals[-100:]

	async def on_partial(self, text: str):
		with self.state.lock:
			self.state.partial = text
		print(f"[ASR PARTIAL] {text}")
		# print(f"[识别-实时] {text}")

	async def on_final(self, text: str):
		norm = _normalize_text(text)
		if not norm:
			print("[ASR FINAL] empty-after-normalize ignored")
			return
		if len(norm) < int(self.args.min_query_chars):
			print(f"[ASR FINAL] too short ignored: {text}")
			return

		now = time.time()
		if norm == self.last_final_norm and (now - self.last_final_ts) < float(self.args.duplicate_final_window):
			print("[ASR FINAL] duplicate ignored")
			return
		self.last_final_text = text
		self.last_final_norm = norm
		self.last_final_ts = now

		self._append_final(text)
		self._record_asr_final_meta(text)
		with self.state.lock:
			self.state.partial = ""
		self._ui_publish_partial_text("")
		self._ui_publish_user_text(text)
		print(f"[ASR FINAL] {text}")
		# print(f"[识别-最终] {text}")

		async with self.interrupt_lock:
			with self.state.lock:
				if self.state.ai_playing:
					print("[CHAIN] AI播放中，忽略本次final触发")
					return
			print("[CHAIN] ASR final accepted -> handle_user_text")
			await self.handle_user_text(text)

	async def hot_reset(self, text: str):
		async with self.interrupt_lock:
			await self.full_system_reset(text)
			print(f"[HOTWORD] {text}")

	async def handle_user_text(self, user_text: str):
		turn_dir = self._create_turn_dir()
		print(f"[SAVE] 当前问答目录: {turn_dir}")
		self._ui_publish_state("PROCESSING")

		# 阅读任务按“指令触发抓拍”：收到语音指令后再抓取高质量多帧图像，避免持续视频编码开销。
		await self.capture_command_image(turn_dir=turn_dir)

		page_match = re.search(r"第\s*([0-9一二三四五六七八九十百千万两\d]+)\s*页", user_text)
		if page_match and any(k in user_text for k in ("找", "找到", "定位", "翻到", "查到", "页")):
			print("[ROUTE] FIND_PAGE")
			with self.state.lock:
				self.state.state = FIND_PAGE

			target_page = parse_page_token(page_match.group(1))
			if target_page is None:
				await self.reply_text("请说出明确页码，例如：请帮我找到第42页。", turn_dir=turn_dir)
				return

			with self.state.lock:
				jpeg = self.state.latest_jpeg
			if not jpeg:
				await self.reply_text(self.page_finder.build_no_camera_guidance(target_page), turn_dir=turn_dir)
				return

			try:
				report = await self.page_finder.find_page(target_page=target_page, jpeg_bytes=jpeg)
				answer = str(report.get("message") or "").strip()
			except Exception as e:
				await self.reply_text(f"页码定位失败：{e}", turn_dir=turn_dir)
				return

			# find_page 内部统一生成引导语，避免在路由层重复拼接兜底文案。
			await self.reply_page_result_tts(answer, turn_dir=turn_dir)
			return

		with self.state.lock:
			self.state.state = CHAT
		self._ui_publish_state(CHAT)
		print("[ROUTE] CHAT")
		await self.reply_text(user_text, turn_dir=turn_dir)

	def _open_camera(self) -> cv2.VideoCapture:
		cap = cv2.VideoCapture(self.args.camera_index, cv2.CAP_DSHOW)
		if not cap.isOpened():
			cap = cv2.VideoCapture(self.args.camera_index)
		if not cap.isOpened():
			raise RuntimeError(f"无法打开摄像头 index={self.args.camera_index}")
		return cap

	def _apply_camera_mode(self, cap: cv2.VideoCapture, w: int, h: int, fps: float):
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
		cap.set(cv2.CAP_PROP_FPS, float(fps))
		time.sleep(0.05)
		rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
		rf = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
		self.camera_info = {
			"width": rw,
			"height": rh,
			"fps": rf,
		}
		print(f"[CAMERA] mode={rw}x{rh}@{rf:.1f}")

	def _camera_worker(self):
		try:
			cap = self._open_camera()
			with self.camera_lock:
				self.camera_cap = cap
				self._apply_camera_mode(cap, self.args.idle_camera_width, self.args.idle_camera_height, self.args.idle_camera_fps)

			idle_interval = 1.0 / max(1.0, float(self.args.idle_camera_fps))
			while not self.camera_stop.is_set():
				with self.camera_lock:
					ok, frame = cap.read()
				if ok and frame is not None:
					with self.state.lock:
						self.state.latest_bgr = frame
				time.sleep(idle_interval)
		except Exception as e:
			print(f"[CAMERA] worker error: {e}")
		finally:
			with self.camera_lock:
				cap = self.camera_cap
				self.camera_cap = None
			if cap is not None:
				try:
					cap.release()
				except Exception:
					pass

	def _capture_command_image_sync(self) -> Optional[bytes]:
		best_frame = None
		best_score = -1.0

		with self.camera_lock:
			cap = self.camera_cap
			if cap is None:
				return None

			self._apply_camera_mode(cap, self.args.command_camera_width, self.args.command_camera_height, self.args.command_camera_fps)

			for _ in range(max(1, int(self.args.command_capture_frames))):
				ok, frame = cap.read()
				if not ok or frame is None:
					continue
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
				if score > best_score:
					best_score = score
					best_frame = frame.copy()
				time.sleep(max(0.0, float(self.args.command_capture_gap_ms) / 1000.0))

			self._apply_camera_mode(cap, self.args.idle_camera_width, self.args.idle_camera_height, self.args.idle_camera_fps)

		if best_frame is None:
			return None

		ok_enc, enc = cv2.imencode(
			".jpg",
			best_frame,
			[int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.command_jpeg_quality)],
		)
		if not ok_enc:
			return None
		return enc.tobytes()

	async def capture_command_image(self, turn_dir: Optional[str] = None):
		loop = asyncio.get_running_loop()
		jpeg = await loop.run_in_executor(None, self._capture_command_image_sync)
		if jpeg:
			self.command_capture_seq += 1
			ts = time.strftime("%Y%m%d_%H%M%S")
			save_dir = turn_dir or self.run_session_dir or self.args.response_save_dir
			os.makedirs(save_dir, exist_ok=True)
			capture_path = os.path.join(
				save_dir,
				f"command_image_{ts}_{self.command_capture_seq:04d}.jpg",
			)
			try:
				with open(capture_path, "wb") as f:
					f.write(jpeg)
			except Exception as e:
				print(f"[CAMERA] 保存抓拍图失败: {e}")
				capture_path = ""

			with self.state.lock:
				self.state.latest_jpeg = jpeg
			self._ui_publish_frame(jpeg)
			if capture_path:
				print(f"[CAMERA] captured command frame: {capture_path} bytes={len(jpeg)}")
				if bool(self.args.open_command_image) and os.name == "nt":
					try:
						os.startfile(capture_path)
					except Exception as e:
						print(f"[CAMERA] 打开抓拍图失败: {e}")
			else:
				print(f"[CAMERA] captured command frame bytes={len(jpeg)}")
		else:
			print("[CAMERA] 未获取到指令抓拍图像")

	async def reply_page_result_tts(self, text: str, turn_dir: Optional[str] = None):
		with self.state.lock:
			self.state.ai_playing = True
			reply_epoch = self.play_epoch

		if not turn_dir:
			turn_dir = self._create_turn_dir()
			print(f"[SAVE] reply_page_result_tts未传入目录，已自动创建: {turn_dir}")

		self._append_final(f"[页码] {text}")
		self._ui_publish_model_final(f"[页码] {text}")
		print(f"[PAGE REPLY] {text}")

		loop = asyncio.get_running_loop()
		tts_wav_path = os.path.join(turn_dir, "page_result_tts.wav")
		try:
			tts_opts = TTSOptions(
				rate=int(getattr(self.args, "tts_rate", 180)),
				volume=float(getattr(self.args, "tts_volume", 1.0)),
				voice_name=str(getattr(self.args, "tts_voice_name", "") or ""),
			)
			ok = await loop.run_in_executor(None, synthesize_tts_wav, text, tts_wav_path, tts_opts)
			if not ok:
				return

			pcm16k = await loop.run_in_executor(None, load_wav_to_pcm16k_mono, tts_wav_path)
			if not pcm16k:
				print("[TTS] 合成音频为空，跳过播放")
				return

			self._enqueue_play_chunk(reply_epoch, pcm16k)

			play_seconds = len(pcm16k) / float(16000 * 2)
			await asyncio.sleep(max(0.05, play_seconds + 0.05))
		except asyncio.CancelledError:
			print("[TTS] 页码播报任务已取消")
			self._append_final("[页码] 已停止")
			self._ui_publish_error("页码播报任务已取消")
			raise
		except Exception as e:
			print(f"[TTS] 页码播报失败: {e}")
			self._append_final(f"[页码] 播报失败：{e}")
			self._ui_publish_error(f"页码播报失败：{e}")
		finally:
			with self.state.lock:
				self.state.partial = ""
				self.state.ai_playing = False
				if self.state.state == FIND_PAGE:
					self.state.state = IDLE
			self._ui_publish_state(IDLE)

	async def reply_text(self, text: str, turn_dir: Optional[str] = None):
		with self.state.lock:
			self.state.ai_playing = True
			reply_epoch = self.play_epoch

		if not turn_dir:
			turn_dir = self._create_turn_dir()
			print(f"[SAVE] reply_text未传入目录，已自动创建: {turn_dir}")

		content_list: List[Dict[str, Any]] = []
		with self.state.lock:
			jpeg = self.state.latest_jpeg
		if jpeg:
			img_b64 = base64.b64encode(jpeg).decode("ascii")
			content_list.append({
				"type": "image_url",
				"image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
			})
		content_list.append({"type": "text", "text": text})
		print(f"[LLM] request start, text_len={len(text)}, has_image={bool(jpeg)}")
		print(f"[LLM PROMPT] {text}")
		self._ui_publish_state("THINKING")

		ai_buf: List[str] = []
		audio_buf: List[bytes] = []
		rate_state = None
		printed_text_chunk = False
		printed_audio_chunk = False
		delta_print_open = False
		try:
			async for piece in stream_chat(content_list, voice="Cherry", audio_format="wav"):
				if piece.text_delta:
					if not printed_text_chunk:
						print("[LLM] first text delta arrived")
						printed_text_chunk = True
						# print("[LLM REPLY DELTA] ", end="", flush=True)
						delta_print_open = True
					ai_buf.append(piece.text_delta)
					print(piece.text_delta, end="", flush=True)
					with self.state.lock:
						self.state.partial = "[AI] " + "".join(ai_buf)
					self._ui_publish_model_delta(piece.text_delta)

				if piece.audio_b64:
					if not printed_audio_chunk:
						print("[LLM] first audio chunk arrived")
						printed_audio_chunk = True
					try:
						pcm24 = base64.b64decode(piece.audio_b64)
					except Exception:
						pcm24 = b""
					if pcm24:
						pcm16k, rate_state = audioop.ratecv(pcm24, 2, 1, 24000, 16000, rate_state)
						pcm16k = audioop.mul(pcm16k, 2, 0.8)
						pcm16k = _ensure_even_bytes(pcm16k)
						if pcm16k:
							audio_buf.append(pcm16k)
							self._enqueue_play_chunk(reply_epoch, pcm16k)
		except asyncio.CancelledError:
			print("[LLM] 回复任务已取消")
			self._append_final("[AI] 已停止")
			self._ui_publish_error("LLM回复任务已取消")
			raise
		except Exception as e:
			if delta_print_open:
				print("")
			self._append_final(f"[AI] 发生错误：{e}")
			self._ui_publish_error(f"AI发生错误：{e}")
			print(f"[LLM ERROR] {e}")
		finally:
			if delta_print_open:
				print("")

			final_text = ("".join(ai_buf)).strip() or "（空响应）"
			print(f"[LLM REPLY FINAL] {final_text}")
			self._append_final("[AI] " + final_text)
			self._ui_publish_model_final(final_text)
			with self.state.lock:
				self.state.partial = ""
				self.state.ai_playing = False
				# FIND_PAGE 是一次性任务态，回答结束后回到 IDLE。
				if self.state.state == FIND_PAGE:
					self.state.state = IDLE
			self._ui_publish_state(IDLE)

	def playback_worker(self, stop_flag: threading.Event):
		with sd.RawOutputStream(
			samplerate=16000,
			channels=1,
			dtype="int16",
			blocksize=320,
		) as out:
			while not stop_flag.is_set():
				try:
					item = self.play_queue.get(timeout=0.2)
				except queue.Empty:
					continue
				except Exception:
					continue

				epoch = self._current_play_epoch()
				pcm = b""
				if isinstance(item, tuple) and len(item) == 2:
					item_epoch, item_pcm = item
					if item_epoch != epoch:
						continue
					pcm = item_pcm or b""
				else:
					pcm = item or b""

				if not pcm:
					continue

				off = 0
				step = PCM16_16K_CHUNK_BYTES
				while off < len(pcm) and not stop_flag.is_set():
					if epoch != self._current_play_epoch():
						break
					chunk = pcm[off: off + step]
					off += step
					if chunk:
						if self.session_recorder is not None:
							self.session_recorder.add_ai(chunk)
						out.write(chunk)

	async def asr_loop(self):
		loop = asyncio.get_running_loop()
		mic_frames = 0
		send_failures = 0
		asr_core_mod.ASR_DEBUG_RAW = bool(self.args.asr_debug_raw)
		mic_gain = max(0.1, float(self.args.mic_gain))
		noise_gate_rms = max(0, int(self.args.mic_noise_gate_rms))
		send_timeout = max(0.02, float(self.args.asr_send_timeout))
		keepalive_interval = max(0.05, float(self.args.asr_keepalive_interval))

		# 重建会话前清掉旧队列，避免把失效会话期间的历史音频继续发送。
		while not self.mic_queue.empty():
			try:
				self.mic_queue.get_nowait()
			except Exception:
				break

		def _is_recognition_stopped_error(err: Exception) -> bool:
			msg = str(err or "").lower()
			return "has stopped" in msg or "recognition has stopped" in msg

		def mic_callback(indata, frames, time_info, status):
			nonlocal mic_frames
			if status:
				print(f"[MIC] {status}")
			payload = _ensure_even_bytes(indata.tobytes())
			if not payload:
				return
			if mic_gain != 1.0:
				try:
					payload = audioop.mul(payload, 2, mic_gain)
				except Exception:
					pass
			if noise_gate_rms > 0:
				try:
					rms_now = audioop.rms(payload, 2)
				except Exception:
					rms_now = 0
				if rms_now < noise_gate_rms:
					payload = bytes(len(payload))
			mic_frames += 1
			if mic_frames % 120 == 0:
				try:
					rms = audioop.rms(payload, 2)
				except Exception:
					rms = -1
				print(f"[MIC] frames={mic_frames}, rms={rms}")
			try:
				if self.session_recorder is not None:
					self.session_recorder.add_mic(payload)
				self.mic_queue.put_nowait(payload)
			except queue.Full:
				try:
					self.mic_queue.get_nowait()
					self.mic_queue.put_nowait(payload)
				except Exception:
					pass

		def post(coro):
			asyncio.run_coroutine_threadsafe(coro, loop)

		cb = asr_core_mod.ASRCallback(
			on_sdk_error=lambda s: post(self.ui_partial_from_asr(f"[ASR ERROR] {s}")),
			post=post,
			ui_broadcast_partial=self.ui_partial_from_asr,
			ui_broadcast_final=self.ui_final_from_asr,
			is_playing_now_fn=self._is_playing_now,
			start_ai_with_text_fn=self.start_ai_from_asr,
			full_system_reset_fn=self.full_system_reset,
			interrupt_lock=self.interrupt_lock,
		)
		recognition = dash_audio.asr.Recognition(
			api_key=API_KEY,
			model=ASR_MODEL,
			format=ASR_FORMAT,
			sample_rate=ASR_SAMPLE_RATE,
			language_hints=['zh'],
			callback=cb,
		)
		recognition.start()
		try:
			recognition.send_audio_frame(SILENCE_20MS)
		except Exception:
			pass
		await asr_core_mod.set_current_recognition(recognition)
		print(f"[ASR] started model={ASR_MODEL}")

		in_device = self.args.mic_device
		try:
			all_devs = sd.query_devices()
			default_dev = sd.default.device
			if in_device is None:
				default_in = default_dev[0] if isinstance(default_dev, (list, tuple)) else default_dev
				if isinstance(default_in, int) and 0 <= default_in < len(all_devs):
					dev_name = all_devs[default_in].get("name", "unknown")
					print(f"[MIC] using default input device #{default_in}: {dev_name}")
			else:
				if 0 <= int(in_device) < len(all_devs):
					dev_name = all_devs[int(in_device)].get("name", "unknown")
					print(f"[MIC] using selected input device #{in_device}: {dev_name}")
		except Exception as e:
			print(f"[MIC] query_devices failed: {e}")

		stream = sd.InputStream(
			samplerate=ASR_SAMPLE_RATE,
			channels=1,
			dtype="int16",
			blocksize=int(self.args.mic_blocksize),
			device=in_device,
			callback=mic_callback,
		)
		stream.start()
		print(
			f"[MIC] 连续监听中，按 Q 退出 | gain={mic_gain:.2f}, "
			f"noise_gate_rms={noise_gate_rms}, blocksize={int(self.args.mic_blocksize)}"
		)

		last_keepalive = loop.time()
		try:
			while not self.stop_event.is_set():
				try:
					data = self.mic_queue.get_nowait()
				except queue.Empty:
					await asyncio.sleep(send_timeout)
					now = loop.time()
					if now - last_keepalive >= keepalive_interval:
						try:
							recognition.send_audio_frame(SILENCE_20MS)
							send_failures = 0
							last_keepalive = now
						except Exception as e:
							if _is_recognition_stopped_error(e):
								raise RuntimeError("ASR会话已停止，需要重建") from e
							send_failures += 1
							print(f"[ASR SEND] keepalive failed#{send_failures}: {e}")
							if send_failures >= 8:
								raise RuntimeError("ASR keepalive连续发送失败") from e
					continue
				except Exception:
					await asyncio.sleep(send_timeout)
					continue

				off = 0
				while off < len(data):
					chunk = data[off: off + PCM16_16K_CHUNK_BYTES]
					off += PCM16_16K_CHUNK_BYTES
					if chunk:
						try:
							recognition.send_audio_frame(chunk)
							send_failures = 0
							last_keepalive = loop.time()
						except Exception as e:
							if _is_recognition_stopped_error(e):
								raise RuntimeError("ASR会话已停止，需要重建") from e
							send_failures += 1
							print(f"[ASR SEND] chunk failed#{send_failures}: {e}")
							if send_failures >= 8:
								raise RuntimeError("ASR音频连续发送失败") from e
		finally:
			stream.stop()
			stream.close()
			await asr_core_mod.set_current_recognition(None)
			for _ in range(10):
				try:
					recognition.send_audio_frame(SILENCE_20MS)
				except Exception:
					break
			try:
				recognition.stop()
			except Exception:
				pass

	def save_response_wav(self):
		# 兼容保留：改为按问答即时落盘，不再有全局会话音频收尾动作。
		return

	async def run(self):
		playback_stop = threading.Event()
		playback_thread = threading.Thread(target=self.playback_worker, args=(playback_stop,), daemon=True)
		playback_thread.start()
		self._start_run_session()

		if bool(getattr(self.args, "ui_enable", False)):
			try:
				self.ui_state_store = UIStateStore(max_history=int(self.args.ui_max_history))
				self.ui_publisher = UIPublisher(self.ui_state_store)
				self.ui_server = UIServer(
					state_store=self.ui_state_store,
					host=str(self.args.ui_host),
					port=int(self.args.ui_port),
				)
				self.ui_server.start()
				print(f"[UI] 可视化页面已启动: {self.ui_server.base_url}")
				self._ui_publish_state(IDLE)
			except Exception as e:
				print(f"[UI] 可视化页面启动失败: {e}")

		self.camera_stop.clear()
		self.camera_thread = threading.Thread(target=self._camera_worker, daemon=True)
		self.camera_thread.start()

		try:
			restarts = 0
			asr_task: Optional[asyncio.Task] = None
			while not self.stop_event.is_set():
				asr_task = asyncio.create_task(self.asr_loop())
				done, _ = await asyncio.wait([asr_task], return_when=asyncio.FIRST_COMPLETED)
				if asr_task not in done:
					continue

				exc = asr_task.exception()
				if exc is None:
					if self.stop_event.is_set():
						print("[RUN] ASR循环正常结束")
						break
					restarts += 1
					print(f"[RUN] ASR循环意外结束（无异常，第{restarts}次），准备重建")
					if restarts >= int(self.args.asr_restart_max):
						print("[RUN] ASR重启次数超限，准备退出")
						break
					await asyncio.sleep(0.5)
					continue

				restarts += 1
				print(f"[RUN] ASR循环异常退出（第{restarts}次）: {exc}")
				if restarts >= int(self.args.asr_restart_max):
					print("[RUN] ASR重启次数超限，准备退出")
					break
				sleep_s = 0.2 if restarts == 1 else 0.8
				await asyncio.sleep(sleep_s)
		finally:
			self.stop_event.set()
			if asr_task is not None and not asr_task.done():
				asr_task.cancel()
				await asyncio.gather(asr_task, return_exceptions=True)

			self.camera_stop.set()
			if self.camera_thread is not None:
				self.camera_thread.join(timeout=2.0)

			playback_stop.set()
			playback_thread.join(timeout=2.0)

			if self.ui_server is not None:
				self.ui_server.stop()
			if self.session_recorder is not None:
				self.session_recorder.stop()
				self.session_recorder = None
			self.save_response_wav()


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="纯本地阅读助手：无本地HTTP/WS，OCR优先分辨率")
	parser.add_argument("--camera-index", type=int, default=1, help="摄像头索引")
	parser.add_argument("--idle-camera-width", type=int, default=320, help="待机相机宽度（低负载）")
	parser.add_argument("--idle-camera-height", type=int, default=240, help="待机相机高度（低负载）")
	parser.add_argument("--idle-camera-fps", type=float, default=5.0, help="待机相机帧率（低负载）")
	parser.add_argument("--command-camera-width", type=int, default=1920, help="指令抓拍宽度（高质量）")
	parser.add_argument("--command-camera-height", type=int, default=1080, help="指令抓拍高度（高质量）")
	parser.add_argument("--command-camera-fps", type=float, default=30.0, help="指令抓拍帧率")
	parser.add_argument("--command-capture-frames", type=int, default=4, help="每次指令抓拍帧数")
	parser.add_argument("--command-capture-gap-ms", type=int, default=15, help="抓拍帧间隔毫秒")
	parser.add_argument("--command-jpeg-quality", type=int, default=100, help="指令抓拍JPEG质量")
	parser.add_argument("--open-command-image", action="store_true", help="抓拍后自动打开图片（Windows）")
	parser.add_argument("--asr-only", action="store_true", help="仅运行ASR链路，不向Qwen发起请求")
	parser.add_argument("--mic-device", type=int, default=None, help="麦克风输入设备索引，默认系统输入设备")
	parser.add_argument("--mic-gain", type=float, default=1.4, help="麦克风增益倍数，提升弱音识别")
	parser.add_argument("--mic-noise-gate-rms", type=int, default=20, help="噪声门限RMS，低于该值按静音处理")
	parser.add_argument("--mic-blocksize", type=int, default=320, help="麦克风块大小(采样点)，更小可降低延迟")
	parser.add_argument("--asr-send-timeout", type=float, default=0.05, help="ASR发送轮询超时(秒)，更小更实时")
	parser.add_argument("--asr-keepalive-interval", type=float, default=0.30, help="ASR静音保活间隔(秒)")
	parser.add_argument("--asr-restart-max", type=int, default=5, help="ASR异常退出后的最大自动重启次数")
	parser.add_argument("--asr-debug-raw", action="store_true", help="打印ASR原始事件与解析日志")
	parser.add_argument("--min-query-chars", type=int, default=2, help="最终发送给LLM的最少归一化字符数")
	parser.add_argument("--duplicate-final-window", type=float, default=15.0, help="相同final去重时间窗(秒)")
	parser.add_argument("--ui-enable", action="store_true", help="启用本地可视化页面")
	parser.add_argument("--ui-host", type=str, default="127.0.0.1", help="本地可视化页面监听地址")
	parser.add_argument("--ui-port", type=int, default=8765, help="本地可视化页面监听端口")
	parser.add_argument("--ui-max-history", type=int, default=15, help="可视化页面保留的最近对话条数")
	parser.add_argument("--tts-rate", type=int, default=180, help="本地TTS语速（pyttsx3）")
	parser.add_argument("--tts-volume", type=float, default=1.0, help="本地TTS音量，范围0.0-1.0")
	parser.add_argument("--tts-voice-name", type=str, default="", help="本地TTS音色关键词（匹配voice name/id）")
	parser.add_argument("--response-save-dir", default=os.path.join(os.path.dirname(__file__), "save"), help="回复音频会话保存根目录")
	return parser


async def _main_async(args):
	ctrl = LocalReaderController(args)
	await ctrl.run()


def main():
	args = build_parser().parse_args()
	try:
		asyncio.run(_main_async(args))
	except KeyboardInterrupt:
		print("\n[SYSTEM] 已收到 Ctrl+C，结束运行。")


if __name__ == "__main__":
	main()

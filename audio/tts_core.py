#!/usr/bin/env python
# -*- coding: utf-8 -*-

import audioop
import os
import wave
from dataclasses import dataclass
from typing import Optional

try:
	import pyttsx3
except Exception:
	pyttsx3 = None


@dataclass
class TTSOptions:
	rate: int = 180
	volume: float = 1.0
	voice_name: str = ""


def _ensure_even_bytes(data: bytes) -> bytes:
	if len(data) % 2 == 1:
		return data[:-1]
	return data


def synthesize_tts_wav(text: str, wav_path: str, opts: Optional[TTSOptions] = None) -> bool:
	if pyttsx3 is None:
		print("[TTS] pyttsx3 未安装，无法进行本地播报")
		return False

	opts = opts or TTSOptions()
	try:
		engine = pyttsx3.init()
		engine.setProperty("rate", int(opts.rate))
		engine.setProperty("volume", max(0.0, min(1.0, float(opts.volume))))

		voice_name = str(opts.voice_name or "").strip().lower()
		if voice_name:
			try:
				voices = engine.getProperty("voices") or []
				for v in voices:
					name = str(getattr(v, "name", "") or "").lower()
					vid = str(getattr(v, "id", "") or "").lower()
					if voice_name in name or voice_name in vid:
						engine.setProperty("voice", getattr(v, "id", ""))
						break
			except Exception:
				pass

		engine.save_to_file(text, wav_path)
		engine.runAndWait()
		engine.stop()
		ok = os.path.exists(wav_path) and os.path.getsize(wav_path) > 44
		if not ok:
			print("[TTS] 合成结束但未生成有效音频文件")
		return ok
	except Exception as e:
		print(f"[TTS] 合成失败: {e}")
		return False


def load_wav_to_pcm16k_mono(wav_path: str) -> bytes:
	with wave.open(wav_path, "rb") as wf:
		n_channels = max(1, int(wf.getnchannels()))
		sampwidth = max(1, int(wf.getsampwidth()))
		fr = max(1, int(wf.getframerate()))
		raw = wf.readframes(wf.getnframes())

	if not raw:
		return b""

	if n_channels > 1:
		raw = audioop.tomono(raw, sampwidth, 0.5, 0.5)

	if sampwidth != 2:
		raw = audioop.lin2lin(raw, sampwidth, 2)

	if fr != 16000:
		raw, _ = audioop.ratecv(raw, 2, 1, fr, 16000, None)

	return _ensure_even_bytes(raw)

import wave
import contextlib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "向前翻页.wav")
# file_path = os.path.join(BASE_DIR, "向后翻页.wav")
# file_path = os.path.join(BASE_DIR, "右转一点.wav")

with contextlib.closing(wave.open(file_path, 'r')) as f:
   frames = f.getnframes()
   rate = f.getframerate()
   duration = frames / float(rate)
   print(f"{duration * 1000} ms")
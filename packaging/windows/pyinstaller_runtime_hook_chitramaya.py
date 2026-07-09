import os
import sys
from pathlib import Path

# Runs at frozen-app startup, before heavy imports. Ensures the bundle's native
# DLLs (torch, PyNvVideoCodec, TensorRT, cv2) and the bundled ffmpeg/ffprobe
# resolve first — otherwise a frozen CUDA app dies with "DLL load failed" and
# the app's ffmpeg shell-outs fail on a clean machine.
base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))

paths = [str(base), str(base / "bin")]

# Prepend, then keep the system PATH (CUDA driver / cuDNN may live there).
system_path = os.environ.get("PATH", "")
if system_path:
    paths.append(system_path)
os.environ["PATH"] = os.pathsep.join(paths)

# Point ffmpeg-using code at the bundled binaries explicitly (belt + braces
# alongside PATH), if present.
_ff = base / "bin" / "ffmpeg.exe"
_fp = base / "bin" / "ffprobe.exe"
if _ff.exists():
    os.environ.setdefault("CHITRAMAYA_FFMPEG", str(_ff))
if _fp.exists():
    os.environ.setdefault("CHITRAMAYA_FFPROBE", str(_fp))

os.environ.setdefault("CHITRAMAYA_HOME", str(base))

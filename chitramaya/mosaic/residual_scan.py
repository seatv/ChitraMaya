# chitramaya/mosaic/residual_scan.py
"""CM-205: residual scan -- run the detector over the OUTPUT after a run.

Why: the run's own counters classify FRAMES. A frame with one region
restored and a second region the detector never saw counts as "restored",
so ``visible_miss`` is a floor, not what a viewer sees (the 4K60 cut, frame
783: a row of mosaic blocks beside the restored region, invisible to every
counter, found in the headset). The only instrument that sees what the
viewer sees is the detector looking at the finished video.

How: ffmpeg decodes the output's KEY frames only (``-skip_frame nokey``;
NVENC writes one every 2 s, see video/encoder.py gop), scaled to the
detector's input size on the long side, piped as raw BGR; the run's own
warm detector scores them with the run's confidence. Key frames only keeps
the cost under 1 % of the run on any title (I-frame decode is cheap; the
pipe carries ~1 MB per frame) and gives a sample every couple of seconds,
which is the granularity a viewer notices anyway. Frame numbers come from
the showinfo filter's pts_time, so they line up with the run report's other
frame lists and with Test Frame.

Reading the result: a detection on the output is either mosaic the run
left behind (a region the detector did not see during the run -- the same
detector, so usually an occlusion split or a region at the frame edge --
or one it saw and the restorer did not clear) or the detector firing on
restored texture. The report says, per frame, whether the run restored
that frame, which tells the two apart in most cases; the frames are
listed for Test Frame either way. This is a measurement, not a judgment:
it never changes the output.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import torch

_SHOWINFO_PTS = re.compile(r"pts_time:\s*([0-9.eE+-]+)")


def _read_exact(stream, n: int) -> Optional[bytes]:
    """Read exactly n bytes from an unbuffered pipe (short reads are normal);
    None at EOF before n bytes."""
    chunks = []
    got = 0
    while got < n:
        b = stream.read(n - got)
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def _stderr_reader(proc: subprocess.Popen, pts_out: List[float], tail: List[str]) -> None:
    """Collect one pts_time per selected frame (showinfo prints in order)."""
    try:
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", errors="replace")
            if "showinfo" in line:
                m = _SHOWINFO_PTS.search(line)
                if m:
                    try:
                        pts_out.append(float(m.group(1)))
                    except ValueError:
                        pass
            else:
                tail.append(line.rstrip())
                if len(tail) > 40:
                    del tail[:-40]
    except Exception:
        pass


def scan_output(
    output_path: str,
    detector,
    *,
    ffmpeg: str = "ffmpeg",
    device: Optional[torch.device] = None,
    fps: float,
    frame_w: int,
    frame_h: int,
    imgsz: int = 640,
    conf: float = 0.25,
    batch_size: int = 4,
    restored_frames: Optional[Set[int]] = None,
    max_samples: int = 20000,
    timeout_s: float = 900.0,
    cancel_flag=None,
) -> Dict[str, Any]:
    """Scan ``output_path`` key frames with ``detector``. Never raises; on a
    failure the result carries ``error`` and the caller prints one line."""
    t0 = time.perf_counter()
    res: Dict[str, Any] = {
        "method": "keyframes",
        "conf": float(conf),
        "sampled": 0,
        "residual_count": 0,
        "residual_frames": [],
        "residual_in_restored": 0,
        "residual_in_untouched": 0,
        "top": [],
        "scan_seconds": 0.0,
    }
    if not output_path or not os.path.isfile(output_path):
        res["error"] = "output file not found"
        return res

    fps = float(fps) if fps and fps > 0 else 30.0
    w, h = int(frame_w), int(frame_h)
    long_side = max(1, int(imgsz))
    # Scale the long side to imgsz; -2 keeps the other side even. The
    # detector letterboxes to imgsz anyway, so this only changes which
    # resampler shrinks the frame (ffmpeg bicubic vs the detector's).
    vf_scale = f"scale={long_side}:-2" if w >= h else f"scale=-2:{long_side}"
    if w >= h:
        sw = long_side
        sh = max(2, int(round(h * long_side / max(1, w) / 2.0)) * 2)
    else:
        sh = long_side
        sw = max(2, int(round(w * long_side / max(1, h) / 2.0)) * 2)
    frame_bytes = sw * sh * 3

    cmd = [
        str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "info",
        "-skip_frame", "nokey",
        "-i", str(output_path),
        "-an", "-sn", "-dn",
        "-vf", f"select='eq(pict_type\\,I)',showinfo,{vf_scale}",
        "-fps_mode", "passthrough",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
    except Exception as e:
        res["error"] = f"ffmpeg could not start: {e}"
        return res

    pts_list: List[float] = []
    err_tail: List[str] = []
    th = threading.Thread(target=_stderr_reader, args=(proc, pts_list, err_tail), daemon=True)
    th.start()

    dev = device if device is not None else torch.device("cpu")
    frames_buf: List[torch.Tensor] = []
    idx_buf: List[int] = []
    n_read = 0
    hits: List[Dict[str, Any]] = []
    restored = restored_frames or set()
    sx = float(w) / float(sw)
    sy = float(h) / float(sh)
    stopped = ""

    def _flush() -> None:
        nonlocal frames_buf, idx_buf
        if not frames_buf:
            return
        dets = detector.detect_batch(frames_buf)
        for k, det in enumerate(dets):
            b = getattr(det, "boxes", None)
            s = getattr(det, "scores", None)
            n = int(b.shape[0]) if (b is not None and hasattr(b, "shape")) else 0
            if n <= 0:
                continue
            i = idx_buf[k]
            scores = [float(x) for x in s.tolist()] if s is not None else []
            boxes = []
            for row in b.tolist():
                x1, y1, x2, y2 = row[:4]
                boxes.append([int(round(y1 * sy)), int(round(x1 * sx)),
                              int(round(y2 * sy)), int(round(x2 * sx))])   # t,l,b,r frame px
            hits.append({"frame": i, "score": max(scores) if scores else 0.0,
                         "n": n, "boxes": boxes})
        frames_buf = []
        idx_buf = []

    try:
        while True:
            if cancel_flag is not None and cancel_flag.is_set():
                stopped = "cancelled"
                break
            if time.perf_counter() - t0 > timeout_s:
                stopped = f"stopped after {int(timeout_s)} s"
                break
            if n_read >= max_samples:
                stopped = f"stopped at {max_samples} samples"
                break
            buf = _read_exact(proc.stdout, frame_bytes)
            if buf is None:
                break
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(sh, sw, 3)
            # pts for this frame: showinfo runs ahead of the pipe, so the
            # list normally has it already; wait briefly if not.
            for _ in range(200):
                if len(pts_list) > n_read:
                    break
                time.sleep(0.005)
            if len(pts_list) > n_read:
                fnum = int(round(pts_list[n_read] * fps))
            else:
                fnum = int(round(n_read * 2.0 * fps))   # fallback: assume the 2 s GOP
            t = torch.from_numpy(arr.copy()).to(device=dev, non_blocking=False)
            frames_buf.append(t)
            idx_buf.append(fnum)
            n_read += 1
            if len(frames_buf) >= max(1, int(batch_size)):
                _flush()
        _flush()
    except Exception as e:
        res["error"] = f"scan aborted: {e}"
    finally:
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
        th.join(timeout=2)

    if n_read == 0 and "error" not in res:
        res["error"] = "ffmpeg produced no key frames" + (f" ({err_tail[-1]})" if err_tail else "")

    hits.sort(key=lambda d: d["frame"])
    frames_hit = [d["frame"] for d in hits]
    in_restored = sum(1 for f in frames_hit if f in restored)
    res.update({
        "sampled": int(n_read),
        "sample_spacing_s": (round((pts_list[-1] - pts_list[0]) / max(1, len(pts_list) - 1), 2)
                             if len(pts_list) > 1 else None),
        "residual_count": len(hits),
        "residual_pct": (round(100.0 * len(hits) / n_read, 2) if n_read else 0.0),
        "residual_frames": frames_hit,
        "residual_in_restored": int(in_restored),
        "residual_in_untouched": int(len(hits) - in_restored),
        "top": [{"frame": d["frame"], "score": round(d["score"], 3), "n": d["n"],
                 "boxes": d["boxes"][:4], "run_restored_frame": bool(d["frame"] in restored)}
                for d in sorted(hits, key=lambda d: -d["score"])[:20]],
        "scan_seconds": round(time.perf_counter() - t0, 1),
        "sampled_size": [sw, sh],
    })
    if stopped:
        res["stopped"] = stopped
    return res


def format_console(res: Dict[str, Any]) -> List[str]:
    """The console lines for a scan result (ASCII only)."""
    if res.get("error"):
        return [f"[Residual] scan skipped: {res['error']}"]
    n = int(res.get("sampled", 0))
    sp = res.get("sample_spacing_s")
    every = f" (one every {sp:.1f} s)" if isinstance(sp, (int, float)) and sp else ""
    k = int(res.get("residual_count", 0))
    lines: List[str] = []
    if k == 0:
        lines.append(f"[Residual] scanned {n} key frames of the output{every}: no mosaic detected "
                     f"at conf {res.get('conf', 0):.2f} ({res.get('scan_seconds', 0)} s)")
        return lines
    lines.append(f"[Residual] scanned {n} key frames of the output{every}: mosaic still detected on "
                 f"{k} ({res.get('residual_pct', 0):.1f} %) -- {res.get('residual_in_restored', 0)} inside "
                 f"frames the run restored (a region or a rim the run did not clear), "
                 f"{res.get('residual_in_untouched', 0)} on frames the run left untouched "
                 f"(a miss, or the detector firing on clean content) -- {res.get('scan_seconds', 0)} s")
    top = res.get("top") or []
    if top:
        shown = ", ".join(f"{d['frame']} ({d['score']:.2f}{', restored' if d.get('run_restored_frame') else ''})"
                          for d in top[:10])
        lines.append(f"[Residual] highest-scoring frames: {shown}. Open one with Test Frame; "
                     f"the full list is in the run report (frames.residual).")
    if res.get("stopped"):
        lines.append(f"[Residual] note: {res['stopped']}")
    return lines


__all__ = ["scan_output", "format_console"]

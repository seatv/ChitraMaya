# tools/count_delivered.py
"""CM-120 diagnostic: how many frames does PyNvVideoCodec actually deliver?

Field context (2026-08-27): ThreadedDecoder silently delivers fewer frames
than MPEG-TS stream captures contain (5.0s worth, head + interior), and its
2.2-era frame timestamps are SYNTHESIZED (uniform index*interval), so the
loss is invisible to timestamp math. This counter isolates the container
variable: run it against the original file and against a lossless remux
(ffmpeg -c copy to mp4/mkv) -- if the remux delivers the full count, the
TS container is the trigger and auto-remux-before-decode is the fix.

Usage (venv with PyNvVideoCodec, GPU idle):
    python tools/count_delivered.py <video path>

Compare against ffprobe's truth:
    ffprobe -v error -select_streams v:0 -count_packets \
        -show_entries stream=nb_read_packets -of csv=p=0 <video path>
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python tools/count_delivered.py <video path>")
        return 2

    path = sys.argv[1]
    import PyNvVideoCodec as nvc

    dec = nvc.ThreadedDecoder(
        enc_file_path=path,
        buffer_size=32,
        gpu_id=0,
        output_color_type=nvc.OutputColorType.RGBP,
        use_device_memory=True,
        need_scanned_stream_metadata=True,
    )

    promised = 0
    try:
        meta = dec.get_scanned_stream_metadata()
        promised = int(getattr(meta, "num_frames", 0) or 0)
    except Exception:
        try:
            meta = dec.get_stream_metadata()
            promised = int(getattr(meta, "num_frames", 0) or 0)
        except Exception:
            promised = 0

    n = 0
    first_ts = last_ts = None
    t0 = time.perf_counter()
    while True:
        frames = dec.get_batch_frames(32)
        if not frames:
            break
        for fr in frames:
            ts = getattr(fr, "timestamp", None)
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
        n += len(frames)
        if n % 25000 < 32:
            print(f"  ... {n} frames")
    dt = time.perf_counter() - t0

    print(f"file:      {path}")
    print(f"promised:  {promised} (scanned stream metadata)")
    print(f"delivered: {n}   ({dt:.1f}s, {n / dt if dt > 0 else 0:.0f} fps)")
    if promised and n < promised:
        print(f"SHORTFALL: {promised - n} frames missing "
              f"({(promised - n) / max(1, promised) * 100:.2f}%)")
    elif promised:
        print("FULL DELIVERY: decoder handed over every frame the stream reports.")
    print(f"first frame timestamp: {first_ts}")
    print(f"last  frame timestamp: {last_ts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

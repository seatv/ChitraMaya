# tools/diag_tight_crop.py
"""Feed the restorer a hand-drawn rectangle and show what comes back.

The question this answers (09-17, the 4K60 cut, frame 783): a row of mosaic
blocks beside an occluder sat INSIDE the restorer's crop -- at the rim,
next to clean skin -- and came back as mosaic. Is that framing (the model
treats the rim as reference) or recognition (the model never clears a lone
block row)? If a TIGHT crop, where the strip dominates, comes back clean,
it is framing and a pipeline fix exists (CM-210 "second look"); if not, it
is the model's, and the answer is training data (Phase C).

This script builds a clip exactly the way the tracker does -- the same
crop_box_to_target_v3 geometry (border, square-ish expansion, 256 resize,
reflect pad) -- from a rectangle YOU give in frame pixels, runs the
PyTorch restorer on it, and writes pairs to look at:

  <out>/frameN_crop_in.png / _crop_out.png   the 256 x 256 the model saw / returned
  <out>/frameN_region_before.png / _after.png the rectangle (+ context) in the frame, before / after paste-back (Blend None)

No detector, no masks: the rectangle IS the region (Use Seg Masks off),
so the paste covers the whole rectangle -- what you see after is what the
restorer did, nothing hidden by an alpha.

Usage (dev box, the .pth, no TensorRT needed):
  python tools/diag_tight_crop.py --video D:\\cut.mp4 --frames 780-790 \\
      --rect 380,1060,420,1110 --rest-model models\\lada_mosaic_restoration_model_generic_v1.2.pth \\
      --out D:\\diag\\783 [--margin 0] [--fp16] [--border 0.06]

--rect is t,l,b,r in frame pixels (inclusive), as the mask preview / Test
Frame report them. Run it three ways: the whole detected region (the run's
own framing), the leftover strip with the tracker's geometry (--expansion
1.0: the crop still widens toward a square, so the strip stays a minority of
the crop), and the strip with --expansion 0 (border only: the tightest
framing the geometry allows). --margin grows the rectangle like Region
Margin would. The [diag] line prints what share of the crop the rectangle
occupies -- that number is the experiment's x-axis.
ASCII-only output. Content stays local; nothing is uploaded anywhere.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from chitramaya.mosaic.core.scene import Scene  # noqa: E402
from chitramaya.mosaic.core.clip import Clip  # noqa: E402
from chitramaya.mosaic.core.scene_tracker import SceneTracker, TrackerConfig  # noqa: E402
from chitramaya.mosaic.restorer import compositor  # noqa: E402


def _probe(video: str, ffprobe: str):
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,r_frame_rate",
                          "-of", "csv=p=0", video], capture_output=True, text=True, check=True).stdout.strip()
    w, h, fr = out.split(",")[:3]
    num, den = fr.split("/")
    return int(w), int(h), float(num) / float(den or 1)


def _decode_frames(video: str, a: int, b: int, w: int, h: int, ffmpeg: str):
    """Frames a..b inclusive as uint8 HWC BGR numpy arrays (frame-accurate via select)."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", video,
           "-vf", f"select='between(n\\,{a}\\,{b})'", "-fps_mode", "passthrough",
           "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    arr = np.frombuffer(raw[: n * w * h * 3], dtype=np.uint8).reshape(n, h, w, 3)
    return [arr[i].copy() for i in range(n)]


def _save_png(path: Path, bgr_u8: np.ndarray) -> None:
    import cv2
    cv2.imwrite(str(path), bgr_u8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", required=True, help="A-B inclusive, e.g. 780-790")
    ap.add_argument("--rect", default="", help="t,l,b,r in frame pixels (inclusive)")
    ap.add_argument("--rest-model", default="", help="the .pth (PyTorch path; no TensorRT needed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=int, default=0, help="grow the rectangle by N px each side (Region Margin)")
    ap.add_argument("--border", type=float, default=0.06, help="crop border ratio (the tracker's 0.06)")
    ap.add_argument("--expansion", type=float, default=1.0,
                    help="how far the crop may widen toward a square, as a multiple of the rectangle (the tracker's 1.0; "
                         "0 = border only, the tightest framing the geometry allows)")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ffmpeg", default=os.environ.get("CHITRAMAYA_FFMPEG", "ffmpeg"))
    ap.add_argument("--context", type=int, default=120, help="extra px shown around the rectangle in the region PNGs")
    ap.add_argument("--pseudo", action="store_true", help="plumbing check: magenta fill instead of the model (no weights needed)")
    ap.add_argument("--dump", action="store_true",
                    help="write the FIRST frame of --frames as a full-size PNG into --out and exit -- open it at 100 %% "
                         "in Paint / IrfanView and read the rectangle corners (t,l,b,r) off the status bar; --rect is not needed")
    args = ap.parse_args()

    a, b = [int(x) for x in args.frames.split("-")]
    _d, _b = os.path.dirname(args.ffmpeg), os.path.basename(args.ffmpeg)
    ffprobe = os.path.join(_d, _b.replace("ffmpeg", "ffprobe")) if "ffmpeg" in _b else "ffprobe"
    w, h, fps = _probe(args.video, ffprobe)
    if args.dump:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        fr = _decode_frames(args.video, a, a, w, h, args.ffmpeg)
        if not fr:
            print("[diag] no frame decoded"); return 2
        pth = out / f"frame{a}_full.png"
        _save_png(pth, fr[0])
        print(f"[diag] wrote {pth} ({w}x{h}). Open at 100 %, read top,left,bottom,right of the region in pixels, "
              f"then run again with --rect t,l,b,r")
        return 0
    if not args.rect:
        print("[diag] --rect t,l,b,r is required (or use --dump to get the frame to measure)"); return 2
    if not args.pseudo and (not args.rest_model or not os.path.isfile(args.rest_model)):
        print(f"[diag] --rest-model must point at an existing .pth (got {args.rest_model!r})"); return 2
    t, l, bb, r = [int(x) for x in args.rect.split(",")]
    m = int(args.margin)
    roi = (max(0, t - m), max(0, l - m), min(h - 1, bb + m), min(w - 1, r + m))
    print(f"[diag] video {w}x{h} @ {fps:.3f} fps; frames {a}-{b}; rect {(t, l, bb, r)} margin {m} -> roi {roi}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    frames = _decode_frames(args.video, a, b, w, h, args.ffmpeg)
    if not frames:
        print("[diag] no frames decoded"); return 2
    print(f"[diag] decoded {len(frames)} frames")

    # Build the scene exactly as the tracker would for this rectangle.
    trk = SceneTracker(cfg=TrackerConfig(clip_size=256, border_size=float(args.border),
                                         max_box_expansion_factor=float(args.expansion),
                                         use_seg_masks=False, roi_dilate_px=0), seg_mask_only=False)
    scene = Scene(id=0, start_frame=a)
    dev_frames = []
    for i, f in enumerate(frames):
        ft = torch.from_numpy(f).to(device)
        dev_frames.append(ft)
        crop_box, crop_img, crop_mask = trk._compute_crop(ft, roi, None)
        scene.add_frame(frame_num=a + i, roi_box=roi, crop_box=crop_box, crop_img=crop_img, crop_mask=crop_mask)
    ct, cl, cb, cr = scene.crop_boxes[0]
    print(f"[diag] crop box {scene.crop_boxes[0]} = {cr - cl + 1}x{cb - ct + 1} px -> resized to fit 256 "
          f"(scale {256 / max(cr - cl + 1, cb - ct + 1):.3f}); the rectangle is "
          f"{roi[3] - roi[1] + 1}x{roi[2] - roi[0] + 1} px = {100.0 * (roi[3] - roi[1] + 1) * (roi[2] - roi[0] + 1) / ((cr - cl + 1) * (cb - ct + 1)):.0f} % of the crop")
    clip = Clip.from_scene(scene, clip_id=0, clip_size=256, pad_mode="reflect")

    if args.pseudo:
        from chitramaya.mosaic.restorer.pseudo_clip_restorer import PseudoClipRestorer
        restorer = PseudoClipRestorer(device=device)
    else:
        from chitramaya.mosaic.restorer.basicvsrpp_clip_restorer import BasicVSRPPClipRestorer
        restorer = BasicVSRPPClipRestorer(model_path=args.rest_model, device=device, fp16=bool(args.fp16), max_frames=len(frames))
    restored = restorer.restore_clip(clip)
    print(f"[diag] restorer returned {len(restored)} frames")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    store = {a + i: dev_frames[i].clone() for i in range(len(frames))}
    compositor.composite_clip_into_store(clip=clip, restored_frames_u8=restored, store_bgr_u8=store,
                                         model_dtype=torch.float16 if args.fp16 else torch.float32,
                                         blendmask="none", feather_radius=0, secondary=None)
    cx = int(args.context)
    for i in range(len(frames)):
        n = a + i
        _save_png(out / f"frame{n}_crop_in.png", clip.frames[i].detach().cpu().numpy())
        _save_png(out / f"frame{n}_crop_out.png", restored[i].detach().cpu().numpy())
        t0, l0 = max(0, roi[0] - cx), max(0, roi[1] - cx)
        b0, r0 = min(h - 1, roi[2] + cx), min(w - 1, roi[3] + cx)
        _save_png(out / f"frame{n}_region_before.png", frames[i][t0:b0 + 1, l0:r0 + 1])
        _save_png(out / f"frame{n}_region_after.png", store[n][t0:b0 + 1, l0:r0 + 1].detach().cpu().numpy())
    print(f"[diag] wrote {4 * len(frames)} PNGs to {out}")
    print("[diag] look at frameN_crop_in vs frameN_crop_out (what the model saw / returned) and "
          "frameN_region_before vs _after (the rectangle pasted back, Blend None, whole rectangle).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

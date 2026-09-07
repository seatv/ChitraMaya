# tools/make_rest_pairs.py
r"""CM-112 Phase C (Batch T3): build perfect-pair CLIP data for restorer fine-tuning.

Pristine (uncensored) videos in -> (GT clip, LQ clip) pairs out, where the
LQ side carries a mosaic rendered by ChitraMaya's OWN pixelate core plus a
video-compression degradation, and BOTH sides are cropped/resized by the
pipeline's OWN scene tracker. Train/inference parity at the tensor level is
the point: what the trainer sees is exactly what the restorer sees at run
time (256x256 BGR uint8 clips, reflect-padded, same box-expansion math).

    ChitraMaya -make-pairs --input clip.mp4 --out D:\Train\pairs ^
        --det-model models\lada_nsfw_detection_model_v1.3.pt

Region source (--regions):
  nsfw   (default) lada's anatomy detector places the mosaic where a studio
         would -- the restorer must learn anatomy priors, not generic texture.
  random textured random rectangles (no detector; also the sandbox path).
  mix    both, alternating windows.

Degradation recipe v1 (the LQ side, per clip):
  1. pixelation in the FULL frame before cropping (block size in frame
     pixels, fixed for a window like a studio's mosaic), or -- a minority
     class -- a Gaussian-blur mosaic;
  2. optional resize round-trip (downscale/upscale, like a re-release);
  3. an H.264 re-encode of the clip at a random CRF (video compression, not
     per-frame JPEG: temporal artifacts matter to a video model).
  DIVERSITY over fidelity to any one renderer: matching the exam inflates
  scores without helping on studio mosaic.

Output layout (LOCAL artifact -- content never leaves the machine):
  <out>/pairs.json                      manifest (recipe, per-pair metadata)
  <out>/pairs/<id>/gt.png               T frames stacked vertically (T*256 x 256)
  <out>/pairs/<id>/lq.png               same layout, degraded
  <out>/pairs/<id>/mask.png             same layout, uint8 region mask
One file per clip side keeps training reads to two opens per step, which is
what makes a spinning disk workable.

Doctrine: restoration-only, supervised perfect-pair, no generative priors.
Datasets are local artifacts; weights travel, content never does.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402
from chitramaya.mosaic.add_mosaic import pixelate_roi_bgr_u8_inplace  # noqa: E402
from chitramaya.mosaic.core.scene_tracker import SceneTracker, TrackerConfig  # noqa: E402
from chitramaya.mosaic.pipeline import _extract_masks_list, _tensor_boxes_to_list_xyxy  # noqa: E402
from chitramaya.mosaic.pipeline_utils import clip_box_to_bounds  # noqa: E402
from tools.make_det_dataset import _rand_region, _textured_enough  # noqa: E402

RECIPE_VERSION = "pairs-v1"
CLIP_SIZE = 256


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _ffmpeg_exe() -> str:
    """Same resolution the self-check uses: env override, then PATH (the
    frozen dist puts its bundled ffmpeg on PATH at startup)."""
    return os.environ.get("CHITRAMAYA_FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def _nowindow() -> dict:
    try:
        from chitramaya.winproc import NOWINDOW
        return dict(NOWINDOW)
    except Exception:
        return {}


def _blur_roi_inplace(frame_u8: torch.Tensor, roi, sigma: float) -> None:
    """Gaussian-blur mosaic (the minority studio style)."""
    t, l, b, r = roi
    h, w = int(frame_u8.shape[0]), int(frame_u8.shape[1])
    t = max(0, min(t, h - 1)); b = max(0, min(b, h - 1))
    l = max(0, min(l, w - 1)); r = max(0, min(r, w - 1))
    if b - t < 2 or r - l < 2:
        return
    patch = frame_u8[t:b + 1, l:r + 1, :].cpu().numpy()
    k = int(max(3, (round(sigma * 3) * 2 + 1)))
    blurred = cv2.GaussianBlur(patch, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
    frame_u8[t:b + 1, l:r + 1, :] = torch.from_numpy(blurred).to(frame_u8.device)


def _video_roundtrip_bgr(frames: list[np.ndarray], crf: int, fps: int, ffmpeg: str) -> list[np.ndarray] | None:
    """Encode a T-frame clip with libx264 at `crf` and decode it back.

    Returns decoded frames (same count) or None if ffmpeg failed -- callers
    then keep the un-compressed LQ (never silently drop the pair).
    """
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    raw = b"".join(np.ascontiguousarray(f).tobytes() for f in frames)
    enc = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
           "-pix_fmt", "yuv420p", "-g", str(max(1, len(frames))), "-f", "h264", "-"]
    try:
        p1 = subprocess.run(enc, input=raw, capture_output=True, timeout=120, **_nowindow())
        if p1.returncode != 0 or not p1.stdout:
            return None
        dec = [ffmpeg, "-hide_banner", "-loglevel", "error",
               "-f", "h264", "-r", str(fps), "-i", "-",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        p2 = subprocess.run(dec, input=p1.stdout, capture_output=True, timeout=120, **_nowindow())
        if p2.returncode != 0:
            return None
    except Exception:
        return None
    n = len(p2.stdout) // (h * w * 3)
    if n < len(frames):
        return None
    arr = np.frombuffer(p2.stdout[: len(frames) * h * w * 3], dtype=np.uint8)
    arr = arr.reshape(len(frames), h, w, 3)
    return [np.ascontiguousarray(arr[i]) for i in range(len(frames))]


def _resize_roundtrip(frames: list[np.ndarray], factor: float) -> list[np.ndarray]:
    out = []
    for f in frames:
        h, w = f.shape[:2]
        sh, sw = max(8, int(h * factor)), max(8, int(w * factor))
        small = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
        out.append(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR))
    return out


def _strip(frames: list[np.ndarray]) -> np.ndarray:
    """Stack T HxWx3 frames vertically into one (T*H)xWx3 image."""
    return np.concatenate([np.ascontiguousarray(f) for f in frames], axis=0)


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().numpy()


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _make_tracker(clip_len: int) -> SceneTracker:
    # Same knobs the pipeline uses (pipeline.py defaults): clip 256, reflect
    # pad, 6% border, seg masks, TTL 3, match pad 8, no sticky crop.
    cfg = TrackerConfig(
        clip_size=CLIP_SIZE,
        max_clip_length=int(clip_len),
        pad_mode="reflect",
        border_size=0.06,
        use_seg_masks=True,
        ttl_after_end=3,
        crop_quant_px=0,
        crop_sticky=False,
        match_pad_px=8,
    )
    return SceneTracker(cfg=cfg, seg_mask_only=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="CM-112 Phase C: build restorer training pairs")
    ap.add_argument("--input", action="append", required=True,
                    help="pristine source video (repeatable)")
    ap.add_argument("--out", required=True, help="output folder (user-chosen; LOCAL)")
    ap.add_argument("--regions", choices=("nsfw", "random", "mix"), default="nsfw")
    ap.add_argument("--det-model", default=None,
                    help="anatomy detector .pt (lada_nsfw); required for --regions nsfw/mix")
    ap.add_argument("--det-imgsz", type=int, default=640)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--device", default="0", help="CUDA device id, or 'cpu'")
    ap.add_argument("--clip-len", type=int, default=30, help="max frames per pair (tracker MCL)")
    ap.add_argument("--min-clip-len", type=int, default=8, help="discard shorter clips")
    ap.add_argument("--window-frames", type=int, default=120,
                    help="consecutive frames decoded per sampled window")
    ap.add_argument("--windows-per-video", type=int, default=40)
    ap.add_argument("--pairs-per-video", type=int, default=200, help="stop a video after N pairs")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    # degradation recipe v1
    ap.add_argument("--block-min-frac", type=float, default=0.012,
                    help="mosaic block = frame height * U(min,max) px, clamped 4..max(48, 0.045*H)")
    ap.add_argument("--block-max-frac", type=float, default=0.035)
    ap.add_argument("--blur-frac", type=float, default=0.10,
                    help="fraction of windows using a blur-style mosaic instead of pixelation")
    ap.add_argument("--resize-frac", type=float, default=0.25,
                    help="fraction of clips given a downscale/upscale round-trip")
    ap.add_argument("--crf-min", type=int, default=18)
    ap.add_argument("--crf-max", type=int, default=30)
    ap.add_argument("--no-compress", action="store_true", help="skip the H.264 round-trip")
    ap.add_argument("--min-texture-std", type=float, default=12.0,
                    help="texture gate for random regions (always on; 0 disables)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if args.regions in ("nsfw", "mix") and not args.det_model:
        print("[pairs] ERROR: --det-model is required for --regions nsfw/mix "
              "(e.g. models\\lada_nsfw_detection_model_v1.3.pt)")
        return 2
    if args.regions in ("nsfw", "mix") and not os.path.isfile(args.det_model):
        print(f"[pairs] ERROR: detector not found: {args.det_model}")
        return 2

    rng = random.Random(args.seed)
    out = Path(args.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_exe()

    dev_str = "cpu" if str(args.device).lower() == "cpu" else f"cuda:{args.device}"
    if dev_str != "cpu" and not torch.cuda.is_available():
        print("[pairs] CUDA not available; running on CPU")
        dev_str = "cpu"

    detector = None
    if args.regions in ("nsfw", "mix"):
        from chitramaya.mosaic.detector.core import Detector
        detector = Detector(model_path=str(args.det_model), device=dev_str,
                            imgsz=int(args.det_imgsz), conf_thres=float(args.det_conf),
                            iou_thres=0.7, fp16=(dev_str != "cpu"))
        print(f"[pairs] detector: {Path(args.det_model).name} imgsz={args.det_imgsz} "
              f"conf={args.det_conf} device={dev_str}")

    # Disk estimate up front (PNG on natural video ~ 0.5 of raw).
    est_pairs = args.pairs_per_video * len(args.input)
    est_bytes = est_pairs * (args.clip_len * CLIP_SIZE * CLIP_SIZE * 3 * 2) * 0.5
    print(f"[pairs] recipe={RECIPE_VERSION} videos={len(args.input)} clip_len<={args.clip_len} "
          f"target<= {est_pairs} pairs; disk estimate ~{_fmt_bytes(est_bytes)} in {out}")
    print(f"[pairs] degradation: block {args.block_min_frac:.3f}-{args.block_max_frac:.3f} x H, "
          f"blur {args.blur_frac:.0%}, resize-roundtrip {args.resize_frac:.0%}, "
          f"h264 crf {args.crf_min}-{args.crf_max}{' (DISABLED)' if args.no_compress else ''}")

    # Split by SOURCE VIDEO (never by clip). With a single video we split by
    # window and say so -- a same-source val is a weaker exam.
    n_vid = len(args.input)
    by_video = n_vid >= 5
    n_val_vid = int(math.ceil(n_vid * args.val_fraction)) if by_video else 0
    val_videos = set(range(n_vid - n_val_vid, n_vid)) if n_val_vid > 0 else set()
    if not by_video:
        print(f"[pairs] NOTE: {n_vid} source video(s) -> val split is by WINDOW "
              "(same-source; a weaker exam than a held-out video; 5+ videos split by video)")

    manifest: dict = {
        "recipe": RECIPE_VERSION,
        "clip_size": CLIP_SIZE,
        "clip_len_max": int(args.clip_len),
        "regions": args.regions,
        "det_model": (Path(args.det_model).name if args.det_model else None),
        "det_imgsz": int(args.det_imgsz),
        "det_conf": float(args.det_conf),
        "degradation": {
            "block_frac": [args.block_min_frac, args.block_max_frac],
            "blur_frac": args.blur_frac,
            "resize_frac": args.resize_frac,
            "crf": [args.crf_min, args.crf_max],
            "compress": (not args.no_compress),
        },
        "seed": int(args.seed),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "LOCAL provenance; contains source file names; do not distribute.",
        "sources": [],
        "pairs": [],
    }

    pair_id = 0
    total_bytes = 0
    n_train = n_val = 0
    t_start = time.perf_counter()

    for vid_i, vid in enumerate(args.input):
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"[pairs] WARNING: cannot open {vid}; skipping")
            continue
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        manifest["sources"].append({"index": vid_i, "name": Path(vid).name,
                                    "frames": n_total, "fps": fps, "w": fw, "h": fh})
        if n_total <= args.window_frames:
            starts = [0]
        else:
            n_win = min(args.windows_per_video, max(1, n_total // args.window_frames))
            starts = sorted(rng.sample(range(0, n_total - args.window_frames), n_win))
        print(f"[pairs] video {vid_i + 1}/{n_vid}: {Path(vid).name} {fw}x{fh} "
              f"{n_total} frames @ {fps:.2f} fps; {len(starts)} windows x {args.window_frames} frames")

        pairs_this_video = 0
        for win_i, start in enumerate(starts):
            if pairs_this_video >= args.pairs_per_video:
                break
            if by_video:
                split = "val" if vid_i in val_videos else "train"
            else:
                split = "val" if rng.random() < args.val_fraction else "train"

            # Per-window "studio" choices: fixed block in frame pixels, style.
            use_blur = rng.random() < args.blur_frac
            # Block in FRAME pixels, scaled by frame height; the upper clamp
            # scales too (48 px on 1080p, ~97 px on 2160p) so 4K sources get
            # the big blocks real 4K studio mosaic has.
            block_px = int(max(4, min(max(48, round(fh * 0.045)),
                                      round(fh * rng.uniform(args.block_min_frac, args.block_max_frac)))))
            regions_mode = args.regions
            if regions_mode == "mix":
                regions_mode = "nsfw" if (win_i % 2 == 0) else "random"

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
            trk_gt = _make_tracker(args.clip_len)
            trk_lq = _make_tracker(args.clip_len)
            static_rois = None
            got_gt: list = []
            got_lq: list = []
            n_read = 0

            for k in range(args.window_frames):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                n_read += 1
                fn = start + k
                h, w = frame.shape[:2]
                gt_t = torch.from_numpy(np.ascontiguousarray(frame))  # HWC BGR uint8 (CPU)

                # --- regions for this frame ---
                if regions_mode == "nsfw":
                    det = detector.detect_batch([gt_t])[0]
                    boxes = _tensor_boxes_to_list_xyxy(det.boxes, w=w, h=h)
                    masks = _extract_masks_list(det) if det.masks is not None else None
                    boxes = [clip_box_to_bounds(b, w=w, h=h) for b in boxes]
                    if masks is not None and len(masks) != len(boxes):
                        masks = None
                else:
                    if static_rois is None:
                        static_rois = []
                        want = rng.randint(1, 2)
                        tries = 0
                        while len(static_rois) < want and tries < 16:
                            tries += 1
                            roi = _rand_region(rng, w, h)
                            if args.min_texture_std > 0 and not _textured_enough(frame, roi, args.min_texture_std):
                                continue
                            static_rois.append(roi)
                    boxes = list(static_rois)
                    masks = None

                # --- LQ frame: mosaic in the FULL frame, then the same crop ---
                lq_t = gt_t.clone()
                for roi in boxes:
                    if use_blur:
                        _blur_roi_inplace(lq_t, roi, sigma=max(1.5, block_px / 2.0))
                    else:
                        pixelate_roi_bgr_u8_inplace(lq_t, roi=roi, block=block_px)

                s_gt = trk_gt.step_frame(fn, gt_t, boxes, masks)
                s_lq = trk_lq.step_frame(fn, lq_t, boxes, masks)
                got_gt.extend(s_gt.new_clips)
                got_lq.extend(s_lq.new_clips)

            got_gt.extend(trk_gt.flush_eof())
            got_lq.extend(trk_lq.flush_eof())

            if len(got_gt) != len(got_lq):
                print(f"[pairs] WARNING: window {win_i} clip count mismatch "
                      f"gt={len(got_gt)} lq={len(got_lq)}; window dropped")
                continue

            n_win_pairs = 0
            for cg, cl in zip(got_gt, got_lq):
                if cg.frame_nums != cl.frame_nums or len(cg.frames) < args.min_clip_len:
                    continue
                gt_frames = [_to_np(f) for f in cg.frames]
                lq_frames = [_to_np(f) for f in cl.frames]
                mask_frames = [_to_np(m) for m in cg.masks]
                if any(f.shape[:2] != (CLIP_SIZE, CLIP_SIZE) for f in gt_frames):
                    continue

                deg = {"style": ("blur" if use_blur else "pixelate"), "block_px": block_px,
                       "resize": None, "crf": None}
                if rng.random() < args.resize_frac:
                    factor = rng.uniform(0.5, 0.85)
                    lq_frames = _resize_roundtrip(lq_frames, factor)
                    deg["resize"] = round(factor, 3)
                if not args.no_compress:
                    crf = rng.randint(args.crf_min, args.crf_max)
                    rt = _video_roundtrip_bgr(lq_frames, crf=crf, fps=int(round(fps)) or 30, ffmpeg=ffmpeg)
                    if rt is None:
                        print("[pairs] WARNING: h264 round-trip failed; keeping uncompressed LQ for this pair")
                    else:
                        lq_frames = rt
                        deg["crf"] = crf

                pid = f"{pair_id:06d}"
                pdir = out / "pairs" / pid
                pdir.mkdir(parents=True, exist_ok=True)
                gt_path, lq_path, mk_path = pdir / "gt.png", pdir / "lq.png", pdir / "mask.png"
                ok1 = cv2.imwrite(str(gt_path), _strip(gt_frames))
                ok2 = cv2.imwrite(str(lq_path), _strip(lq_frames))
                ok3 = cv2.imwrite(str(mk_path), _strip([m if m.ndim == 2 else m[:, :, 0] for m in mask_frames]))
                if not (ok1 and ok2 and ok3):
                    print(f"[pairs] WARNING: write failed for pair {pid}; skipped")
                    continue
                total_bytes += gt_path.stat().st_size + lq_path.stat().st_size + mk_path.stat().st_size
                manifest["pairs"].append({
                    "id": pid, "split": split, "source": vid_i,
                    "frame_start": int(cg.frame_nums[0]), "frames": int(len(cg.frames)),
                    "box_first": [int(v) for v in cg.boxes[0]],
                    "box_last": [int(v) for v in cg.boxes[-1]],
                    "regions": regions_mode, "degradation": deg,
                })
                if split == "val":
                    n_val += 1
                else:
                    n_train += 1
                pair_id += 1
                pairs_this_video += 1
                n_win_pairs += 1
                if pairs_this_video >= args.pairs_per_video:
                    break

            elapsed = time.perf_counter() - t_start
            print(f"[pairs] video {vid_i + 1}/{n_vid} window {win_i + 1}/{len(starts)} "
                  f"frames={n_read} clips={len(got_gt)} kept={n_win_pairs} "
                  f"pairs={pair_id} bytes={total_bytes} t={elapsed:.0f}s", flush=True)

        cap.release()

    manifest["counts"] = {"pairs": pair_id, "train": n_train, "val": n_val, "bytes": total_bytes}
    with open(out / "pairs.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    print(f"[pairs] done: pairs={pair_id} train={n_train} val={n_val} "
          f"size={_fmt_bytes(total_bytes)} manifest={out / 'pairs.json'}")
    if pair_id == 0:
        print("[pairs] WARNING: no pairs written. With --regions nsfw, the detector found no "
              "regions in the sampled windows (try --det-conf 0.1, more windows, or another video).")
        return 1
    print(f"[pairs] next: ChitraMaya -train-rest --pairs {out} --base models\\<restoration>.pth --out <run folder>")
    return 0


if __name__ == "__main__":
    sys.exit(main())

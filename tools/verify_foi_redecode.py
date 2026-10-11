"""verify_foi_redecode.py -- Test Frame (FOI capture) through the REDECODE path
(CM-244 / T27b). CPU, no models: a 90-frame synthetic clip, pseudo restorer
(flat fill), two fixed synthetic ROIs, FOI target frame 45. Checks that the
re-decode drain captures the target's original + composited frame and that
the composite is bit-identical to the frame-store path's.

    PYTHONPATH=. python tools/verify_foi_redecode.py
Needs ffmpeg on PATH (the clip is generated with lavfi testsrc2).
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main() -> int:
    import torch
    from chitramaya.mosaic.pipeline import MosaicPipelineConfig, MosaicPipeline, Pipeline

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.mp4")
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc2=size=640x360:rate=30", "-frames:v", "90", "-c:v", "libx264", "-g", "30",
                            "-pix_fmt", "yuv420p", "-y", src], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[verify-foi-redecode] SKIP: ffmpeg could not make the clip ({r.stderr.strip()[:120]})")
            return 0

        import io, contextlib

        def run(store):
            cfg = MosaicPipelineConfig(detection_model="det.pt", restoration_model="rest.pth", mask_preview=True,
                                       store_backend=store, use_trt=False, codec="h264", residual_scan=False)
            mp = MosaicPipeline.__new__(MosaicPipeline)
            mp.config = cfg
            mp.gpu_id = 0
            base = mp._build_base_config(src, os.path.join(d, f"out_{store}.mp4"))
            base.data["restoration"]["analysis_use_synth_rois"] = True
            base.data["synth_mosaic"] = {"rois": [[100, 200, 260, 420], [40, 30, 120, 150]]}
            host = Pipeline(base)
            foi = {"target_frame": 45}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                host.run(foi_capture=foi)
            return buf.getvalue(), foi

        log_r, a = run("redecode")
        _, b = run("device")

    checks = {
        "redecode path taken for a Test Frame run (console says REDECODE)": ("backend: REDECODE" in log_r, True),
        "no 'Test Frame runs keep the frame store' note": ("keep the frame store" in log_r, False),
        "FOI found on the redecode path": (bool(a.get("found")), True),
        "FOI original captured (redecode)": (a.get("original") is not None, True),
        "FOI composited captured (redecode)": (a.get("composited") is not None, True),
        "composite differs from original inside the ROI (a paste happened)": (
            a.get("composited") is not None and not torch.equal(a["composited"], a["original"]), True),
        "composite identical to the frame-store path": (
            a.get("composited") is not None and b.get("composited") is not None
            and torch.equal(a["composited"], b["composited"]), True),
    }
    failed = [k for k, (got, want) in checks.items() if got != want]
    for k, (got, want) in checks.items():
        print(f"[{'OK ' if got == want else 'BAD'}] {k}: got={got} want={want}")
    if failed:
        print(f"\n[verify-foi-redecode] FAIL: {failed}")
        return 1
    print("\n[verify-foi-redecode] PASS -- Test Frame captures through REDECODE, bit-identical to the store path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

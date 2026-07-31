"""ChitraMaya — TensorRT-based face swapping pipeline.

Pipeline: Decode → Detect → Swap → Encode

Ported from Rope's schemes with:
  - pyNVVideoCodec for decode/encode (from gRestorer)
  - TensorRT for model inference (replacing ONNX Runtime)
  - PyWebView + Flask GUI (from Tilester)
  - Tilester-pattern config/models/pipeline architecture
"""

# Single source of truth for the app version. BUMP THIS ONE LINE before each
# release (scheme: major.MINOR.patch, e.g. 1.10.00 = SBS+AddMosaic, 1.11.00 =
# next feature, 1.10.01 = patch-only). The HF User-Agent and the window title
# both read it, so a release is a one-line change here.
#
# Batch 31: the packager now ENFORCES the bump -- it reads this line, prints
# the version in the build banner, and refuses to build if the version is
# already listed in packaging/windows/released-versions.txt (append to that
# file right after each publish). v1.20 and v1.30 both shipped showing a
# stale number because this line relied on memory; no longer.
__version__ = "1.40.00"

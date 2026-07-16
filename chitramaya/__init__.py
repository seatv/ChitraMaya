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
__version__ = "1.20.00"
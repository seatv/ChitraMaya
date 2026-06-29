"""ChitraMaya — TensorRT-based face swapping pipeline.

Pipeline: Decode → Detect → Swap → Encode

Ported from Rope's schemes with:
  - pyNVVideoCodec for decode/encode (from gRestorer)
  - TensorRT for model inference (replacing ONNX Runtime)
  - PyWebView + Flask GUI (from Tilester)
  - Tilester-pattern config/models/pipeline architecture
"""

__version__ = "0.1.0"

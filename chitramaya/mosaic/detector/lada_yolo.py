# gRestorer/detector/lada_yolo.py
# Ported 1:1 from LADA v0.10.1 (Yolo11SegmentationModel + PyTorchLetterBox + mask/box conversion)
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import ultralytics
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, nms, ops
from ultralytics.utils.checks import check_imgsz

# LADA v0.10.1 was developed against ultralytics==8.4.4. We do not enforce
# this pin — newer versions are generally compatible. If detection behavior
# diverges from Lada's reference, the ultralytics version is the first place
# to investigate.
_LADA_ULTRALYTICS_PIN = "8.4.4"
_installed_ultralytics = getattr(ultralytics, "__version__", "unknown")
if _installed_ultralytics != _LADA_ULTRALYTICS_PIN:
    print(
        f"[LadaYoloDetector] note: tested against ultralytics=={_LADA_ULTRALYTICS_PIN}, "
        f"running on {_installed_ultralytics}."
    )

# ---- gRestorer-style detection object (shape-compatible with pipeline expectations) ----

@dataclass
class Detection:
    boxes: Optional[torch.Tensor]   # [N,4] float32 xyxy in original coords (CPU)
    scores: Optional[torch.Tensor]  # [N]   float32 (CPU)
    classes: Optional[torch.Tensor] # [N]   int64   (CPU)
    masks: Optional[torch.Tensor]   # [N,H,W] uint8 (CPU), values 0/255


# ---- LADA: torch_letterbox.PyTorchLetterBox (verbatim) ----

# NOTE: This requires torchvision transforms.v2 in your environment (same as LADA).
from torchvision.transforms.v2 import Resize, Pad
from torchvision.transforms.v2.functional import InterpolationMode


class PyTorchLetterBox:
    def __init__(self, imgsz: int | tuple[int, int], original_shape: tuple[int, int], stride: int = 32) -> None:
        if isinstance(imgsz, int):
            new_shape: tuple[int, int] = (imgsz, imgsz)
        else:
            new_shape = imgsz

        self.original_shape = original_shape
        pad_value: float = 114.0 / 255.0
        h, w = original_shape
        new_h, new_w = new_shape

        r = min(new_h / h, new_w / w)
        new_unpad_w = int(round(w * r))
        new_unpad_h = int(round(h * r))

        dw = new_w - new_unpad_w
        dh = new_h - new_unpad_h
        dw = int(dw % stride)
        dh = int(dh % stride)

        resize = None if (h, w) == (new_unpad_h, new_unpad_w) else Resize(
            size=(new_unpad_h, new_unpad_w),
            interpolation=InterpolationMode.BILINEAR,
            antialias=False,
        )
        pad = Pad(
            padding=(dw // 2, dh // 2, dw - (dw // 2), dh - (dh // 2)),
            fill=pad_value,
        )
        self.transform = torch.nn.Sequential(resize, pad) if resize is not None else pad

    def __call__(self, image: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        return self.transform(image)


# ---- LADA: ultralytics_utils conversion (verbatim relevant bits) ----

def convert_yolo_box(yolo_box, img_shape) -> Tuple[int, int, int, int]:
    _box = yolo_box.xyxy[0]
    l = int(torch.clip(_box[0], 0, img_shape[1]).item())
    t = int(torch.clip(_box[1], 0, img_shape[0]).item())
    r = int(torch.clip(_box[2], 0, img_shape[1]).item())
    b = int(torch.clip(_box[3], 0, img_shape[0]).item())
    return t, l, b, r

def _to_mask_img_tensor(masks: torch.Tensor, class_val=0, pixel_val=255) -> torch.Tensor:
    masks_tensor = torch.where(masks != class_val, pixel_val, 0).to(torch.uint8)
    return masks_tensor[0]

def scale_and_unpad_image(masks: torch.Tensor, im0_shape) -> torch.Tensor:
    h0, w0 = im0_shape[:2]
    h1, w1, _ = masks.shape
    if h1 == h0 and w1 == w0:
        return masks
    g = min(h1 / h0, w1 / w0)
    pw, ph = (w1 - w0 * g) / 2, (h1 - h0 * g) / 2
    t, l = round(ph - 0.1), round(pw - 0.1)
    b, r = h1 - round(ph + 0.1), w1 - round(pw + 0.1)
    x = masks[t:b, l:r].permute(2, 0, 1).unsqueeze(0).float()
    y = F.interpolate(x, size=(h0, w0), mode="bilinear", align_corners=False)
    return y.squeeze(0).permute(1, 2, 0).round_().clamp_(0, 255).to(masks.dtype)

def convert_yolo_mask_tensor(yolo_mask, img_shape) -> torch.Tensor:
    mask_img = _to_mask_img_tensor(yolo_mask.data)
    if mask_img.ndim == 2:
        mask_img = mask_img.unsqueeze(-1)
    mask_img = scale_and_unpad_image(mask_img, img_shape)
    mask_img = torch.where(mask_img > 127, 255, 0).to(torch.uint8)
    assert mask_img.ndim == 3 and mask_img.shape[2] == 1
    return mask_img


# ---- LADA: Yolo11SegmentationModel (verbatim with only removed unused LADA imports) ----

class Yolo11SegmentationModel:
    def __init__(self, model_path: str, device, imgsz=640, fp16=False, **kwargs):
        yolo_model = YOLO(model_path)
        assert yolo_model.task == "segment"
        self.stride = 32
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox: PyTorchLetterBox | LetterBox = LetterBox(self.imgsz, auto=True, stride=self.stride)

        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "device": device, "half": fp16}
        args = {**yolo_model.overrides, **custom, **kwargs}  # highest priority args on the right
        self.args = get_cfg(DEFAULT_CFG, args)

        self.device: torch.device = torch.device(device)
        self.is_cuda_device: bool = self.device.type == "cuda"
        self.model = AutoBackend(
            model=yolo_model.model,
            device=self.device,
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=False,
        )
        self.args.half = self.model.fp16
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        # Use AutoBackend's detected fp16, not the original constructor param.
        # For .pt + fp16=True, AutoBackend casts the model to half and sets
        # self.model.fp16 = True — dtype = float16 below, matches.
        # For .engine, AutoBackend reads the engine's input binding dtype:
        # ultralytics' fp16 engines have FLOAT (fp32) input bindings even
        # though weights are fp16, so AutoBackend sets self.model.fp16 = False.
        # We must match that — feeding fp16 to an FLOAT-input engine produces
        # silent garbage (no error, but near-zero detections).
        self.dtype = torch.float16 if self.model.fp16 else torch.float32

    def _preprocess_cpu(self, imgs: list[torch.Tensor]) -> torch.Tensor:
        im = np.stack([self.letterbox(image=x.numpy()) for x in imgs])
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        return torch.from_numpy(im)

    def _preprocess_gpu(self, imgs: list[torch.Tensor]) -> torch.Tensor:
        return self.letterbox(torch.stack(imgs, dim=0))

    def preprocess(self, imgs: list[torch.Tensor]) -> torch.Tensor:
        is_cpu_input = imgs[0].device.type == "cpu"
        if is_cpu_input:
            return self._preprocess_cpu(imgs)
        else:
            if self.letterbox is None or imgs[0].shape[:2] != self.letterbox.original_shape:
                self.letterbox = PyTorchLetterBox(self.imgsz, imgs[0].shape[:2], stride=self.stride)
            return self._preprocess_gpu(imgs)

    def inference(self, image_batch: torch.Tensor):
        return self.model(image_batch, augment=False, visualize=False, embed=None)

    def inference_and_postprocess(self, imgs: torch.Tensor, orig_imgs: list[torch.Tensor]):
        with torch.inference_mode():
            input = imgs.to(device=self.device).to(dtype=self.dtype).div_(255.0)
            preds = self.inference(input)
            return self.postprocess(preds, input, orig_imgs)

    def postprocess(self, preds, img: torch.Tensor, orig_imgs: list[torch.Tensor]):
        protos = preds[0][-1]
        # LADA patches ultralytics' NMS time limit from 0.05s to 0.3s per image
        # to keep dense 4K frames from exceeding the timeout and silently dropping
        # detections. We pass it explicitly here to avoid modifying the installed
        # ultralytics package; same effect, version-independent.
        preds = nms.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            max_time_img=0.3,
        )
        return [self.construct_result(pred, img, orig_img, proto) for pred, orig_img, proto in zip(preds, orig_imgs, protos)]

    def construct_result(self, preds: torch.Tensor, img: torch.Tensor, orig_img: torch.Tensor, proto: torch.Tensor):
        if not len(preds):  # save empty boxes
            masks = None
        else:
            masks = ops.process_mask(proto, preds[:, 6:], preds[:, :4], img.shape[2:], upsample=True)  # HWC
            preds[:, :4] = ops.scale_boxes(img.shape[2:], preds[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            preds, masks = preds[keep], masks[keep]
        # Return a minimal ultralytics Results-like object using ultralytics Results container
        from ultralytics.engine.results import Results
        return Results(orig_img, path="", names=self.model.names, boxes=preds[:, :6].cpu(), masks=masks)


# ---- gRestorer wrapper: LadaYoloDetector ----

class LadaYoloDetector:
    """
    B = LADA-exact detector stack for A/B.

    Input frames MUST be:
      - BGR uint8 [H,W,3]
      - (CPU or CUDA) – if on CUDA, we intentionally move to CPU for preprocessing to match LADA’s common path.
    """

    def __init__(
        self,
        model_path: str,
        device: str | torch.device = "cuda:0",
        imgsz: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[Sequence[int]] = None,
        fp16: bool = True,
    ) -> None:
        self.device = str(device)
        self.model = Yolo11SegmentationModel(
            model_path=model_path,
            device=self.device,
            imgsz=imgsz,
            fp16=fp16,
            conf=conf_thres,
            iou=iou_thres,
            classes=classes,
        )
        print("[Detector] Running Lada-Yolo (LADA-exact port)")

    def detect_batch(self, frames: List[torch.Tensor]) -> List[Detection]:
        if not frames:
            return []

        # LADA’s typical pipeline decodes on CPU and preprocesses on CPU.
        # For parity, we run preprocess on CPU even if frames came from GPU decode.
        cpu_frames: List[torch.Tensor] = []
        for f in frames:
            if f.dtype != torch.uint8 or f.ndim != 3 or f.shape[-1] != 3:
                raise ValueError("LadaYoloDetector expects BGR uint8 HWC frames.")
            cpu_frames.append(f.detach().to("cpu").contiguous() if f.device.type != "cpu" else f.contiguous())

        imgs = self.model.preprocess(cpu_frames)
        results_list = self.model.inference_and_postprocess(imgs, cpu_frames)

        out: List[Detection] = []
        for res in results_list:
            if res.boxes is None or len(res.boxes) == 0:
                out.append(Detection(None, None, None, None))
                continue

            # Boxes + scores + classes (CPU)
            boxes_tlbr: List[Tuple[int, int, int, int]] = [convert_yolo_box(res.boxes[i], res.orig_shape) for i in range(len(res.boxes))]
            # gRestorer expects xyxy (x1,y1,x2,y2)
            boxes_xyxy = torch.tensor([[l, t, r, b] for (t, l, b, r) in boxes_tlbr], dtype=torch.float32, device="cpu")
            scores = res.boxes.conf.detach().to("cpu").float() if hasattr(res.boxes, "conf") else None
            classes = res.boxes.cls.detach().to("cpu").to(torch.int64) if hasattr(res.boxes, "cls") else None

            # Masks (CPU) – match LADA’s conversion used in restoration pipeline
            masks_out: Optional[torch.Tensor] = None
            if res.masks is not None and len(res.masks) == len(res.boxes):
                masks_list: List[torch.Tensor] = []
                for i in range(len(res.boxes)):
                    m = convert_yolo_mask_tensor(res.masks[i], res.orig_shape).to(device=res.orig_img.device)  # (H,W,1) uint8
                    masks_list.append(m.squeeze(-1).to("cpu"))
                if masks_list:
                    masks_out = torch.stack(masks_list, dim=0)

            out.append(Detection(boxes_xyxy, scores, classes, masks_out))

        return out


__all__ = ["LadaYoloDetector", "Detection"]

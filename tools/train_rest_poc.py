# tools/train_rest_poc.py
r"""CM-112 Phase C (Batch T3): fine-tune the BasicVSR++ mosaic restorer on
perfect-pair clips from make_rest_pairs.py.

    ChitraMaya -train-rest --pairs D:\Train\pairs ^
        --base models\lada_mosaic_restoration_model_generic_v1.2.pth ^
        --out D:\Train\runs\rest-v0

Design (parity with inference is the whole game):
  - The network is the app's own vendored BasicVSR++ generator (64 ch, 15
    blocks, SPyNet inside), same-size output, fed BGR uint8/255 clips of
    T x 256 x 256 exactly as basicvsrpp_clip_restorer does. The net works at
    quarter resolution internally (feat_extract stride 4), so 256x256 clips
    are cheap: T=15 fits a 12 GB card with room to spare. Clips must be
    >= 256 px (the net asserts >= 64 after its 0.25x flow downsample).
  - Fine-tune, never from scratch: --base is lada's generic .pth (AGPL --
    the fine-tune is a derivative; share it under AGPL with credit, per the
    Giants doctrine; keep "lada" in the output name).
  - Loss v0: Charbonnier (robust L1), uniform over the crop, optional extra
    weight on the mosaic region (--mask-weight). Known risk: L1 fine-tuning
    of a GAN-stage generator drifts toward the mean (blur). Mitigations:
    low LR (default 2e-5), frequent checkpoints, judge early by eye.
  - SPyNet frozen for the first --spynet-freeze-iters, then trained at
    lr * --spynet-lr-mult (mmagic practice).
  - Honest yardsticks printed before training: val PSNR of the UNTOUCHED
    base model and of the raw LQ input. If fine-tuning does not beat the
    base on your own val pairs, it did not help -- say so.
  - Output: <out>/weights/latest.pth (resumable) and best.pth (by val PSNR)
    in the SAME checkpoint shape the app loads ({"state_dict": {"generator.*"}}).
    Copy best.pth into models\ and compile it from Manage Models untouched.

Machine-parsable lines (B82 shape; the UI parses these):
  [train-rest] start iters=N train_pairs=A val_pairs=B
  [train-rest] iter i/N loss=0.0123 lr=2.0e-05 it/s=1.20 eta=1234s vram=3.1GB
  [train-rest] val iter=i psnr=31.25 base=30.90 lq=27.10 best=31.25
  [train-rest] done best_psnr=... best_iter=... weights=<path>

Doctrine: restoration-only, supervised perfect-pair, no generative priors.
Datasets and runs are local artifacts; weights travel, content never does.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from chitramaya.mosaic.models.basicvsrpp.grestorer.basicvsr_plusplus_net import BasicVSRPlusPlusNet  # noqa: E402
from chitramaya.mosaic.models.basicvsrpp.inference import (  # noqa: E402
    _load_checkpoint_state_dict, _strip_known_prefixes, get_default_gan_inference_config,
)

CLIP_SIZE = 256


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _read_strip(path: Path, frames: int, gray: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    h = img.shape[0] // frames
    if gray:
        return img.reshape(frames, h, img.shape[1])
    return img.reshape(frames, h, img.shape[1], 3)


class PairsDataset(Dataset):
    """One item = (lq TCHW float, gt TCHW float, mask T1HW float), BGR 0..1."""

    def __init__(self, root: Path, entries: list, clip_len: int, train: bool, seed: int = 0):
        self.root = Path(root)
        self.entries = entries
        self.clip_len = int(clip_len)
        self.train = bool(train)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        d = self.root / "pairs" / e["id"]
        n = int(e["frames"])
        gt = _read_strip(d / "gt.png", n)
        lq = _read_strip(d / "lq.png", n)
        mk = _read_strip(d / "mask.png", n, gray=True)
        T = min(self.clip_len, n)
        s = self.rng.randint(0, n - T) if self.train else 0
        gt, lq, mk = gt[s:s + T], lq[s:s + T], mk[s:s + T]
        if self.train:
            if self.rng.random() < 0.5:                 # horizontal flip
                gt, lq, mk = gt[:, :, ::-1], lq[:, :, ::-1], mk[:, :, ::-1]
            if self.rng.random() < 0.5:                 # temporal reverse (still a video)
                gt, lq, mk = gt[::-1], lq[::-1], mk[::-1]
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).permute(0, 3, 1, 2).float().div_(255.0)
        lq_t = torch.from_numpy(np.ascontiguousarray(lq)).permute(0, 3, 1, 2).float().div_(255.0)
        mk_t = torch.from_numpy(np.ascontiguousarray(mk)).unsqueeze(1).float().div_(255.0)
        return lq_t, gt_t, mk_t


def _collate(batch):
    # Clips in a batch may differ in T (short pairs); trim to the shortest.
    T = min(b[0].shape[0] for b in batch)
    lq = torch.stack([b[0][:T] for b in batch])
    gt = torch.stack([b[1][:T] for b in batch])
    mk = torch.stack([b[2][:T] for b in batch])
    return lq, gt, mk


# ---------------------------------------------------------------------------
# Model / loss / metrics
# ---------------------------------------------------------------------------

def _build_generator() -> BasicVSRPlusPlusNet:
    g = get_default_gan_inference_config()["generator"]
    return BasicVSRPlusPlusNet(
        mid_channels=int(g["mid_channels"]), num_blocks=int(g["num_blocks"]),
        max_residue_magnitude=int(g["max_residue_magnitude"]), spynet_pretrained=None,
    )


def _load_base(gen: torch.nn.Module, path: str) -> None:
    sd = _strip_known_prefixes(_load_checkpoint_state_dict(path))
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    print(f"[train-rest] base loaded: {Path(path).name} tensors={len(sd)} "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[train-rest] WARNING: {len(missing)} missing keys, e.g. {missing[:3]} -- "
              "is --base a BasicVSR++ restoration checkpoint?")


def _save_app_checkpoint(gen: torch.nn.Module, path: Path, meta: dict) -> None:
    """Same shape as lada checkpoints: {'state_dict': {'generator.<k>': v}}
    -> inference.load_model strips the prefix; trt_export follows the same path."""
    sd = {f"generator.{k}": v.detach().cpu() for k, v in gen.state_dict().items()}
    torch.save({"state_dict": sd, "meta": meta}, str(path))


def charbonnier(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None, mask_weight: float,
                eps: float = 1e-6) -> torch.Tensor:
    diff = torch.sqrt((pred - gt) ** 2 + eps)
    if mask is not None and mask_weight != 1.0:
        w = 1.0 + (float(mask_weight) - 1.0) * mask          # B,T,1,H,W
        diff = diff * w
    return diff.mean()


@torch.no_grad()
def psnr_btchw(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a.float().clamp(0, 1), b.float().clamp(0, 1)).item()
    return 99.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse))


@torch.no_grad()
def evaluate(gen: torch.nn.Module, loader: DataLoader, device: torch.device, amp: bool,
             max_batches: int, model_fn=None) -> tuple[float, float]:
    """Returns (mean PSNR of model output, mean PSNR of raw LQ) over val."""
    gen.eval()
    ps, pl, n = 0.0, 0.0, 0
    for bi, (lq, gt, _) in enumerate(loader):
        if bi >= max_batches:
            break
        lq, gt = lq.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = gen(lq) if model_fn is None else model_fn(lq)
        ps += psnr_btchw(out, gt)
        pl += psnr_btchw(lq, gt)
        n += 1
    gen.train()
    return (ps / max(1, n), pl / max(1, n))


def _vram_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="CM-112 Phase C: fine-tune the BasicVSR++ restorer")
    ap.add_argument("--pairs", required=True, help="folder from make_rest_pairs.py (has pairs.json)")
    ap.add_argument("--base", required=True, help="starting checkpoint (.pth), e.g. lada generic v1.2")
    ap.add_argument("--out", required=True, help="run folder (user-chosen; weights/ lands here)")
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--clip-len", type=int, default=15, help="frames per training sample")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--spynet-freeze-iters", type=int, default=1000)
    ap.add_argument("--spynet-lr-mult", type=float, default=0.25)
    ap.add_argument("--mask-weight", type=float, default=1.0,
                    help="loss weight on the mosaic region (1.0 = uniform)")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-batches", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="0", help="CUDA device id, or 'cpu'")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast (CUDA)")
    ap.add_argument("--resume", default=None, help="latest.pth from a previous run")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dev = torch.device("cpu" if str(args.device).lower() == "cpu" else f"cuda:{args.device}")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[train-rest] CUDA not available; running on CPU (slow -- mechanism only)")
        dev = torch.device("cpu")
    amp = (dev.type == "cuda") and not args.no_amp

    pairs_root = Path(args.pairs)
    manifest_path = pairs_root / "pairs.json"
    if not manifest_path.is_file():
        print(f"[train-rest] ERROR: no pairs.json in {pairs_root}")
        return 2
    if not os.path.isfile(args.base):
        print(f"[train-rest] ERROR: base checkpoint not found: {args.base}")
        return 2
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    entries = [e for e in manifest.get("pairs", []) if int(e.get("frames", 0)) >= 2]
    train_e = [e for e in entries if e.get("split") == "train"]
    val_e = [e for e in entries if e.get("split") == "val"]
    if not train_e:
        print("[train-rest] ERROR: no training pairs in manifest")
        return 2
    if not val_e:
        # Never train blind: hold out a slice of train as val and say so.
        k = max(1, len(train_e) // 10)
        val_e, train_e = train_e[-k:], train_e[:-k]
        print(f"[train-rest] NOTE: manifest had no val pairs; holding out {k} train pairs as val")

    run_dir = Path(args.out)
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)

    train_ds = PairsDataset(pairs_root, train_e, args.clip_len, train=True, seed=args.seed)
    val_ds = PairsDataset(pairs_root, val_e, args.clip_len, train=False)
    pin = dev.type == "cuda"
    train_ld = DataLoader(train_ds, batch_size=int(args.batch), shuffle=True, num_workers=int(args.workers),
                          collate_fn=_collate, pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=_collate)

    gen = _build_generator()
    _load_base(gen, args.base)
    gen.to(dev).train()

    spynet_params = [p for n, p in gen.named_parameters() if n.startswith("spynet.")]
    other_params = [p for n, p in gen.named_parameters() if not n.startswith("spynet.")]
    opt = torch.optim.Adam([
        {"params": other_params, "lr": float(args.lr)},
        {"params": spynet_params, "lr": float(args.lr) * float(args.spynet_lr_mult)},
    ], betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch, "amp") else torch.cuda.amp.GradScaler(enabled=amp)

    start_iter = 0
    best_psnr = -1.0
    best_iter = -1
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        gen.load_state_dict(_strip_known_prefixes(ck["state_dict"]), strict=True)
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_iter = int(ck.get("iter", 0))
        best_psnr = float(ck.get("best_psnr", -1.0))
        best_iter = int(ck.get("best_iter", -1))
        print(f"[train-rest] resumed from {args.resume} at iter {start_iter} (best {best_psnr:.2f} @ {best_iter})")

    print(f"[train-rest] device={dev} amp={amp} batch={args.batch} clip_len={args.clip_len} "
          f"lr={args.lr:.1e} spynet: frozen<{args.spynet_freeze_iters} then x{args.spynet_lr_mult} "
          f"mask_weight={args.mask_weight}")
    print(f"[train-rest] start iters={args.iters} train_pairs={len(train_e)} val_pairs={len(val_e)}")

    # Honest yardsticks BEFORE training: the untouched base, and the raw LQ.
    base_psnr, lq_psnr = evaluate(gen, val_ld, dev, amp, args.val_batches)
    print(f"[train-rest] val iter={start_iter} psnr={base_psnr:.2f} base={base_psnr:.2f} lq={lq_psnr:.2f} "
          f"best={max(best_psnr, base_psnr):.2f}  (base model, before any training)")
    if best_psnr < 0:
        best_psnr, best_iter = base_psnr, start_iter

    meta_common = {
        "phase": "CM-112 Phase C fine-tune", "base": Path(args.base).name,
        "pairs_recipe": manifest.get("recipe"), "train_pairs": len(train_e), "val_pairs": len(val_e),
        "lr": args.lr, "clip_len": args.clip_len, "batch": args.batch, "mask_weight": args.mask_weight,
        "license_note": "Derivative of an AGPL checkpoint if --base was lada's; share under AGPL with credit.",
    }

    def _set_spynet(trainable: bool) -> None:
        for p in spynet_params:
            p.requires_grad_(bool(trainable))

    _set_spynet(start_iter >= args.spynet_freeze_iters)
    spynet_on = start_iter >= args.spynet_freeze_iters

    it = start_iter
    t0 = time.perf_counter()
    t_log = t0
    loss_acc, n_acc = 0.0, 0
    data_iter = iter(train_ld)
    while it < args.iters:
        try:
            lq, gt, mk = next(data_iter)
        except StopIteration:
            data_iter = iter(train_ld)
            lq, gt, mk = next(data_iter)
        if not spynet_on and it >= args.spynet_freeze_iters:
            _set_spynet(True)
            spynet_on = True
            print(f"[train-rest] iter {it}: SPyNet unfrozen (lr x{args.spynet_lr_mult})")

        lq = lq.to(dev, non_blocking=True)
        gt = gt.to(dev, non_blocking=True)
        mk = mk.to(dev, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            out = gen(lq)
            loss = charbonnier(out.float(), gt, mk, args.mask_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        it += 1
        loss_acc += float(loss.item())
        n_acc += 1

        if it % args.log_every == 0 or it == args.iters:
            now = time.perf_counter()
            ips = args.log_every / max(1e-6, (now - t_log))
            eta = (args.iters - it) / max(1e-6, ips)
            print(f"[train-rest] iter {it}/{args.iters} loss={loss_acc / max(1, n_acc):.5f} "
                  f"lr={opt.param_groups[0]['lr']:.1e} it/s={ips:.2f} eta={eta:.0f}s vram={_vram_gb(dev):.1f}GB",
                  flush=True)
            t_log = now
            loss_acc, n_acc = 0.0, 0

        if it % args.val_every == 0 or it == args.iters:
            v_psnr, _ = evaluate(gen, val_ld, dev, amp, args.val_batches)
            improved = v_psnr > best_psnr
            if improved:
                best_psnr, best_iter = v_psnr, it
                _save_app_checkpoint(gen, run_dir / "weights" / "best.pth",
                                     {**meta_common, "iter": it, "val_psnr": v_psnr})
            print(f"[train-rest] val iter={it} psnr={v_psnr:.2f} base={base_psnr:.2f} lq={lq_psnr:.2f} "
                  f"best={best_psnr:.2f}{' *' if improved else ''}", flush=True)

        if it % args.save_every == 0 or it == args.iters:
            sd = {f"generator.{k}": v.detach().cpu() for k, v in gen.state_dict().items()}
            torch.save({"state_dict": sd, "optimizer": opt.state_dict(), "iter": it,
                        "best_psnr": best_psnr, "best_iter": best_iter, "meta": meta_common},
                       str(run_dir / "weights" / "latest.pth"))

    elapsed = time.perf_counter() - t0
    best_path = run_dir / "weights" / "best.pth"
    if not best_path.is_file():
        _save_app_checkpoint(gen, best_path, {**meta_common, "iter": it, "val_psnr": best_psnr})
    print(f"[train-rest] done best_psnr={best_psnr:.2f} best_iter={best_iter} base={base_psnr:.2f} "
          f"lq={lq_psnr:.2f} elapsed={elapsed:.0f}s weights={best_path}")
    if best_iter <= start_iter:
        print("[train-rest] RESULT: fine-tuning did NOT beat the base model on your val pairs "
              "(best is the untouched base). More/other data, lower LR, or more iterations before trusting it.")
    else:
        print(f"[train-rest] RESULT: +{best_psnr - base_psnr:.2f} dB over the base on val pairs "
              f"(synthetic exam; the eye and the headset decide).")
    print(f"[train-rest] next: copy {best_path.name} into models\\ as lada_mosaic_restoration_<yourname>.pth, "
          "then Manage Models -> compile. Keep 'lada' in the name (AGPL derivative).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Does PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold do anything on this
torch build without a per-process memory fraction? Five-second probe (CM-238).

    python tools/gc_threshold_probe.py              # threshold only -- what RC3 sets
    python tools/gc_threshold_probe.py --fraction   # plus set_per_process_memory_fraction(1.0)
    python tools/gc_threshold_probe.py --fraction --runtime
                                                    # plus torch.cuda.memory._set_allocator_settings(...)
                                                    # (bypasses the env-var parse)

The probe fills ~70 % of the free VRAM with 64 MB blocks, drops them (torch keeps
them cached: 'reserved' stays), then asks for ONE 256 MB block. A 256 MB request
cannot be carved from a 64 MB cached block, so the allocator has to go to
cudaMalloc -- and that is the moment the garbage-collection threshold (0.5 here,
well below the ~60 % of the card we just cached) is consulted:

  GC live  -> cached blocks are handed back first: reserved AFTER < reserved BEFORE
  GC inert -> reserved AFTER = reserved BEFORE + 256 MB

Run it both ways on the same card and compare the two 'verdict' lines.

Result on record (2026-10-01, RTX 4060, torch 2.11.0+cu128 -- the wheel the app
ships): INERT in every variant. The allocator reported gc_threshold=0.5 as
parsed, a 1.0 fraction was set, 77 x 64 MB = 4928 MB (60 %) sat in cache, the
256 MB request went to a fresh cudaMalloc and `cudaFree calls` stayed at 0.
That is why T24 removed the threshold from the app (CM-238 keeps only the torch
allocated/reserved split on the checkpoint line). Re-run this after a torch
upgrade before considering the knob again.
"""
import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.5"

import torch  # noqa: E402  (after the env var -- the allocator reads it once)

MB = 1024 * 1024


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    dev = torch.device("cuda:0")
    torch.cuda.init()
    with_fraction = "--fraction" in sys.argv
    if with_fraction:
        torch.cuda.set_per_process_memory_fraction(1.0, dev)
    if "--runtime" in sys.argv:
        # Bypass the env-var parse: hand the setting to the allocator directly.
        torch.cuda.memory._set_allocator_settings("garbage_collection_threshold:0.5")
    free, total = torch.cuda.mem_get_info(dev)
    print(f"[probe] {torch.cuda.get_device_name(dev)}  torch {torch.__version__}  "
          f"free {free // MB} / total {total // MB} MB  "
          f"PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}  "
          f"set_per_process_memory_fraction(1.0)={'yes' if with_fraction else 'no'}  "
          f"--runtime={'yes' if '--runtime' in sys.argv else 'no'}")
    # What did the allocator actually parse? (AllocatorConfigInfo, torch >= 2.1)
    try:
        _settings = torch.cuda.memory._snapshot()["allocator_settings"]
        print(f"[probe] allocator_settings as the allocator sees them: "
              f"gc_threshold={_settings.get('garbage_collection_threshold')}  "
              f"max_split_size={_settings.get('max_split_size')}  "
              f"expandable_segments={_settings.get('expandable_segments')}  "
              f"last_allocator_settings={_settings.get('last_allocator_settings')!r}")
    except Exception as e:  # pragma: no cover - diagnostic only
        print(f"[probe] allocator_settings unavailable ({type(e).__name__}: {e})")
    n = int(free * 0.7) // (64 * MB)
    blocks = [torch.empty(64 * MB, dtype=torch.uint8, device=dev) for _ in range(n)]
    torch.cuda.synchronize(dev)
    r_full = torch.cuda.memory_reserved(dev)
    del blocks
    r_cached = torch.cuda.memory_reserved(dev)
    a_cached = torch.cuda.memory_allocated(dev)
    print(f"[probe] filled {n} x 64 MB -> reserved {r_full // MB} MB; after del: "
          f"allocated {a_cached // MB} / reserved {r_cached // MB} MB "
          f"({100.0 * r_cached / total:.0f} % of the card cached, threshold 50 %)")
    big = torch.empty(256 * MB, dtype=torch.uint8, device=dev)
    torch.cuda.synchronize(dev)
    r_after = torch.cuda.memory_reserved(dev)
    _st = torch.cuda.memory_stats(dev)
    print(f"[probe] one 256 MB request -> allocated {torch.cuda.memory_allocated(dev) // MB} / "
          f"reserved {r_after // MB} MB  (segments {_st.get('segment.large_pool.current')}, "
          f"alloc_retries {_st.get('num_alloc_retries')}, "
          f"cudaMalloc calls {_st.get('num_device_alloc', '?')}, "
          f"cudaFree calls {_st.get('num_device_free', '?')})")
    if r_after < r_cached:
        print(f"[probe] verdict: GC threshold LIVE -- the allocator handed back "
              f"{(r_cached - r_after) // MB} MB of cache before the new cudaMalloc.")
    else:
        print("[probe] verdict: GC threshold INERT -- the cache was kept and the new block "
              "was added on top (reserved grew by "
              f"{(r_after - r_cached) // MB} MB).")
    del big
    return 0


if __name__ == "__main__":
    sys.exit(main())

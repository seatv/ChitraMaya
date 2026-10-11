# Scaling Guide — the Secondary scaler and Output Upscale

ChitraMaya scales pixels in two very different places, and it pays to keep
them apart in your head:

| | **Secondary** (Restoration panel) | **Output Upscale** (Encoder panel) |
|---|---|---|
| What it scales | each restored *region* (a 256-px crop), before paste-back | the *whole frame*, after restoration, before encode |
| Why | large regions were stretched at paste-back and went soft | you want a bigger output (720p → 2160p, 1080p → 1440p / 2160p) |
| Cost | a few percent of a run | **hours per hour of video** on most cards |
| Default | **on** (RTX Super-Res 4× on NVIDIA, Real-ESRGAN 4× elsewhere) | **off** |
| Use it for | every run | clips and segments; titles overnight, and only on NVIDIA's RTX path in reasonable time |

Both take the same three kinds of scaler: **RTX Super-Res** (NVIDIA Maxine;
needs an RTX card and a recent driver), **Real-ESRGAN 4×** (bundled, every
edition), and **custom models** — any super-resolution `.pth` /
`.safetensors` you drop into `models\upscalers\`.

---

## 1. The Secondary — why it is on by default

The restorer works on a fixed 256 × 256 crop of each region. When the
region on the frame is larger than that — close-ups, VR, anything at 4K — the
restored crop used to be *stretched* back up at paste-back, and the result
went soft exactly where you look closest. With a secondary scaler the
restored crop is upscaled 4× first (to 1024 px), so the final resize
*shrinks* instead of stretching. Regions at or below 256 px skip the stage
automatically (the run report's `counts.secondary` says how many crops went
through and how many were too small to need it).

**Cost** is small because only regions are scaled: on an RTX 4060 the 17 s
1080p test cut restores in 32.5 s with RTX Super-Res 4× and 32.5 s with
Real-ESRGAN 4× — the same. On AMD and Intel the secondary runs in PyTorch
and is a larger share of a slower run, but still single-digit percent.

**Which scaler?** They produce two different *looks*, and neither is more
correct — pick by eye with **Test Frame**, which shows every region as
Mosaic → Restored:

- **RTX Super-Res** (NVIDIA): the smoother result — about half the fine
  texture of the other path on the same clip, with the same residual mosaic
  and flicker. Its **Scaler Quality** (low / medium / high / ultra; high is
  the pre-1.80 value) and **Scaler Denoise** (a second Maxine pass) are the
  two controls beneath it. Denoise changed nothing visible in our
  measurements at ~60 % more restore time; leave it at *none*.
- **Real-ESRGAN 4×** (`realesr-general-x4v3`, every edition): keeps the
  restorer's texture. The default on AMD and Intel, and a fair choice on
  NVIDIA if you prefer texture to smoothness.
- **A custom model** — see §3. The one we recommend trying is
  `4xNomos8k_span_otf_strong` (a SPAN model trained by Philip Hofmann): to
  our eyes a touch better than either built-in on skin, and on AMD and
  Intel it is *faster* than Real-ESRGAN (9 % less restore time on an RX
  9060 XT, 18 % on an Arc 140V) once its layers are fused at load, which
  ChitraMaya does. On NVIDIA it costs about the same as the built-ins.

If RTX Super-Res is selected on a machine where the Maxine runtime is not
available, the run falls back to Real-ESRGAN and the console and run report
(`effective.secondary`) say so.

**Region Margin** (Detection panel) grows the detector's box before it
reaches the restorer and the paste-back mask with it; **Feather** and
**Blend Mask** (Restoration panel) shape the edge where the scaled crop
meets the frame — the Face Swap blend feathers *outward* into clean skin,
so every mosaic pixel is fully replaced.

---

## 2. Output Upscale — a clip feature, or an overnight one

**Output Upscale** (Encoder panel) takes a factor — **2×**, **4×** — or a
target — **1080p**, **1440p**, **2160p** — and scales every finished frame
before it is encoded. It works inside a normal restoration run, and on its
own in **Upscale** mode (Alternate Execution Modes): decode → scale →
encode, no detector, no restorer, no tracker — for SFW content you simply
want bigger, or a restored output you want to bring to 4K in a second pass.
**Save Segment** writes a marked segment as its own file, so you can judge a
scaler on a minute before committing a title. The **Upscale Scaler**
dropdown picks the model; the progress modal, the console checkpoints and
the run report (`counts.output_upscale` — scaler, factor, ms per frame) all
know the mode.

A 4× model is run once and the result resized to the target when the target
is not a multiple of the model's factor (a 4× model to 2160p from 1080p runs
at 4× and shrinks by half); a 2× model to 2160p from 720p runs twice. The
run report's `effective.output_upscale` shows what happened.

### What it costs — measured

Whole-frame scaling touches every pixel of every frame, so it is a different
animal from the Secondary. These are real runs, not estimates:

| card | job | scaler | speed |
|---|---|---|---|
| RTX 3060 Ti 8 GB | 1080p title restored **and** upscaled to 2160p | RTX Super-Res (Quality ultra, Denoise ultra) | **6h30m for 2h03m** (0.32×); the upscale was 81 % of the wall; 85 ms per frame |
| RTX 5060 Ti | 1080p title, Upscale mode to 2160p | RTX Super-Res (ultra + ultra) | **10h02m for 2h41m** (0.27×); 61 ms per frame |
| RTX 3060 Ti 8 GB | 1080p title, Upscale mode to 2160p | Real-ESRGAN | **~29 h for 2h03m** (2.1 fps) |
| RX 9060 XT 16 GB | 720p title restored and upscaled to 2160p | custom 4× SPAN | **10h00m for 2h25m** (0.24×); 98 ms per frame; the upscale was 71 % of the wall |
| RX 9060 XT 16 GB | 17 s 1080p cut restored and upscaled to 2160p | custom 4× SPAN | 143 s (3.9 fps); 183 ms per frame for the upscale alone |
| Arc 140V (laptop iGPU) | 17 s 1080p cut restored and upscaled to 2160p | custom 4× SPAN | **505 s (1.1 fps)**; the scaler took 76 ms per frame, the 4K frames' trip to the encoder the rest |

Read it as three tiers:

- **NVIDIA with RTX Super-Res: the title-scale path.** 60–85 ms per 4K frame
  at the *ultra + ultra* settings those runs used, i.e. roughly three to four
  hours per hour of 1080p60 video on a 3060 Ti–5060 Ti class card — and
  note the Known issue below: those runs paid for a second denoise pass they
  did not need. At the defaults (Quality high, Denoise none) expect less.
- **Real-ESRGAN and custom models on NVIDIA: clips.** A compact 4× network
  over a 4K frame is ~475 ms on a 3060 Ti; a 2-hour title is a 29-hour job.
  Use them on a segment where you want their look, not on a title.
- **AMD and Intel: clips, or overnight.** There is no Maxine, so every
  whole-frame scaler is a PyTorch network over every frame — 100–180 ms per
  frame on an RX 9060 XT, and on the Arc 140V the 4K *output* itself runs at
  about 1 fps because the frames' trip to the encoder is the wall. A 720p
  title to 2160p on the 9060 XT is a ten-hour job; keep titles at their
  native resolution on the Arc.

The console's `[Pipeline] frame N/M avg X fps ... ETA` line prints the true
average every 500 frames; a minute into a run you know the whole run's
length. **Save Segment** first, then decide.

### The two things to know before a title

1. **Upscale Scaler = RTX Super-Res takes its Quality and Denoise from the
   Secondary's controls** (a known issue in 1.80.00; own controls are
   planned). With the Secondary at ultra + ultra, every 4K frame gets a
   second denoise pass. Set the Secondary's Quality / Denoise to what you
   want the *upscale* to run at — the defaults, high / none, are the cheap
   and sensible pair — or accept the cost knowingly.
2. **Resolution and memory.** A 2160p output is four times the pixels of
   1080p at every stage after the scaler — the encoder's surfaces, the
   frames in flight. On 8 GB cards, and on 16 GB laptops, the VRAM / RAM
   preflight in the console is worth reading before a long run.

---

## 3. Custom models — `models\upscalers\`

Drop any super-resolution model file into `models\upscalers\` (or use
**Manage Models → Custom scalers → Add**) and press **Rescan**: it appears
in both scaler dropdowns as `<file> (<architecture> Nx)`. ChitraMaya
identifies the architecture from the file itself — Real-ESRGAN / ESRGAN /
RRDB, SPAN, PLKSR and RealPLKSR, DAT, HAT, OmniSR, SwinIR, ATD (community
layout), Compact and some thirty more — through the **spandrel** library.
Files it cannot identify are listed with a reason (official-repository ATD
checkpoints, for instance, use a revised layout the identifier does not
know yet; a training checkpoint, a diffusion model, or a YOLO file in the
wrong folder each get their own hint).

**What runs, and how:**

- Custom models run in **PyTorch** on every edition (fp16 where the GPU
  supports it), tiled when the frame is large, with the model's required
  padding handled for it.
- As a **Secondary**, a custom model's output is **colour-matched** to the
  restorer's crop before paste-back (mean and contrast per channel, gain
  clamped), so a model's tint never leaks into the frame. Its factor is
  honoured: a 2× model upscales the crop 2×.
- As the **Upscale Scaler**, the model's factor and your target decide the
  pass count, as in §2.
- **SPAN models** get one special treatment: the reference implementation
  re-fuses their reparameterised layers on every frame, which made them run
  slower than they are; ChitraMaya fuses them once at load (bit-identical
  output, checked), which is why a SPAN model is the fastest custom option.

**Where to find models:** [OpenModelDB](https://openmodeldb.info) is the
community index; Philip Hofmann's
[models repository](https://github.com/Phhofm/models) carries many of the
best-regarded ones with training notes. **Check each model's licence** — they
differ, and they are not ChitraMaya's to grant.

**What to expect:** as a *Secondary*, none of the community models we
measured beat the two built-ins clearly; `4xNomos8k_span_otf_strong` is the
one that came closest — to our eyes slightly better on skin, and faster on
AMD and Intel. As a *whole-frame* scaler the picture is more open, because
the built-in Real-ESRGAN is a compact network chosen for speed, and a
stronger model may well look better on a clip you care about — at a cost
the table above lets you estimate. Measure with **Save Segment**; trust your
eyes over any of our numbers.

---

## 4. Recipes

**A 1080p title, best quality in reasonable time (NVIDIA):** defaults
(Secondary RTX Super-Res 4×, Temporal Stability 1, Output Upscale off).
Try Temporal Stability 3 on a segment if shimmer bothers you.

**The same title on AMD or Intel:** defaults (Secondary Real-ESRGAN 4×), or
`4xNomos8k_span_otf_strong` as the Secondary for a slightly better and
faster run. Output Upscale off.

**A 720p or 1080p title you want at 2160p (NVIDIA):** Output Upscale
**2160p** with Upscale Scaler **RTX Super-Res**, Secondary Quality **high**,
Denoise **none**. Overnight on a 3060 Ti-class card; check the ETA line
after the first 500 frames.

**A clip or segment you want at 2160p with a particular look (any
edition):** mark the segment, Output Upscale 2160p with Real-ESRGAN or a
custom model, **Save Segment**. Minutes, not hours.

**SFW content you just want bigger:** Alternate Execution Modes → **Upscale**,
pick the target and scaler, Restore & Save. No models to download.

**A 4K VR title on an 8 GB card:** the built-in **VR 4K** configuration
(Image Size 800, Max Clip 180, Split SBS), Secondary on, Output Upscale off,
and **Crop Cap** if the preflight asks for it.

"""Task 2 SigLIP feature caching.

Manual: "Use the 196 final patch features in their 14 x 14 spatial order, applying SigLIP's final
layer normalization exactly once. Cache them in FP16 with deterministic image preprocessing,
then remove the vision encoder from GPU memory before training Qwen." Caching cost is logged
separately from training ("separating feature caching from the training run").

- Encoder: google/siglip-base-patch16-224 at a pinned revision, FP32, eval mode, frozen.
- Preprocessing: the model's own image processor (resize to 224 x 224 bicubic, rescale,
  normalize with mean = std = 0.5); images converted to RGB. No randomness.
- Features: `last_hidden_state` of the vision tower = post_layernorm(encoder output), shape
  [196, 768] in the patch-embedding's row-major 14 x 14 order. The run checks this equals
  post_layernorm applied once to the encoder output, and casts to FP16 for storage.
- Images: the Kaggle mirror of COCO 2014; a seeded sample is checked byte-for-byte against
  images.cocodataset.org (DECISIONS.md T2-1).
- Also renders the grounding dev/test pairs (image, question, label) for label inspection.

Outputs (under --out-root)
  task2/features/siglip_{train,val}.safetensors   "features" [N, 196, 768] fp16, "image_ids" [N] int64
  task2/results/feature_cache_summary.json
  task2/results/label_inspection/grounding_{dev,test}_pNN.jpg
Usage: python cache_features.py   (on Kaggle, with the COCO 2014 dataset mounted)
"""

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

SELF = (Path(os.environ["PA3_BUNDLE"]) / "task2" / "cache_features.py" if "PA3_BUNDLE" in os.environ
        else Path(__file__).resolve())
REPO = SELF.parents[1]

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import save_file

SEED = 6304
MODEL_ID = "google/siglip-base-patch16-224"
MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
N_PATCHES, DIM = 196, 768
BATCH = 64
N_BYTE_CHECK = 4  # per split


def gb(x):
    return round(x / 2**30, 4)


def find_image_dir(split):
    hits = [d for d in glob.glob(f"/kaggle/input/**/{split}2014", recursive=True)
            if glob.glob(f"{d}/COCO_{split}2014_*.jpg")[:1]]
    assert hits, f"no {split}2014 image folder under /kaggle/input"
    return Path(sorted(hits)[0])


def byte_check(images, dirs):
    """Seeded sample of cached images compared with the official COCO server."""
    rng = np.random.default_rng(SEED)
    out = []
    for split in ["train", "val"]:
        for k in rng.choice(len(images[split]), N_BYTE_CHECK, replace=False):
            f = images[split][k]["file"]
            local = hashlib.sha256((dirs[split] / f).read_bytes()).hexdigest()
            with urllib.request.urlopen(f"http://images.cocodataset.org/{split}2014/{f}", timeout=60) as r:
                remote = hashlib.sha256(r.read()).hexdigest()
            out.append({"file": f, "identical": local == remote})
    return out


def load_processor():
    try:
        from transformers import SiglipImageProcessor  # reference (PIL) implementation
        proc = SiglipImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    except Exception:
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    return proc


class Images(torch.utils.data.Dataset):
    def __init__(self, files, root, proc):
        self.files, self.root, self.proc = files, root, proc

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with Image.open(self.root / self.files[i]) as im:
            return self.proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def encode_split(model, proc, items, root):
    files = [x["file"] for x in items]
    loader = torch.utils.data.DataLoader(Images(files, root, proc), batch_size=BATCH, shuffle=False,
                                         num_workers=4)
    feats = torch.empty(len(files), N_PATCHES, DIM, dtype=torch.float16)
    torch.cuda.reset_peak_memory_stats()
    t0, enc_s, n, ln_check = time.time(), 0.0, 0, None
    for px in loader:
        px = px.cuda(non_blocking=True)
        torch.cuda.synchronize()
        t = time.time()
        out = model(pixel_values=px).last_hidden_state
        if ln_check is None:  # final layer norm applied exactly once
            vm = getattr(model, "vision_model", model)  # transformers 4.x nests the tower; 5.x does not
            enc = vm.encoder(inputs_embeds=vm.embeddings(px))
            enc = enc.last_hidden_state if hasattr(enc, "last_hidden_state") else enc[0]
            ln_check = float((vm.post_layernorm(enc) - out).abs().max())
            assert ln_check < 1e-4, ln_check
        torch.cuda.synchronize()
        enc_s += time.time() - t
        assert out.shape[1:] == (N_PATCHES, DIM) and torch.isfinite(out).all()
        feats[n:n + len(out)] = out.half().cpu()
        n += len(out)
    assert n == len(files)
    return feats, {"images": n, "seconds_total": time.time() - t0, "seconds_encoder": enc_s,
                   "images_per_second": n / (time.time() - t0),
                   "peak_mem_allocated_gb": gb(torch.cuda.max_memory_allocated()),
                   "peak_mem_reserved_gb": gb(torch.cuda.max_memory_reserved()),
                   "post_layernorm_once_max_abs_diff": ln_check,
                   "fp16_feature_abs_max": float(feats.float().abs().max()),
                   "mean_token_norm": float(feats.float().norm(dim=-1).mean())}


def font(size):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def label_sheets(split, root, out_dir, per_page=8, h=224):
    """One row per pair: question, then both images with their annotation-derived answers."""
    pairs = json.loads((REPO / f"task2/splits/grounding_{split}.json").read_text())
    f_title, f_lab = font(16), font(14)
    paths = []
    for page in range(0, len(pairs), per_page):
        rows = []
        for p in pairs[page:page + per_page]:
            ims = []
            for x in p["items"]:
                with Image.open(root / x["file"]) as im:
                    im = im.convert("RGB")
                    ims.append((im.resize((round(im.width * h / im.height), h)), x))
            w = sum(im.width for im, _ in ims) + 30
            row = Image.new("RGB", (max(w, 700), h + 52), "white")
            d = ImageDraw.Draw(row)
            d.text((5, 3), f"{p['pair_id']}  |  {p['question']}", fill="black", font=f_title)
            x0 = 5
            for im, x in ims:
                row.paste(im, (x0, 26))
                d.text((x0, h + 30), f"answer: {x['answer']}   (image {x['image_id']})", fill="black", font=f_lab)
                x0 += im.width + 20
            rows.append(row)
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), "white")
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        path = out_dir / f"grounding_{split}_p{page // per_page + 1:02d}.jpg"
        sheet.save(path, quality=80)
        paths.append(path.name)
    return paths


KAGGLE_ARGS = []


def main():
    if len(sys.argv) == 1 and KAGGLE_ARGS:
        sys.argv += KAGGLE_ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path,
                   default=Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("."))
    args = p.parse_args()
    feat_dir = args.out_root / "task2/features"
    res_dir = args.out_root / "task2/results"
    insp_dir = res_dir / "label_inspection"
    for d in (feat_dir, insp_dir):
        d.mkdir(parents=True, exist_ok=True)

    from transformers import SiglipVisionModel
    import transformers
    images = json.loads((REPO / "task2/splits/images.json").read_text())
    dirs = {s: find_image_dir(s) for s in ["train", "val"]}
    missing = {s: [x["file"] for x in images[s] if not (dirs[s] / x["file"]).exists()] for s in dirs}
    assert not any(missing.values()), {s: v[:5] for s, v in missing.items()}
    byte = byte_check(images, dirs)
    assert all(b["identical"] for b in byte), byte

    t = time.time()
    proc = load_processor()
    model, info = SiglipVisionModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.float32,
                                                    output_loading_info=True)
    assert not info["missing_keys"], info["missing_keys"]  # unexpected keys = the unused text tower
    model = model.cuda().eval()
    for q in model.parameters():
        q.requires_grad_(False)
    load_s = time.time() - t

    summary = {"model": MODEL_ID, "model_revision": MODEL_REVISION, "dtype_compute": "float32",
               "dtype_stored": "float16", "processor": type(proc).__name__,
               "processor_config": {k: v for k, v in proc.to_dict().items()
                                    if k in ("size", "resample", "do_resize", "do_rescale", "rescale_factor",
                                             "do_normalize", "image_mean", "image_std")},
               "image_dirs": {s: str(d) for s, d in dirs.items()}, "byte_check": byte,
               "gpu": torch.cuda.get_device_name(0), "model_load_seconds": load_s, "splits": {},
               "missing_keys": sorted(info["missing_keys"]),
               "unexpected_keys_all_text_tower": all(not k.startswith("vision_model") for k in info["unexpected_keys"]),
               "versions": {"torch": torch.__version__, "transformers": transformers.__version__}}
    for split in ["train", "val"]:
        feats, rec = encode_split(model, proc, images[split], dirs[split])
        ids = torch.tensor([x["image_id"] for x in images[split]], dtype=torch.int64)
        path = feat_dir / f"siglip_{split}.safetensors"
        t = time.time()
        save_file({"features": feats.contiguous(), "image_ids": ids}, str(path))
        rec.update(save_seconds=time.time() - t, file=str(path.relative_to(args.out_root)),
                   file_bytes=path.stat().st_size)
        summary["splits"][split] = rec
        print(f"[{split}] {rec['images']} images, {rec['seconds_total']:.0f} s, peak {rec['peak_mem_allocated_gb']} GB")
        del feats

    # The encoder leaves the GPU here; training runs in a separate job that only reads the cache.
    del model
    torch.cuda.empty_cache()
    summary["gpu_mem_allocated_after_release_gb"] = gb(torch.cuda.memory_allocated())
    summary["label_inspection"] = {s: label_sheets(s, dirs["val"], insp_dir) for s in ["dev", "test"]}
    (res_dir / "feature_cache_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "label_inspection"}, indent=1, default=str))


if __name__ == "__main__":
    main()

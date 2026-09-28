"""
evaluate_datasets.py  —  EF-LIC tensor-size evaluation on Kodak + Tecnick + Cityscapes
---------------------------------------------------------------------------------------
For each dataset and each rate point (force_ind = 0..4):
  - Encode + decode every image
  - Record original size (uint8), compressed tensor size, ratio, latency, PSNR
  - Save per-image rows and averaged summary rows to CSV files
"""

import argparse
import csv
import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from EF_LIC import model
from test import (
    pack_inds,
    unpack_inds,
    replicate_pad,
    load_checkpoint,
    load_image,
)

warnings.filterwarnings("ignore")

FORCE_INDS = range(5)
PAD_MULTIPLE = 64


# ------------------------------------------------------------------
# Size helpers
# ------------------------------------------------------------------
def raw_tensor_bytes(inds):
    """Memory footprint of every VQ index tensor (numel * element_size)."""
    total = inds["z_inds"].numel() * inds["z_inds"].element_size()
    for t in inds["y_inds"]:
        total += t.numel() * t.element_size()
    return int(total)


def list_images(root: Path):
    """Recursively collect images (handles Cityscapes' nested city folders)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in exts and p.is_file())
    if not files:
        raise RuntimeError(f"No images found in {root.resolve()}")
    return files


# ------------------------------------------------------------------
# Encode / decode a single image
# ------------------------------------------------------------------
@torch.inference_mode()
def process_image(net, frame, force_ind):
    B, _, H, W = frame.shape
    padded = replicate_pad(frame, H, W)
    device = padded.device

    # --- ENCODE ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    inds = net.compress(padded, force_ind=force_ind)
    payload, meta, total_valid_bits = pack_inds(net, inds)

    if device.type == "cuda":
        torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - t0) * 1000.0

    # --- DECODE ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    inds_dec = unpack_inds(payload, meta, total_valid_bits, device)
    x_hat = net.decompress(inds_dec, force_ind=force_ind)[:, :, :H, :W]

    if device.type == "cuda":
        torch.cuda.synchronize()
    dec_ms = (time.perf_counter() - t0) * 1000.0

    # --- sizes ---
    orig_uint8_bytes = H * W * 3               # on-disk image
    comp_tensor_bytes = raw_tensor_bytes(inds) # VQ index tensor
    ratio = orig_uint8_bytes / comp_tensor_bytes

    # --- PSNR ---
    mse = F.mse_loss(
        ((x_hat + 1.0) * 0.5).clamp(0, 1),
        ((frame + 1.0) * 0.5).clamp(0, 1),
    ).item()
    psnr = -10.0 * math.log10(mse)

    return {
        "orig_bytes":   orig_uint8_bytes,
        "comp_bytes":   comp_tensor_bytes,
        "ratio":        ratio,
        "enc_ms":       enc_ms,
        "dec_ms":       dec_ms,
        "psnr":         psnr,
        "H":            H,
        "W":            W,
    }


# ------------------------------------------------------------------
# Evaluate one dataset
# ------------------------------------------------------------------
def evaluate_dataset(net, dataset_name, data_dir, device, out_dir,
                     max_images=None):
    images = list_images(data_dir)
    if max_images is not None and len(images) > max_images:
        images = images[:max_images]

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}   ({len(images)} images)   dir={data_dir}")
    print(f"{'='*70}")

    per_image_rows = []
    summary_rows   = []

    for force_ind in FORCE_INDS:
        net.prepare_inference_(force_ind=force_ind)

        # warm-up on first image once per rate point
        warm = load_image(images[0], device)
        _ = process_image(net, warm, force_ind)

        metrics = []
        print(f"\n--- force_ind = {force_ind} ---")
        header = (f"{'#':>3} | {'file':<45} | {'orig(B)':>10} | "
                  f"{'comp(B)':>10} | {'ratio':>8} | "
                  f"{'Enc(ms)':>9} | {'Dec(ms)':>9} | {'PSNR':>7}")
        print(header)
        print("-" * len(header))

        for idx, path in enumerate(images, 1):
            frame = load_image(path, device)
            r = process_image(net, frame, force_ind)
            metrics.append(r)

            row = {
                "dataset":     dataset_name,
                "rate":        force_ind,
                "image":       path.name,
                "H":           r["H"],
                "W":           r["W"],
                "orig_bytes":  r["orig_bytes"],
                "comp_bytes":  r["comp_bytes"],
                "ratio":       round(r["ratio"], 4),
                "enc_ms":      round(r["enc_ms"], 3),
                "dec_ms":      round(r["dec_ms"], 3),
                "psnr_db":     round(r["psnr"], 4),
            }
            per_image_rows.append(row)

            print(f"{idx:>3} | {path.name[:45]:<45} | "
                  f"{r['orig_bytes']:>10,} | {r['comp_bytes']:>10,} | "
                  f"{r['ratio']:>8.2f} | "
                  f"{r['enc_ms']:>9.2f} | {r['dec_ms']:>9.2f} | "
                  f"{r['psnr']:>7.3f}")

        # averages for this rate point
        avg_orig  = float(np.mean([m["orig_bytes"]   for m in metrics]))
        avg_comp  = float(np.mean([m["comp_bytes"]   for m in metrics]))
        avg_ratio = float(np.mean([m["ratio"]        for m in metrics]))
        avg_enc   = float(np.mean([m["enc_ms"]       for m in metrics]))
        avg_dec   = float(np.mean([m["dec_ms"]       for m in metrics]))
        avg_psnr  = float(np.mean([m["psnr"]         for m in metrics]))

        summary_rows.append({
            "dataset":       dataset_name,
            "rate":          force_ind,
            "num_images":    len(metrics),
            "orig_bytes":    round(avg_orig, 2),
            "comp_bytes":    round(avg_comp, 2),
            "ratio":         round(avg_ratio, 4),
            "enc_ms":        round(avg_enc, 3),
            "dec_ms":        round(avg_dec, 3),
            "psnr_db":       round(avg_psnr, 4),
        })

        print(f"\nAVG rate={force_ind} | "
              f"orig={avg_orig:,.0f} B | comp={avg_comp:,.0f} B | "
              f"ratio={avg_ratio:.2f}x | "
              f"Enc={avg_enc:.2f} ms | Dec={avg_dec:.2f} ms | "
              f"PSNR={avg_psnr:.3f} dB\n")

    # ---- write CSVs ----
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_csv = out_dir / f"{dataset_name}_per_image.csv"
    summary_csv   = out_dir / f"{dataset_name}_summary.csv"

    with open(per_image_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader()
        w.writerows(per_image_rows)

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Saved per-image CSV : {per_image_csv}")
    print(f"Saved summary CSV   : {summary_csv}")

    return summary_rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kodak-dir",      type=Path, default=Path("kodak"))
    ap.add_argument("--tecnick-dir",    type=Path, default=Path("tecnick_flat"),
                    help="Flattened Tecnick folder (or nested; rglob handles both).")
    ap.add_argument("--cityscapes-dir", type=Path,
                    default=Path("/home/common/EF-LIC/datasets/cityscapes/"
                                 "leftImg8bit/val/frankfurt"),
                    help="Cityscapes root or city folder. Defaults to Frankfurt val.")
    ap.add_argument("--ckpt-path",      type=Path,
                    default=Path("ckpt/checkpoint.pth.tar"))
    ap.add_argument("--device",         type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir",        type=Path, default=Path("results"))
    ap.add_argument("--datasets",       type=str, nargs="+",
                    default=["kodak", "tecnick", "cityscapes"],
                    choices=["kodak", "tecnick", "cityscapes"])
    ap.add_argument("--max-cityscapes", type=int, default=None,
                    help="Subsample Cityscapes to at most this many images "
                         "(full Frankfurt val = 268 images).")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")

    # Load model once, reuse across datasets
    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    print(f"Loaded checkpoint: {args.ckpt_path}")

    dir_map = {
        "kodak":      args.kodak_dir,
        "tecnick":    args.tecnick_dir,
        "cityscapes": args.cityscapes_dir,
    }

    all_summaries = []
    for name in args.datasets:
        data_dir = dir_map[name]
        if not data_dir.exists():
            print(f"[skip] {name}: directory not found ({data_dir})")
            continue

        max_imgs = args.max_cityscapes if name == "cityscapes" else None
        summary = evaluate_dataset(net, name, data_dir, device,
                                   args.out_dir, max_images=max_imgs)
        all_summaries.extend(summary)

    # ---- combined summary ----
    if all_summaries:
        combined_csv = args.out_dir / "combined_summary.csv"
        with open(combined_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            w.writerows(all_summaries)
        print(f"\nSaved combined summary -> {combined_csv}")

    # ---- pretty final table ----
    print(f"\n{'='*110}")
    print(f"{'FINAL SUMMARY':^110}")
    print(f"{'='*110}")
    hdr = (f"{'dataset':<12} | {'rate':>4} | {'N':>5} | "
           f"{'orig(B)':>12} | {'comp(B)':>12} | {'ratio':>8} | "
           f"{'Enc(ms)':>9} | {'Dec(ms)':>9} | {'PSNR':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in all_summaries:
        print(f"{r['dataset']:<12} | {r['rate']:>4} | {r['num_images']:>5} | "
              f"{r['orig_bytes']:>12,.0f} | {r['comp_bytes']:>12,.0f} | "
              f"{r['ratio']:>8.2f} | "
              f"{r['enc_ms']:>9.2f} | {r['dec_ms']:>9.2f} | "
              f"{r['psnr_db']:>8.3f}")


if __name__ == "__main__":
    main()

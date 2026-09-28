"""
evaluate_all.py
---------------
Runs EF-LIC + YOLOS on Kodak, Tecnick and Cityscapes.

For every (dataset, image, rate) it records:
  - input image dimensions    : (H, W, 3)
  - latent tensor shapes      : z_inds shape, y_inds shapes, total numel
  - EF-LIC size / ratio / latency
  - YOLOS detections on original and reconstructed
  - detection loss = |det_orig - det_recon|

Outputs:
  results_all/<dataset>_per_image.csv     one row per (image, rate)
  results_all/<dataset>_summary.csv       one row per rate (averages)
  results_all/combined_summary.csv        all datasets, all rates
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
from transformers import YolosImageProcessor, YolosForObjectDetection

from EF_LIC import model
from test import (
    pack_inds, unpack_inds, replicate_pad,
    load_checkpoint, load_image,
)

warnings.filterwarnings("ignore")

FORCE_INDS = range(5)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def raw_tensor_bytes(inds):
    total = inds["z_inds"].numel() * inds["z_inds"].element_size()
    for t in inds["y_inds"]:
        total += t.numel() * t.element_size()
    return int(total)


def latent_shapes(inds):
    """Return a compact string describing the latent tensor dimensions."""
    z_shape = tuple(inds["z_inds"].shape)
    y_shapes = [tuple(t.shape) for t in inds["y_inds"]]
    z_numel = inds["z_inds"].numel()
    y_numel = sum(t.numel() for t in inds["y_inds"])
    return {
        "z_shape":       str(z_shape),
        "y_shapes":      str(y_shapes),
        "z_numel":       z_numel,
        "y_numel":       y_numel,
        "total_numel":   z_numel + y_numel,
        "z_dtype":       str(inds["z_inds"].dtype),
        "y_dtype":       str(inds["y_inds"][0].dtype),
    }


def list_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in exts and p.is_file())
    if not files:
        raise RuntimeError(f"No images found in {root.resolve()}")
    return files


# ------------------------------------------------------------------
# EF-LIC encode / decode
# ------------------------------------------------------------------
@torch.inference_mode()
def eflic_process(net, frame, force_ind):
    B, _, H, W = frame.shape
    padded = replicate_pad(frame, H, W)
    device = padded.device

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    inds = net.compress(padded, force_ind=force_ind)
    payload, meta, total_valid_bits = pack_inds(net, inds)
    if device.type == "cuda":
        torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - t0) * 1000.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    inds_dec = unpack_inds(payload, meta, total_valid_bits, device)
    x_hat = net.decompress(inds_dec, force_ind=force_ind)[:, :, :H, :W]
    if device.type == "cuda":
        torch.cuda.synchronize()
    dec_ms = (time.perf_counter() - t0) * 1000.0

    mse = F.mse_loss(
        ((x_hat + 1.0) * 0.5).clamp(0, 1),
        ((frame + 1.0) * 0.5).clamp(0, 1),
    ).item()
    psnr = -10.0 * math.log10(mse)

    return {
        "x_hat":       x_hat,
        "comp_bytes":  raw_tensor_bytes(inds),
        "enc_ms":      enc_ms,
        "dec_ms":      dec_ms,
        "psnr":        psnr,
        "H":           H,
        "W":           W,
        "latent":      latent_shapes(inds),
    }


def tensor_to_pil(t):
    t = ((t.clamp(-1, 1) + 1.0) * 0.5).cpu()
    arr = (t.squeeze(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


# ------------------------------------------------------------------
# YOLOS
# ------------------------------------------------------------------
def count_detections(yolos_model, yolos_processor, pil_img,
                     threshold, device):
    inputs = yolos_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = yolos_model(**inputs)
    target_sizes = torch.tensor([pil_img.size[::-1]]).to(device)
    results = yolos_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes)[0]
    return len(results["scores"])


# ------------------------------------------------------------------
# Evaluate one dataset
# ------------------------------------------------------------------
def evaluate_dataset(net, yolos_model, yolos_processor,
                     dataset_name, data_dir, device, out_dir,
                     threshold, max_images=None):
    images = list_images(data_dir)
    if max_images is not None and len(images) > max_images:
        images = images[:max_images]

    print(f"\n{'='*90}")
    print(f"Dataset: {dataset_name}   ({len(images)} images)   dir={data_dir}")
    print(f"{'='*90}")

    per_image_rows = []
    summary_rows   = []

    for force_ind in FORCE_INDS:
        net.prepare_inference_(force_ind=force_ind)

        # warm-up
        warm = load_image(images[0], device)
        _ = eflic_process(net, warm, force_ind)

        metrics = []
        print(f"\n--- force_ind = {force_ind} ---")
        header = (f"{'#':>3} | {'file':<38} | {'orig(B)':>10} | "
                  f"{'comp(B)':>9} | {'ratio':>7} | "
                  f"{'det_o':>5} | {'det_r':>5} | {'loss':>5} | "
                  f"{'Enc':>7} | {'Dec':>7}")
        print(header)
        print("-" * len(header))

        for idx, path in enumerate(images, 1):
            frame = load_image(path, device)
            original_pil = Image.open(path).convert("RGB")

            # EF-LIC
            r = eflic_process(net, frame, force_ind)
            recon_pil = tensor_to_pil(r["x_hat"])

            # YOLOS on both
            det_orig  = count_detections(yolos_model, yolos_processor,
                                         original_pil, threshold, str(device))
            det_recon = count_detections(yolos_model, yolos_processor,
                                         recon_pil, threshold, str(device))
            det_loss  = abs(det_orig - det_recon)

            orig_u8 = r["H"] * r["W"] * 3
            ratio   = orig_u8 / r["comp_bytes"]

            row = {
                "dataset":       dataset_name,
                "rate":          force_ind,
                "image":         path.name,
                "input_H":       r["H"],
                "input_W":       r["W"],
                "input_shape":   f"({r['H']}, {r['W']}, 3)",
                "z_shape":       r["latent"]["z_shape"],
                "y_shapes":      r["latent"]["y_shapes"],
                "z_numel":       r["latent"]["z_numel"],
                "y_numel":       r["latent"]["y_numel"],
                "latent_numel":  r["latent"]["total_numel"],
                "z_dtype":       r["latent"]["z_dtype"],
                "y_dtype":       r["latent"]["y_dtype"],
                "orig_bytes":    orig_u8,
                "comp_bytes":    r["comp_bytes"],
                "ratio":         round(ratio, 4),
                "enc_ms":        round(r["enc_ms"], 3),
                "dec_ms":        round(r["dec_ms"], 3),
                "psnr_db":       round(r["psnr"], 4),
                "det_orig":      det_orig,
                "det_recon":     det_recon,
                "det_loss":      det_loss,
            }
            per_image_rows.append(row)
            metrics.append(row)

            print(f"{idx:>3} | {path.name[:38]:<38} | "
                  f"{orig_u8:>10,} | {r['comp_bytes']:>9,} | "
                  f"{ratio:>7.2f} | "
                  f"{det_orig:>5} | {det_recon:>5} | {det_loss:>5} | "
                  f"{r['enc_ms']:>7.2f} | {r['dec_ms']:>7.2f}")

        # --- averages ---
        def mean(key): return float(np.mean([m[key] for m in metrics]))

        summary_rows.append({
            "dataset":             dataset_name,
            "rate":                force_ind,
            "num_images":          len(metrics),
            "input_shape":         metrics[0]["input_shape"],
            "z_shape":             metrics[0]["z_shape"],
            "y_shapes":            metrics[0]["y_shapes"],
            "z_numel":             metrics[0]["z_numel"],
            "y_numel":             metrics[0]["y_numel"],
            "latent_numel":        metrics[0]["latent_numel"],
            "orig_bytes":          round(mean("orig_bytes"), 2),
            "comp_bytes":          round(mean("comp_bytes"), 2),
            "ratio":               round(mean("ratio"), 4),
            "enc_ms":              round(mean("enc_ms"), 3),
            "dec_ms":              round(mean("dec_ms"), 3),
            "psnr_db":             round(mean("psnr_db"), 4),
            "mean_det_orig":       round(mean("det_orig"), 4),
            "mean_det_recon":      round(mean("det_recon"), 4),
            "mean_det_loss":       round(mean("det_loss"), 4),
            "max_det_loss":        int(max(m["det_loss"] for m in metrics)),
            "pct_images_perfect":  round(
                100.0 * sum(1 for m in metrics if m["det_loss"] == 0)
                / len(metrics), 2),
        })

        print(f"\nAVG rate={force_ind} | "
              f"comp={mean('comp_bytes'):,.0f} B | ratio={mean('ratio'):.2f}x | "
              f"det_orig={mean('det_orig'):.2f}  det_recon={mean('det_recon'):.2f}  "
              f"loss={mean('det_loss'):.3f} | "
              f"Enc={mean('enc_ms'):.2f} ms  Dec={mean('dec_ms'):.2f} ms | "
              f"PSNR={mean('psnr_db'):.2f} dB\n")

    # ---- write CSVs ----
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{dataset_name}_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader(); w.writerows(per_image_rows)

    with open(out_dir / f"{dataset_name}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    print(f"Saved -> {out_dir / (dataset_name + '_per_image.csv')}")
    print(f"Saved -> {out_dir / (dataset_name + '_summary.csv')}")

    return summary_rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kodak-dir",      type=Path, default=Path("kodak"))
    ap.add_argument("--tecnick-dir",    type=Path, default=Path("tecnick_flat"))
    ap.add_argument("--cityscapes-dir", type=Path,
                    default=Path("/home/common/EF-LIC/datasets/cityscapes/"
                                 "leftImg8bit/val/frankfurt"))
    ap.add_argument("--ckpt-path",      type=Path,
                    default=Path("ckpt/checkpoint.pth.tar"))
    ap.add_argument("--yolos-model",    type=str, default="hustvl/yolos-small")
    ap.add_argument("--threshold",      type=float, default=0.5,
                    help="YOLOS confidence threshold")
    ap.add_argument("--device",         type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir",        type=Path, default=Path("results_all"))
    ap.add_argument("--datasets",       type=str, nargs="+",
                    default=["kodak", "tecnick", "cityscapes"],
                    choices=["kodak", "tecnick", "cityscapes"])
    ap.add_argument("--max-kodak",      type=int, default=None)
    ap.add_argument("--max-tecnick",    type=int, default=None)
    ap.add_argument("--max-cityscapes", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}")
    print(f"YOLOS : {args.yolos_model}  (threshold={args.threshold})\n")

    # ---- load models once ----
    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    print(f"EF-LIC checkpoint: {args.ckpt_path}")

    yolos_processor = YolosImageProcessor.from_pretrained(args.yolos_model)
    yolos_model     = YolosForObjectDetection.from_pretrained(
        args.yolos_model).to(device).eval()
    print(f"YOLOS loaded\n")

    dir_map = {
        "kodak":      (args.kodak_dir,      args.max_kodak),
        "tecnick":    (args.tecnick_dir,    args.max_tecnick),
        "cityscapes": (args.cityscapes_dir, args.max_cityscapes),
    }

    all_summaries = []
    for name in args.datasets:
        data_dir, max_imgs = dir_map[name]
        if not data_dir.exists():
            print(f"[skip] {name}: {data_dir} not found")
            continue
        summary = evaluate_dataset(
            net, yolos_model, yolos_processor,
            name, data_dir, device, args.out_dir,
            args.threshold, max_images=max_imgs,
        )
        all_summaries.extend(summary)

    if all_summaries:
        combined_csv = args.out_dir / "combined_summary.csv"
        with open(combined_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader(); w.writerows(all_summaries)
        print(f"\nSaved combined summary -> {combined_csv}")

    # ---- final pretty table ----
    print(f"\n{'='*120}")
    print(f"{'FINAL SUMMARY':^120}")
    print(f"{'='*120}")
    hdr = (f"{'dataset':<11} | {'rate':>4} | {'N':>4} | "
           f"{'input':>15} | {'z_shape':>14} | {'latent_numel':>12} | "
           f"{'ratio':>7} | "
           f"{'det_o':>6} | {'det_r':>6} | {'loss':>6} | "
           f"{'%perfect':>8} | "
           f"{'Enc':>7} | {'Dec':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in all_summaries:
        print(f"{r['dataset']:<11} | {r['rate']:>4} | {r['num_images']:>4} | "
              f"{r['input_shape']:>15} | {r['z_shape']:>14} | "
              f"{r['latent_numel']:>12,} | "
              f"{r['ratio']:>7.2f} | "
              f"{r['mean_det_orig']:>6.2f} | {r['mean_det_recon']:>6.2f} | "
              f"{r['mean_det_loss']:>6.3f} | "
              f"{r['pct_images_perfect']:>7.2f}% | "
              f"{r['enc_ms']:>7.2f} | {r['dec_ms']:>7.2f}")

    # ---- detection loss trend per dataset ----
    print(f"\n{'='*70}")
    print(f"{'DETECTION LOSS TREND (mean |Δ| detections)':^70}")
    print(f"{'='*70}")
    print(f"{'dataset':<12} | "
          + " | ".join(f"rate={r}" for r in FORCE_INDS))
    print("-" * 70)
    for name in args.datasets:
        rows = [r for r in all_summaries if r["dataset"] == name]
        if not rows:
            continue
        line = f"{name:<12} | " + " | ".join(
            f"{r['mean_det_loss']:>7.3f}" for r in rows)
        print(line)


if __name__ == "__main__":
    main()

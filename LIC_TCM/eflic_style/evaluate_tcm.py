"""
evaluate_tcm.py
---------------
Runs LIC_TCM on Kodak, Tecnick and Cityscapes, over all discovered rates.
Writes CSVs: results_tcm/<dataset>_per_image.csv, <dataset>_summary.csv,
             combined_summary.csv
"""

import argparse
import csv
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from lic_tcm_wrapper import LICTCMAdapter

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
def list_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in exts and p.is_file())
    if not files:
        raise RuntimeError(f"No images found in {root.resolve()}")
    return files


def load_image(path, device):
    from torchvision import transforms
    t = transforms.ToTensor()(Image.open(path).convert("RGB")) * 2.0 - 1.0
    return t.unsqueeze(0).to(device, non_blocking=True)


# ------------------------------------------------------------------
@torch.inference_mode()
def process_image(net, frame, force_ind):
    _, _, H, W = frame.shape
    device = frame.device

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    compressed = net.compress(frame, force_ind=force_ind)
    if device.type == "cuda":
        torch.cuda.synchronize()
    enc_ms = (time.perf_counter() - t0) * 1000.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_hat = net.decompress(compressed, force_ind=force_ind)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dec_ms = (time.perf_counter() - t0) * 1000.0

    orig_bytes = H * W * 3
    comp_bytes = compressed["packed_bytes"]
    ratio = orig_bytes / comp_bytes

    mse = F.mse_loss(
        ((x_hat + 1.0) * 0.5).clamp(0, 1),
        ((frame + 1.0) * 0.5).clamp(0, 1),
    ).item()
    psnr = -10.0 * math.log10(mse)

    return {
        "orig_bytes": orig_bytes,
        "comp_bytes": comp_bytes,
        "ratio":      ratio,
        "enc_ms":     enc_ms,
        "dec_ms":     dec_ms,
        "psnr":       psnr,
        "H":          H, "W": W,
    }


# ------------------------------------------------------------------
def evaluate_dataset(net, dataset_name, data_dir, device, out_dir,
                     max_images=None):
    images = list_images(data_dir)
    if max_images is not None and len(images) > max_images:
        # deterministic subsample: take evenly-spaced images
        step = len(images) / max_images
        images = [images[int(i * step)] for i in range(max_images)]

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}   ({len(images)} images)   dir={data_dir}")
    print(f"{'='*80}")

    per_image_rows, summary_rows = [], []

    for force_ind in range(net.num_rates):
        lam = net.available_lambdas[force_ind]
        net.prepare_inference_(force_ind=force_ind)

        warm = load_image(images[0], device)
        _ = process_image(net, warm, force_ind)

        metrics = []
        print(f"\n--- force_ind = {force_ind}  (λ={lam}) ---")
        header = (f"{'#':>3} | {'file':<42} | {'orig(B)':>10} | "
                  f"{'comp(B)':>9} | {'ratio':>8} | "
                  f"{'Enc(ms)':>9} | {'Dec(ms)':>9} | {'PSNR':>7}")
        print(header); print("-" * len(header))

        for idx, path in enumerate(images, 1):
            frame = load_image(path, device)
            r = process_image(net, frame, force_ind)
            metrics.append(r)

            per_image_rows.append({
                "dataset": dataset_name,
                "rate": force_ind,
                "lambda": lam,
                "image": path.name,
                "H": r["H"], "W": r["W"],
                "orig_bytes": r["orig_bytes"],
                "comp_bytes": r["comp_bytes"],
                "ratio": round(r["ratio"], 4),
                "enc_ms": round(r["enc_ms"], 3),
                "dec_ms": round(r["dec_ms"], 3),
                "psnr_db": round(r["psnr"], 4),
            })

            print(f"{idx:>3} | {path.name[:42]:<42} | "
                  f"{r['orig_bytes']:>10,} | {r['comp_bytes']:>9,} | "
                  f"{r['ratio']:>8.2f} | "
                  f"{r['enc_ms']:>9.2f} | {r['dec_ms']:>9.2f} | "
                  f"{r['psnr']:>7.3f}")

        def m(k): return float(np.mean([x[k] for x in metrics]))
        summary_rows.append({
            "dataset": dataset_name,
            "rate": force_ind,
            "lambda": lam,
            "num_images": len(metrics),
            "orig_bytes": round(m("orig_bytes"), 2),
            "comp_bytes": round(m("comp_bytes"), 2),
            "ratio": round(m("ratio"), 4),
            "enc_ms": round(m("enc_ms"), 3),
            "dec_ms": round(m("dec_ms"), 3),
            "psnr_db": round(m("psnr"), 4),
        })
        print(f"\nAVG rate={force_ind} | orig={m('orig_bytes'):,.0f} B | "
              f"comp={m('comp_bytes'):,.0f} B | ratio={m('ratio'):.2f}x | "
              f"Enc={m('enc_ms'):.2f} ms | Dec={m('dec_ms'):.2f} ms | "
              f"PSNR={m('psnr'):.3f} dB\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("per_image", per_image_rows), ("summary", summary_rows)]:
        p = out_dir / f"{dataset_name}_{name}.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"Saved -> {p}")
    return summary_rows


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kodak-dir",      type=Path, default=Path("../kodak"))
    ap.add_argument("--tecnick-dir",    type=Path, default=Path("../tecnick_flat"))
    ap.add_argument("--cityscapes-dir", type=Path,
                    default=Path("/home/common/EF-LIC/datasets/cityscapes/"
                                 "leftImg8bit/val/frankfurt"))
    ap.add_argument("--ckpt-dir",       type=Path,
                    default=Path(__file__).resolve().parent.parent / "checkpoints")
    ap.add_argument("--device",         type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir",        type=Path, default=Path("results_tcm"))
    ap.add_argument("--datasets", nargs="+",
                    default=["kodak", "tecnick", "cityscapes"],
                    choices=["kodak", "tecnick", "cityscapes"])
    ap.add_argument("--model-size",     type=int, default=64, choices=[64, 128])

    # ---- per-dataset caps ----
    ap.add_argument("--max-kodak",       type=int, default=None,
                    help="Cap Kodak images (only 24 exist; cap is a no-op unless >24).")
    ap.add_argument("--max-tecnick",     type=int, default=None,
                    help="Cap Tecnick images (default: all 360).")
    ap.add_argument("--max-cityscapes",  type=int, default=None,
                    help="Cap Cityscapes images (default: all in folder).")
    ap.add_argument("--max-images",      type=int, default=None,
                    help="Global cap applied to any dataset without a specific cap.")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    print(f"Device     : {device}")
    print(f"Model size : N={args.model_size}")
    print(f"Checkpoints: {args.ckpt_dir}")
    print(f"Caps       : kodak={args.max_kodak} tecnick={args.max_tecnick} "
          f"cityscapes={args.max_cityscapes} global={args.max_images}\n")

    net = LICTCMAdapter(args.ckpt_dir, device=device,
                        model_size=args.model_size)
    print(f"Discovered {net.num_rates} rates: {net.available_lambdas}\n")

    dir_map = {
        "kodak":      (args.kodak_dir,      args.max_kodak),
        "tecnick":    (args.tecnick_dir,    args.max_tecnick),
        "cityscapes": (args.cityscapes_dir, args.max_cityscapes),
    }

    all_summaries = []
    for name in args.datasets:
        d, cap = dir_map[name]
        if not d.exists():
            print(f"[skip] {name}: {d} not found"); continue
        # Global cap applies only if no per-dataset cap is set
        if cap is None:
            cap = args.max_images
        summary = evaluate_dataset(net, name, d, device, args.out_dir,
                                   max_images=cap)
        all_summaries.extend(summary)

    if all_summaries:
        p = args.out_dir / "combined_summary.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader(); w.writerows(all_summaries)
        print(f"\nCombined summary -> {p}")

    print(f"\n{'='*115}")
    print(f"{'FINAL SUMMARY':^115}")
    print(f"{'='*115}")
    hdr = (f"{'dataset':<12} | {'rate':>4} | {'lambda':>8} | {'N':>5} | "
           f"{'orig(B)':>12} | {'comp(B)':>12} | {'ratio':>8} | "
           f"{'Enc(ms)':>9} | {'Dec(ms)':>9} | {'PSNR':>8}")
    print(hdr); print("-" * len(hdr))
    for r in all_summaries:
        print(f"{r['dataset']:<12} | {r['rate']:>4} | {r['lambda']:>8.4f} | "
              f"{r['num_images']:>5} | "
              f"{r['orig_bytes']:>12,.0f} | {r['comp_bytes']:>12,.0f} | "
              f"{r['ratio']:>8.2f} | "
              f"{r['enc_ms']:>9.2f} | {r['dec_ms']:>9.2f} | "
              f"{r['psnr_db']:>8.3f}")


if __name__ == "__main__":
    main()

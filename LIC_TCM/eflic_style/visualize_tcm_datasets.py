"""
visualize_tcm_datasets.py
-------------------------
Side-by-side visualizations for TCM on kodak / tecnick / cityscapes.

Per image and per rate, the overlay shows:
  - input dimensions
  - latent y spatial dimensions
  - per-stream byte breakdown (y[i], z[i])
  - the explicit arithmetic summing to the compressed size
  - encoder / decoder mean ± std over N profiling iterations
  - ratio, PSNR
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from lic_tcm_wrapper import LICTCMAdapter, HEADER_BYTES


# ------------------------------------------------------------------
def list_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in exts and p.is_file())
    if not files:
        raise RuntimeError(f"No images found in {root.resolve()}")
    return files


def pick_images(root: Path, n: int, seed: int = 42):
    files = list_images(root)
    if len(files) <= n:
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, n), key=lambda p: p.name)


def load_image(path, device):
    from torchvision import transforms
    t = transforms.ToTensor()(Image.open(path).convert("RGB")) * 2.0 - 1.0
    return t.unsqueeze(0).to(device)


def tensor_to_pil(t):
    t = ((t.clamp(-1, 1) + 1.0) * 0.5).cpu()
    arr = (t.squeeze(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def fmt_bytes(n):
    if n >= 1024 * 1024: return f"{n/1024/1024:.2f} MB"
    if n >= 1024:        return f"{n/1024:.1f} KB"
    return f"{n} B"


def latent_y_dims(compressed):
    """Approximate latent y spatial dims (TCM downsamples 16x)."""
    return compressed["H"] // 16, compressed["W"] // 16


# ------------------------------------------------------------------
# Profiling
# ------------------------------------------------------------------
@torch.inference_mode()
def profile_encode_decode(net, frame, force_ind, iters):
    device = frame.device
    enc_times, dec_times = [], []
    compressed = None
    x_hat = None

    for _ in range(3):                         # warm-up
        compressed = net.compress(frame, force_ind=force_ind)
        x_hat = net.decompress(compressed, force_ind=force_ind)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        compressed = net.compress(frame, force_ind=force_ind)
        if device.type == "cuda":
            torch.cuda.synchronize()
        enc_times.append((time.perf_counter() - t0) * 1000.0)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_hat = net.decompress(compressed, force_ind=force_ind)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dec_times.append((time.perf_counter() - t0) * 1000.0)

    enc = np.asarray(enc_times); dec = np.asarray(dec_times)
    return {
        "enc_mean": float(enc.mean()), "enc_std": float(enc.std()),
        "enc_min":  float(enc.min()),  "enc_max": float(enc.max()),
        "dec_mean": float(dec.mean()), "dec_std": float(dec.std()),
        "dec_min":  float(dec.min()),  "dec_max": float(dec.max()),
        "iters": iters, "x_hat": x_hat, "compressed": compressed,
    }


# ------------------------------------------------------------------
# Image building
# ------------------------------------------------------------------
def save_side_by_side(orig_pil, recon_pil, title, stats_lines, out_path,
                      header_h=290, gap=14, max_panel_width=1000):
    W, H = orig_pil.size
    if W > max_panel_width:
        scale = max_panel_width / W
        new_size = (int(W * scale), int(H * scale))
        orig_pil  = orig_pil.resize(new_size, Image.LANCZOS)
        recon_pil = recon_pil.resize(new_size, Image.LANCZOS)
        W, H = orig_pil.size

    canvas = Image.new("RGB", (W * 2 + gap, H + header_h), (248, 248, 250))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
        font_label = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        font_stat  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except Exception:
        font_title = font_label = font_stat = ImageFont.load_default()

    draw.text((12, 8), title, fill=(20, 20, 20), font=font_title)
    y = 34
    for line in stats_lines:
        if line.startswith("---"):
            draw.text((14, y), line, fill=(120, 120, 120), font=font_stat)
        elif line.startswith("TOTAL") or line.startswith("Compressed size"):
            draw.text((14, y), line, fill=(150, 0, 0), font=font_stat)
        else:
            draw.text((14, y), line, fill=(40, 40, 40), font=font_stat)
        y += 17

    img_top = header_h
    draw.text((12, img_top - 24), "Original",
              fill=(0, 110, 0), font=font_label)
    draw.text((W + gap + 12, img_top - 24),
              "Compressed -> Decompressed",
              fill=(150, 0, 0), font=font_label)

    canvas.paste(orig_pil, (0, img_top))
    canvas.paste(recon_pil, (W + gap, img_top))
    draw.line([(W + gap // 2, img_top), (W + gap // 2, img_top + H)],
              fill=(200, 200, 200), width=2)
    canvas.save(out_path)


# ------------------------------------------------------------------
def process_dataset(net, dataset_name, img_dir, num_images, seed,
                    device, out_root, profile_iters):
    images = pick_images(img_dir, num_images, seed=seed)
    print(f"\n{'='*70}")
    print(f"{dataset_name}: {len(images)} images  ({img_dir})")
    print(f"{'='*70}")

    for i, img_path in enumerate(images, 1):
        subdir = out_root / dataset_name / f"img{i}"
        subdir.mkdir(parents=True, exist_ok=True)

        frame = load_image(img_path, device)
        original_pil = Image.open(img_path).convert("RGB")
        H, W = original_pil.height, original_pil.width

        print(f"\n[{i}/{len(images)}] {img_path.name}  ({W}x{H})")

        for force_ind in range(net.num_rates):
            lam = net.available_lambdas[force_ind]
            net.prepare_inference_(force_ind=force_ind)

            prof = profile_encode_decode(net, frame, force_ind, profile_iters)
            x_hat = prof["x_hat"]
            compressed = prof["compressed"]

            orig_b = H * W * 3
            comp_b = compressed["packed_bytes"]
            ratio = orig_b / comp_b
            mse = F.mse_loss(
                ((x_hat + 1.0) * 0.5).clamp(0, 1),
                ((frame + 1.0) * 0.5).clamp(0, 1),
            ).item()
            psnr = -10.0 * np.log10(mse)

            latent_H, latent_W = latent_y_dims(compressed)
            breakdown = compressed["stream_breakdown"]
            header_b  = compressed["header_bytes"]
            naive_b   = compressed["naive_bytes"]

            # ---- Stream arithmetic ----
            y_items = [(lbl, b) for lbl, b in breakdown if lbl.startswith("y[")]
            z_items = [(lbl, b) for lbl, b in breakdown if lbl.startswith("z[")]
            others  = [(lbl, b) for lbl, b in breakdown
                       if not (lbl.startswith("y[") or lbl.startswith("z["))]

            def items_str(items):
                return "  ".join(f"{lbl}={b}" for lbl, b in items)

            sum_expr_parts = [str(b) for _, b in y_items] \
                           + [str(b) for _, b in z_items] \
                           + [str(b) for _, b in others] \
                           + [str(header_b)]

            if len(sum_expr_parts) <= 8:
                sum_expr = " + ".join(sum_expr_parts)
            else:
                head = " + ".join(sum_expr_parts[:4])
                tail = " + ".join(sum_expr_parts[-3:])
                sum_expr = f"{head} + ... + {tail}"

            stats_lines = [
                f"Source                  : {img_path.name[:70]}",
                f"Input dimensions        : (H={H}, W={W}, C=3)  uint8",
                f"Latent y dims (H, W)    : ({latent_H}, {latent_W})",
                f"Streams                 : {len(breakdown)} entropy-coded streams",
                "---",
                f"  y-slices (per slice)  : {items_str(y_items)}",
                f"  hyperprior z          : {items_str(z_items) if z_items else 'n/a'}",
                f"  header / metadata     : {header_b} B",
                "---",
                f"SUM expression          : {sum_expr}",
                f"TOTAL compressed size   : {comp_b:,} B  ({fmt_bytes(comp_b)})",
                "---",
                f"Original size (uint8)   : {orig_b:,} B  ({fmt_bytes(orig_b)})",
                f"Compressed size         : {comp_b:,} B  ({fmt_bytes(comp_b)})   "
                f"|  ratio = {ratio:.2f}x",
                f"Encoder (mean ± std)    : {prof['enc_mean']:.2f} ± "
                f"{prof['enc_std']:.2f} ms   min {prof['enc_min']:.2f}  "
                f"max {prof['enc_max']:.2f}   over {profile_iters} iters",
                f"Decoder (mean ± std)    : {prof['dec_mean']:.2f} ± "
                f"{prof['dec_std']:.2f} ms   min {prof['dec_min']:.2f}  "
                f"max {prof['dec_max']:.2f}   over {profile_iters} iters",
                f"Quality / rate          : PSNR = {psnr:.2f} dB   |   λ = {lam}",
                f"(old naive formula gave : {naive_b:,} B — counts only y[0]+z[0])",
            ]

            title = (f"{dataset_name}/img{i} | rate={force_ind} | λ={lam} "
                     f"| ratio={ratio:.1f}x | PSNR={psnr:.2f} dB")

            out_path = subdir / f"rate{force_ind}_compare.png"
            save_side_by_side(original_pil, tensor_to_pil(x_hat),
                              title, stats_lines, out_path)

            print(f"   rate={force_ind} (λ={lam})  ratio={ratio:6.2f}x  "
                  f"PSNR={psnr:6.2f} dB  comp={comp_b:,} B")
            print(f"      streams: " +
                  " ".join(f"{lbl}={b}" for lbl, b in breakdown) +
                  f"  +header={header_b}")
            print(f"      Enc {prof['enc_mean']:7.2f} ± {prof['enc_std']:5.2f} ms  "
                  f"| Dec {prof['dec_mean']:7.2f} ± {prof['dec_std']:5.2f} ms")
            print(f"      old naive would have printed {naive_b:,} B")
            print(f"      -> {out_path.name}")


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
    ap.add_argument("--num-images",     type=int, default=5)
    ap.add_argument("--profile-iters",  type=int, default=20)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--model-size",     type=int, default=64, choices=[64, 128])
    ap.add_argument("--device",         type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir",        type=Path, default=Path("visualize_tcm"))
    ap.add_argument("--datasets", nargs="+",
                    default=["kodak", "tecnick", "cityscapes"],
                    choices=["kodak", "tecnick", "cityscapes"])
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    print(f"Device         : {device}")
    print(f"Model N        : {args.model_size}")
    print(f"Images/dataset : {args.num_images}")
    print(f"Profile iters  : {args.profile_iters}\n")

    net = LICTCMAdapter(args.ckpt_dir, device=device,
                        model_size=args.model_size)

    dir_map = {
        "kodak":      args.kodak_dir,
        "tecnick":    args.tecnick_dir,
        "cityscapes": args.cityscapes_dir,
    }

    for name in args.datasets:
        d = dir_map[name]
        if not d.exists():
            print(f"[skip] {name}: {d} not found"); continue
        process_dataset(net, name, d, args.num_images, args.seed,
                        device, args.out_dir, args.profile_iters)

    print(f"\nDone. Output in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

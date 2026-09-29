"""
compare_visualize_tcm.py
------------------------
For a single image, produce one side-by-side frame per rate point:
    Original | Compressed -> Decompressed
Each frame overlays size / ratio / latency / PSNR.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from lic_tcm_wrapper import LICTCMAdapter


# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
@torch.inference_mode()
def encode_decode(net, frame, force_ind):
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

    return x_hat, H * W * 3, compressed["packed_bytes"], enc_ms, dec_ms


# ------------------------------------------------------------------
def save_side_by_side(orig_pil, recon_pil, title, stats_lines, out_path,
                      header_h=125, gap=14, max_panel_width=1000):
    W, H = orig_pil.size
    if W > max_panel_width:
        scale = max_panel_width / W
        new_size = (int(W * scale), int(H * scale))
        orig_pil = orig_pil.resize(new_size, Image.LANCZOS)
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
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        font_title = font_label = font_stat = ImageFont.load_default()

    draw.text((12, 8), title, fill=(20, 20, 20), font=font_title)
    y = 38
    for line in stats_lines:
        draw.text((14, y), line, fill=(40, 40, 40), font=font_stat)
        y += 21

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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path, default=Path("../kodak/01.png"))
    ap.add_argument("--ckpt-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "checkpoints")
    ap.add_argument("--model-size", type=int, default=64, choices=[64, 128])
    ap.add_argument("--device", type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=Path("visualize_tcm"))
    args = ap.parse_args()

    if not args.img.exists():
        raise SystemExit(f"Image not found: {args.img}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    net = LICTCMAdapter(args.ckpt_dir, device=device,
                        model_size=args.model_size)

    frame = load_image(args.img, device)
    original_pil = Image.open(args.img).convert("RGB")
    H, W = original_pil.height, original_pil.width

    print(f"\nImage: {args.img}  ({W}x{H})\n")
    header = (f"{'rate':>4} | {'lambda':>8} | {'orig(B)':>10} | "
              f"{'comp(B)':>9} | {'ratio':>8} | "
              f"{'Enc(ms)':>9} | {'Dec(ms)':>9} | {'PSNR':>7}")
    print(header); print("-" * len(header))

    for force_ind in range(net.num_rates):
        lam = net.available_lambdas[force_ind]
        net.prepare_inference_(force_ind=force_ind)

        x_hat, orig_b, comp_b, enc_ms, dec_ms = encode_decode(
            net, frame, force_ind)
        ratio = orig_b / comp_b
        mse = F.mse_loss(
            ((x_hat + 1.0) * 0.5).clamp(0, 1),
            ((frame + 1.0) * 0.5).clamp(0, 1),
        ).item()
        psnr = -10.0 * np.log10(mse)

        print(f"{force_ind:>4} | {lam:>8.4f} | "
              f"{orig_b:>10,} | {comp_b:>9,} | {ratio:>8.2f} | "
              f"{enc_ms:>9.2f} | {dec_ms:>9.2f} | {psnr:>7.3f}")

        recon_pil = tensor_to_pil(x_hat)
        stats_lines = [
            f"Source image            : {args.img.name}",
            f"Original image (uint8)  : {orig_b:>10,} B ({fmt_bytes(orig_b)})",
            f"Compressed (bitstream)  : {comp_b:>10,} B ({fmt_bytes(comp_b)})",
            f"Compression ratio       : {ratio:>10.2f} x",
            f"Latency                 : Enc {enc_ms:.2f} ms   Dec {dec_ms:.2f} ms",
        ]
        title = (f"{args.img.stem} | rate={force_ind} | λ={lam} "
                 f"| PSNR={psnr:.2f} dB")

        out_path = args.out_dir / f"{args.img.stem}_rate{force_ind}_compare.png"
        save_side_by_side(original_pil, recon_pil, title, stats_lines, out_path)
        print(f"   saved -> {out_path}")

    print(f"\nAll comparisons in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

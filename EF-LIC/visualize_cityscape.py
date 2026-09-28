"""
visualize_cityscapes.py
-----------------------
For one Cityscapes image at each force_ind (0..4):
  - Encode + pack + unpack + decode
  - Save side-by-side:  [ Original | Compressed -> Decompressed ]
  - Overlay: original (uint8) size, compressed tensor size, ratio
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from EF_LIC import model
from test import (
    pack_inds,
    unpack_inds,
    replicate_pad,
    load_checkpoint,
    load_image,
)

FORCE_INDS = range(5)


# ------------------------------------------------------------------
# Size helpers
# ------------------------------------------------------------------
def raw_tensor_bytes(inds):
    total = inds["z_inds"].numel() * inds["z_inds"].element_size()
    for t in inds["y_inds"]:
        total += t.numel() * t.element_size()
    return int(total)


def fmt_bytes(n):
    if n >= 1024 * 1024:
        return f"{n/1024/1024:.2f} MB"
    if n >= 1024:
        return f"{n/1024:.1f} KB"
    return f"{n} B"


# ------------------------------------------------------------------
# Encode / decode
# ------------------------------------------------------------------
@torch.inference_mode()
def encode_decode(net, frame, force_ind):
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

    orig_u8 = H * W * 3
    comp    = raw_tensor_bytes(inds)

    return {
        "x_hat":   x_hat,
        "orig_u8": orig_u8,
        "comp":    comp,
        "enc_ms":  enc_ms,
        "dec_ms":  dec_ms,
    }


def tensor_to_pil(t):
    t = ((t.clamp(-1, 1) + 1.0) * 0.5).cpu()
    arr = (t.squeeze(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


# ------------------------------------------------------------------
# Side-by-side save
# ------------------------------------------------------------------
def save_side_by_side(original_pil, recon_pil, title, stats_lines,
                      out_path, header_h=125, gap=14,
                      max_panel_width=1000):
    # Downscale panels so the whole canvas isn't absurdly wide
    W, H = original_pil.size
    if W > max_panel_width:
        scale = max_panel_width / W
        new_size = (int(W * scale), int(H * scale))
        original_pil = original_pil.resize(new_size, Image.LANCZOS)
        recon_pil    = recon_pil.resize(new_size, Image.LANCZOS)
        W, H = original_pil.size

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

    canvas.paste(original_pil, (0, img_top))
    canvas.paste(recon_pil,    (W + gap, img_top))
    draw.line([(W + gap // 2, img_top), (W + gap // 2, img_top + H)],
              fill=(200, 200, 200), width=2)

    canvas.save(out_path)
    return out_path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=Path,
                    default=Path(
                        "/home/common/EF-LIC/datasets/cityscapes/"
                        "leftImg8bit/val/frankfurt/"
                        "frankfurt_000000_000294_leftImg8bit.png"))
    ap.add_argument("--ckpt-path", type=Path,
                    default=Path("ckpt/checkpoint.pth.tar"))
    ap.add_argument("--device", type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("visualize_cityscapes"))
    args = ap.parse_args()

    if not args.img.exists():
        raise SystemExit(f"Image not found: {args.img}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)

    frame        = load_image(args.img, device)
    original_pil = Image.open(args.img).convert("RGB")
    H, W         = original_pil.height, original_pil.width

    print(f"Image       : {args.img}")
    print(f"Resolution  : {W} x {H}\n")

    header = (f"{'rate':>4} | "
              f"{'orig uint8':>12} | {'comp tensor':>12} | "
              f"{'ratio':>8} | "
              f"{'Enc(ms)':>9} | {'Dec(ms)':>9} | {'PSNR':>7}")
    print(header)
    print("-" * len(header))

    for force_ind in FORCE_INDS:
        net.prepare_inference_(force_ind=force_ind)
        r = encode_decode(net, frame, force_ind)

        ratio = r["orig_u8"] / r["comp"]

        mse = F.mse_loss(
            ((r["x_hat"] + 1.0) * 0.5).clamp(0, 1),
            ((frame   + 1.0) * 0.5).clamp(0, 1),
        ).item()
        psnr = -10.0 * np.log10(mse)

        print(f"{force_ind:>4} | "
              f"{r['orig_u8']:>12,} | {r['comp']:>12,} | "
              f"{ratio:>8.2f} | "
              f"{r['enc_ms']:>9.2f} | {r['dec_ms']:>9.2f} | {psnr:>7.3f}")

        stats_lines = [
            f"Original image (uint8)     : {r['orig_u8']:>10,} B "
            f"({fmt_bytes(r['orig_u8'])})",
            f"Compressed tensor          : {r['comp']:>10,} B "
            f"({fmt_bytes(r['comp'])})",
            f"Compression ratio          : {ratio:>10.2f} x",
        ]

        title = (f"{args.img.stem} | rate={force_ind} | "
                 f"Enc={r['enc_ms']:.2f} ms  Dec={r['dec_ms']:.2f} ms  "
                 f"PSNR={psnr:.2f} dB")

        recon_pil = tensor_to_pil(r["x_hat"])
        out_path  = (args.out_dir /
                     f"{args.img.stem}_rate{force_ind}_compare.png")

        save_side_by_side(original_pil, recon_pil, title,
                          stats_lines, out_path)
        print(f"     saved -> {out_path}")

    print(f"\nAll comparisons saved to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

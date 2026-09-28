"""
yolos_batch_kodak.py
--------------------
For N Kodak images and ALL 5 rate points:
  - EF-LIC compress + decompress
  - YOLOS on original and reconstructed
  - Save ONLY the side-by-side frame

Output layout:
    compare_yolo_kodak/img1/rate0_side_by_side.png
    compare_yolo_kodak/img1/rate1_side_by_side.png
    ...
    compare_yolo_kodak/img1/rate4_side_by_side.png
    compare_yolo_kodak/img2/rate0_side_by_side.png
    ...
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import YolosImageProcessor, YolosForObjectDetection

from EF_LIC import model
from test import (
    pack_inds, unpack_inds, replicate_pad,
    load_checkpoint, load_image,
)

FORCE_INDS = range(5)


# ------------------------------------------------------------------
# EF-LIC helpers
# ------------------------------------------------------------------
@torch.inference_mode()
def eflic_encode_decode(net, frame, force_ind):
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

    def _raw_bytes(inds):
        t = inds["z_inds"].numel() * inds["z_inds"].element_size()
        for y in inds["y_inds"]:
            t += y.numel() * y.element_size()
        return int(t)

    return x_hat, _raw_bytes(inds), enc_ms, dec_ms


def tensor_to_pil(t):
    t = ((t.clamp(-1, 1) + 1.0) * 0.5).cpu()
    arr = (t.squeeze(0).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


# ------------------------------------------------------------------
# YOLOS
# ------------------------------------------------------------------
def run_yolos(yolos_model, yolos_processor, pil_img,
              threshold=0.9, device="cuda:0"):
    inputs = yolos_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = yolos_model(**inputs)
    target_sizes = torch.tensor([pil_img.size[::-1]]).to(device)
    results = yolos_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes)[0]

    dets = []
    for score, label, box in zip(results["scores"],
                                 results["labels"],
                                 results["boxes"]):
        dets.append((tuple(box.tolist()),
                     int(label),
                     yolos_model.config.id2label[int(label)],
                     float(score)))
    return dets


def draw_boxes(pil_img, dets, color, line_width=3, font_size=20):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), _, cls_name, conf in dets:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        label = f"{cls_name} {conf:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 2), label, fill=(255, 255, 255), font=font)
    return img


# ------------------------------------------------------------------
# Side-by-side builder
# ------------------------------------------------------------------
def build_frame(orig_pil, recon_pil, title, stats_lines, out_path,
                max_panel_width=1000, header_h=160, gap=14):
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
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_label = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_stat  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        font_title = font_label = font_stat = ImageFont.load_default()

    draw.text((14, 8), title, fill=(20, 20, 20), font=font_title)
    y = 40
    for line in stats_lines:
        draw.text((16, y), line, fill=(40, 40, 40), font=font_stat)
        y += 22

    img_top = header_h
    draw.text((14, img_top - 26), "Original + YOLOS",
              fill=(0, 110, 0), font=font_label)
    draw.text((W + gap + 14, img_top - 26),
              "Compressed -> Decompressed + YOLOS",
              fill=(150, 0, 0), font=font_label)

    canvas.paste(orig_pil, (0, img_top))
    canvas.paste(recon_pil, (W + gap, img_top))
    draw.line([(W + gap // 2, img_top), (W + gap // 2, img_top + H)],
              fill=(180, 180, 180), width=2)

    canvas.save(out_path)
    return out_path


# ------------------------------------------------------------------
# Image selection
# ------------------------------------------------------------------
def pick_images(root: Path, n: int, seed: int = 42):
    exts = {".png", ".jpg", ".jpeg"}
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in exts and p.is_file())
    if not files:
        raise SystemExit(f"No images found under {root}")
    if len(files) <= n:
        return files
    rng = random.Random(seed)
    return sorted(rng.sample(files, n), key=lambda p: p.name)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=Path, default=Path("kodak"))
    ap.add_argument("--num-images", type=int, default=10)
    ap.add_argument("--ckpt-path", type=Path,
                    default=Path("ckpt/checkpoint.pth.tar"))
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=Path("compare_yolo_kodak"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # ---- Load models once ----
    print("Loading EF-LIC ...")
    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)

    print("Loading YOLOS ...")
    yolos_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
    yolos_model = YolosForObjectDetection.from_pretrained(
        "hustvl/yolos-small").to(device).eval()

    images = pick_images(args.img_dir, args.num_images, seed=args.seed)
    print(f"\nSelected {len(images)} images:")
    for p in images:
        print(f"  {p.name}")
    print()

    # ---- Loop: image × rate ----
    for i, img_path in enumerate(images, 1):
        subdir = args.out_dir / f"img{i}"
        subdir.mkdir(parents=True, exist_ok=True)

        # Load once per image
        frame        = load_image(img_path, device)
        original_pil = Image.open(img_path).convert("RGB")
        H, W         = original_pil.height, original_pil.width
        orig_u8      = H * W * 3

        # YOLOS on original (independent of rate)
        dets_orig = run_yolos(yolos_model, yolos_processor, original_pil,
                              threshold=args.threshold, device=str(device))
        orig_boxed = draw_boxes(original_pil, dets_orig, color=(0, 140, 0))

        print(f"[{i}/{len(images)}] {img_path.name}  "
              f"({W}x{H})  (orig detections: {len(dets_orig)})")

        for force_ind in FORCE_INDS:
            net.prepare_inference_(force_ind=force_ind)

            x_hat, comp_bytes, enc_ms, dec_ms = eflic_encode_decode(
                net, frame, force_ind)
            recon_pil = tensor_to_pil(x_hat)
            ratio = orig_u8 / comp_bytes

            dets_recon = run_yolos(yolos_model, yolos_processor, recon_pil,
                                   threshold=args.threshold, device=str(device))
            recon_boxed = draw_boxes(recon_pil, dets_recon,
                                     color=(200, 30, 30))

            stats_lines = [
                f"Source image               : {img_path.name}",
                f"Original image (uint8)     : {orig_u8:>11,} B    "
                f"|   YOLOS detections: {len(dets_orig)}",
                f"Compressed tensor          : {comp_bytes:>11,} B    "
                f"|   YOLOS detections: {len(dets_recon)}",
                f"Compression ratio          : {ratio:>10.2f} x    "
                f"|   Rate point: {force_ind}",
                f"Latency                    : Enc {enc_ms:.2f} ms   "
                f"Dec {dec_ms:.2f} ms    |   YOLOS-Small (ViT)",
            ]

            title = (f"img{i}  |  {img_path.name[:60]}  |  rate={force_ind}  "
                     f"|  detections: orig={len(dets_orig)}  "
                     f"recon={len(dets_recon)}")

            out_path = subdir / f"rate{force_ind}_side_by_side.png"
            build_frame(orig_boxed, recon_boxed, title,
                        stats_lines, out_path)

            print(f"   rate={force_ind}  ratio={ratio:6.2f}x  "
                  f"dets: orig={len(dets_orig):>2}  "
                  f"recon={len(dets_recon):>2}  -> {out_path.name}")

    print(f"\nDone. Output: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

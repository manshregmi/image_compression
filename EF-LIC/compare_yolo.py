"""
yolos_compare.py
----------------
For a given image:
  1. Run EF-LIC to get the compressed -> decompressed image at a chosen rate
  2. Run YOLOS (ViT-based) on BOTH the original and reconstructed image
  3. Draw bounding boxes on each
  4. Save a single side-by-side frame with detection counts
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import YolosImageProcessor, YolosForObjectDetection

from EF_LIC import model
from test import (
    pack_inds,
    unpack_inds,
    replicate_pad,
    load_checkpoint,
    load_image,
)


# ------------------------------------------------------------------
# EF-LIC helpers (same as before)
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
# YOLOS helpers
# ------------------------------------------------------------------
def run_yolos(yolos_model, yolos_processor, pil_img, threshold=0.9, device="cuda:0"):
    """Returns a list of detections: (xyxy, class_id, class_name, conf)."""
    inputs = yolos_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = yolos_model(**inputs)

    # Post-process
    target_sizes = torch.tensor([pil_img.size[::-1]]).to(device)
    results = yolos_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    dets = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        dets.append((tuple(box), int(label), yolos_model.config.id2label[int(label)], float(score)))
    return dets


def draw_boxes(pil_img, dets, line_width=3, font_size=20):
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), cls_id, cls_name, conf in dets:
        color = (255, 0, 0)  # Red for YOLOS
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        label = f"{cls_name} {conf:.2f}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 2), label, fill=(255, 255, 255), font=font)
    return img


# ------------------------------------------------------------------
# Build single side-by-side frame
# ------------------------------------------------------------------
def build_frame(orig_pil, recon_pil, title, stats_lines, out_path,
                max_panel_width=1000, header_h=160, gap=14):
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
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_label = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_stat = ImageFont.truetype(
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
    ap.add_argument("--rate", type=int, default=0,
                    help="EF-LIC rate point (force_ind): 0..4")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="YOLOS confidence threshold")
    ap.add_argument("--device", type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=Path("yolos_compare"))
    args = ap.parse_args()

    if not args.img.exists():
        raise SystemExit(f"Image not found: {args.img}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # ---- Load EF-LIC ----
    print("Loading EF-LIC ...")
    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    net.prepare_inference_(force_ind=args.rate)

    # ---- Load YOLOS ----
    print("Loading YOLOS (ViT-based) ...")
    yolos_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
    yolos_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small").to(device).eval()

    # ---- Load image ----
    frame = load_image(args.img, device)
    original_pil = Image.open(args.img).convert("RGB")
    H, W = original_pil.height, original_pil.width

    print(f"\nImage       : {args.img}")
    print(f"Resolution  : {W} x {H}")
    print(f"Rate point  : {args.rate}\n")

    # ---- EF-LIC compress + decompress ----
    print("Running EF-LIC compress + decompress ...")
    x_hat, comp_bytes, enc_ms, dec_ms = eflic_encode_decode(net, frame, args.rate)
    recon_pil = tensor_to_pil(x_hat)
    orig_u8 = H * W * 3
    ratio = orig_u8 / comp_bytes

    # ---- YOLOS on original ----
    print("Running YOLOS on original ...")
    dets_orig = run_yolos(yolos_model, yolos_processor, original_pil, threshold=args.threshold, device=str(device))
    orig_boxed = draw_boxes(original_pil, dets_orig)

    # ---- YOLOS on reconstructed ----
    print("Running YOLOS on reconstructed ...")
    dets_recon = run_yolos(yolos_model, yolos_processor, recon_pil, threshold=args.threshold, device=str(device))
    recon_boxed = draw_boxes(recon_pil, dets_recon)

    # ---- Report ----
    print(f"\nEF-LIC   : orig={orig_u8:,} B | comp={comp_bytes:,} B | ratio={ratio:.2f}x | Enc={enc_ms:.2f} ms | Dec={dec_ms:.2f} ms")
    print(f"YOLOS orig: {len(dets_orig)} detections")
    print(f"YOLOS recon: {len(dets_recon)} detections")

    def class_counts(dets):
        c = {}
        for _, _, name, _ in dets:
            c[name] = c.get(name, 0) + 1
        return c

    print(f"\nClasses (original)     : {class_counts(dets_orig)}")
    print(f"Classes (reconstructed): {class_counts(dets_recon)}")

    # ---- Build side-by-side ----
    stats_lines = [
        f"Original image (uint8)     : {orig_u8:>11,} B    |   YOLOS detections: {len(dets_orig)}",
        f"Compressed tensor          : {comp_bytes:>11,} B    |   YOLOS detections: {len(dets_recon)}",
        f"Compression ratio          : {ratio:>10.2f} x    |   Rate point: {args.rate}",
        f"Latency                    : Enc {enc_ms:.2f} ms   Dec {dec_ms:.2f} ms    |   Model: YOLOS-Small (ViT)",
    ]

    title = (f"{args.img.stem[:60]}  |  rate={args.rate}  |  detections: orig={len(dets_orig)}  recon={len(dets_recon)}")

    out_path = (args.out_dir / f"{args.img.stem}_rate{args.rate}_yolos.png")
    build_frame(orig_boxed, recon_boxed, title, stats_lines, out_path)
    print(f"\nSaved side-by-side frame -> {out_path}")


if __name__ == "__main__":
    main()

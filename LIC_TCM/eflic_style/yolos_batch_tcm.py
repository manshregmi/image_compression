"""
yolos_batch_tcm.py
------------------
For N images per dataset and every TCM rate:
  - EF-LIC-style compress + decompress via LICTCMAdapter
  - YOLOS on both original and reconstructed images
  - Save only side-by-side frames

Output:
    compare_yolo_tcm/<dataset>/img{N}/rate{R}_side_by_side.png
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import YolosImageProcessor, YolosForObjectDetection

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from lic_tcm_wrapper import LICTCMAdapter


# ------------------------------------------------------------------
def list_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in exts and p.is_file())
    if not files:
        raise RuntimeError(f"No images found in {root}")
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
def run_yolos(yolos_model, yolos_processor, pil_img, threshold, device):
    inputs = yolos_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = yolos_model(**inputs)
    target_sizes = torch.tensor([pil_img.size[::-1]]).to(device)
    results = yolos_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes)[0]
    dets = []
    for s, l, b in zip(results["scores"], results["labels"], results["boxes"]):
        dets.append((tuple(b.tolist()), int(l),
                     yolos_model.config.id2label[int(l)], float(s)))
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


# ------------------------------------------------------------------
def process_dataset(net, dataset_name, img_dir, num_images, seed,
                    yolos_model, yolos_processor,
                    threshold, device, out_root):
    images = pick_images(img_dir, num_images, seed=seed)
    print(f"\n{'='*70}\n{dataset_name}: {len(images)} images\n{'='*70}")

    for i, img_path in enumerate(images, 1):
        subdir = out_root / dataset_name / f"img{i}"
        subdir.mkdir(parents=True, exist_ok=True)

        frame = load_image(img_path, device)
        original_pil = Image.open(img_path).convert("RGB")
        H, W = original_pil.height, original_pil.width

        dets_orig = run_yolos(yolos_model, yolos_processor,
                              original_pil, threshold, str(device))
        orig_boxed = draw_boxes(original_pil, dets_orig, color=(0, 140, 0))

        print(f"[{dataset_name}] [{i}/{len(images)}] {img_path.name}  "
              f"({W}x{H})  dets orig={len(dets_orig)}")

        for force_ind in range(net.num_rates):
            lam = net.available_lambdas[force_ind]
            net.prepare_inference_(force_ind=force_ind)

            x_hat, orig_b, comp_b, enc_ms, dec_ms = encode_decode(
                net, frame, force_ind)
            recon_pil = tensor_to_pil(x_hat)
            ratio = orig_b / comp_b

            dets_recon = run_yolos(yolos_model, yolos_processor,
                                   recon_pil, threshold, str(device))
            recon_boxed = draw_boxes(recon_pil, dets_recon,
                                     color=(200, 30, 30))
            det_loss = abs(len(dets_orig) - len(dets_recon))

            stats_lines = [
                f"Source image          : {img_path.name[:70]}",
                f"Original (uint8)      : {orig_b:>10,} B  "
                f"| YOLOS detections: {len(dets_orig)}",
                f"Compressed bitstream  : {comp_b:>10,} B  "
                f"| YOLOS detections: {len(dets_recon)}",
                f"Compression ratio     : {ratio:>10.2f} x   "
                f"| detection loss = {det_loss}",
                f"Latency               : Enc {enc_ms:.2f} ms   "
                f"Dec {dec_ms:.2f} ms   | λ = {lam}",
            ]
            title = (f"{dataset_name}/img{i} | {img_path.name[:50]} | "
                     f"rate={force_ind} | λ={lam} | dets orig={len(dets_orig)} "
                     f"recon={len(dets_recon)} loss={det_loss}")

            out_path = subdir / f"rate{force_ind}_side_by_side.png"
            build_frame(orig_boxed, recon_boxed, title, stats_lines, out_path)
            print(f"   rate={force_ind} (λ={lam})  ratio={ratio:6.2f}x  "
                  f"dets {len(dets_orig)}->{len(dets_recon)} "
                  f"(loss={det_loss})  -> {out_path.name}")


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
    ap.add_argument("--yolos-model",    type=str, default="hustvl/yolos-small")
    ap.add_argument("--threshold",      type=float, default=0.5)
    ap.add_argument("--num-images",     type=int, default=10)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--model-size",     type=int, default=64, choices=[64, 128])
    ap.add_argument("--device",         type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir",        type=Path, default=Path("compare_yolo_tcm"))
    ap.add_argument("--datasets", nargs="+",
                    default=["kodak", "tecnick", "cityscapes"],
                    choices=["kodak", "tecnick", "cityscapes"])
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    print(f"Device  : {device}")
    print(f"Model N : {args.model_size}")
    print(f"YOLOS   : {args.yolos_model}  (threshold={args.threshold})\n")

    net = LICTCMAdapter(args.ckpt_dir, device=device,
                        model_size=args.model_size)
    yolos_processor = YolosImageProcessor.from_pretrained(args.yolos_model)
    yolos_model     = YolosForObjectDetection.from_pretrained(
        args.yolos_model).to(device).eval()

    dir_map = {
        "kodak":      args.kodak_dir,
        "tecnick":    args.tecnick_dir,
        "cityscapes": args.cityscapes_dir,
    }

    for name in args.datasets:
        d = dir_map[name]
        if not d.exists():
            print(f"[skip] {name}: {d} not found"); continue
        process_dataset(
            net, name, d, args.num_images, args.seed,
            yolos_model, yolos_processor,
            args.threshold, device, args.out_dir,
        )

    print(f"\nDone. Output in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

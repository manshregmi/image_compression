"""
evaluate_datasets.py  —  EF-LIC tensor-size evaluation
                       with optional TensorRT / torch.compile + 1000-iter bench
---------------------------------------------------------------------------------------
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
    pack_inds, unpack_inds, replicate_pad,
    load_checkpoint, load_image,
)

warnings.filterwarnings("ignore")

FORCE_INDS = range(5)
PAD_MULTIPLE = 64

# ------------------------------------------------------------------
# Benchmark settings
# ------------------------------------------------------------------
WARMUP_ITERS   = 100     # discarded
MEASURE_ITERS  = 1000    # total timed iterations
AVERAGE_LAST_N = 900     # of the MEASURE_ITERS, average these


# ------------------------------------------------------------------
# Size helpers
# ------------------------------------------------------------------
def raw_tensor_bytes(inds):
    total = inds["z_inds"].numel() * inds["z_inds"].element_size()
    for t in inds["y_inds"]:
        total += t.numel() * t.element_size()
    return int(total)


def list_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in exts and p.is_file())
    if not files:
        raise RuntimeError(f"No images found in {root.resolve()}")
    return files


# ==================================================================
# Accelerator setup
# ==================================================================
def try_torch_compile(net, device):
    """torch.compile with max-autotune — closest thing to TensorRT that
    works with arbitrary PyTorch modules (RVQ, autoregressive transforms).
    """
    if not hasattr(torch, "compile"):
        print("[accel] torch.compile not available (PyTorch < 2.0)")
        return net

    try:
        print("[accel] Applying torch.compile(mode='max-autotune-no-cudagraphs')...")
        # Note: dynamic=True because our input sizes vary per dataset
        net = torch.compile(net, mode="max-autotune-no-cudagraphs",
                            dynamic=True, fullgraph=False)
        print("[accel] torch.compile succeeded")
        return net
    except Exception as e:
        print(f"[accel] torch.compile failed: {e}")
        return net


def try_onnx_export(net, sample_frame, out_path):
    """Attempt ONNX export for a TensorRT pipeline. Returns True on success.

    EF-LIC contains custom ops (RVQ nearest-neighbour, context transforms),
    so this may fail. Even if it succeeds, TRT will need plugin implementations
    for some ops.
    """
    try:
        print("[accel] Attempting ONNX export...")
        net.eval()
        dummy = sample_frame
        torch.onnx.export(
            net,
            dummy,
            str(out_path),
            input_names=["image"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={"image": {2: "H", 3: "W"},
                          "output": {2: "H", 3: "W"}},
        )
        print(f"[accel] ONNX exported -> {out_path}")
        return True
    except Exception as e:
        print(f"[accel] ONNX export failed: {e}")
        print("[accel] Falling back to torch.compile / eager.")
        return False


def build_trt_engine(onnx_path, engine_path, fp16=True):
    """Build a TensorRT engine from ONNX. Requires `tensorrt` + `polygraphy`
    installed and matching CUDA version.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[trt] tensorrt not installed; skipping engine build")
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[trt] ONNX parsing failed:")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print("[trt] Building engine (this can take a few minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[trt] Engine build failed")
        return None

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[trt] Engine saved -> {engine_path}")
    return engine_path


# ==================================================================
# Timed benchmark
# ==================================================================
def benchmark_repeated(fn,
                       warmup=WARMUP_ITERS,
                       iters=MEASURE_ITERS,
                       average_last=AVERAGE_LAST_N,
                       device="cuda"):
    """Run fn() warmup times to stabilise, then iters timed runs.
    Returns (mean_ms, std_ms, min_ms, p50_ms, p99_ms) computed over the
    last `average_last` of the timed runs.
    """
    is_cuda = device == "cuda"

    # ---- warm-up (discarded) ----
    for _ in range(warmup):
        fn()

    # ---- timed iterations ----
    times = []
    if is_cuda:
        torch.cuda.synchronize()

    if is_cuda:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start_ev.record()
            fn()
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)

    times = np.asarray(times)
    # Average over the LAST `average_last` samples
    tail = times[-average_last:]
    return (
        float(tail.mean()),
        float(tail.std()),
        float(tail.min()),
        float(np.percentile(tail, 50)),
        float(np.percentile(tail, 99)),
    )


# ==================================================================
# Single-image forward (returns sizes as before)
# ==================================================================
@torch.inference_mode()
def forward_once(net, padded, force_ind):
    inds = net.compress(padded, force_ind=force_ind)
    payload, meta, total_valid_bits = pack_inds(net, inds)
    inds_dec = unpack_inds(payload, meta, total_valid_bits, padded.device)
    x_hat = net.decompress(inds_dec, force_ind=force_ind)
    return x_hat, inds


def sizes_from_inds(inds, H, W):
    orig = H * W * 3
    comp = raw_tensor_bytes(inds)
    return orig, comp, orig / comp


def psnr_of(x_hat, frame, H, W):
    x_hat = x_hat[:, :, :H, :W]
    mse = F.mse_loss(
        ((x_hat + 1.0) * 0.5).clamp(0, 1),
        ((frame  + 1.0) * 0.5).clamp(0, 1),
    ).item()
    return -10.0 * math.log10(mse)


# ==================================================================
# Evaluate one dataset (with benchmark)
# ==================================================================
def evaluate_dataset(net, dataset_name, data_dir, device, out_dir,
                     max_images=None, do_bench=False,
                     warmup=WARMUP_ITERS, iters=MEASURE_ITERS,
                     avg_last=AVERAGE_LAST_N):
    images = list_images(data_dir)
    if max_images is not None and len(images) > max_images:
        images = images[:max_images]

    print(f"\n{'='*90}")
    print(f"Dataset: {dataset_name}   ({len(images)} images)   dir={data_dir}")
    print(f"{'='*90}")
    if do_bench:
        print(f"Benchmark: warmup={warmup}  iters={iters}  "
              f"average last {avg_last}\n")

    per_image_rows = []
    summary_rows   = []

    for force_ind in FORCE_INDS:
        net.prepare_inference_(force_ind=force_ind)

        # warm-up (also primes any compiled kernels)
        warm = load_image(images[0], device)
        warm_pad = replicate_pad(warm, warm.shape[2], warm.shape[3])
        _ = forward_once(net, warm_pad, force_ind)

        metrics = []
        print(f"\n--- force_ind = {force_ind} ---")
        if do_bench:
            header = (f"{'#':>3} | {'file':<35} | {'orig(B)':>10} | "
                      f"{'comp(B)':>9} | {'ratio':>7} | "
                      f"{'Enc_mean':>9} | {'Enc_std':>8} | "
                      f"{'Dec_mean':>9} | {'Dec_std':>8} | {'PSNR':>7}")
        else:
            header = (f"{'#':>3} | {'file':<35} | {'orig(B)':>10} | "
                      f"{'comp(B)':>9} | {'ratio':>7} | "
                      f"{'Enc(ms)':>9} | {'Dec(ms)':>9} | {'PSNR':>7}")
        print(header)
        print("-" * len(header))

        for idx, path in enumerate(images, 1):
            frame = load_image(path, device)
            H, W = frame.shape[2], frame.shape[3]
            padded = replicate_pad(frame, H, W)

            # --- first pass: sizes + PSNR ---
            x_hat, inds = forward_once(net, padded, force_ind)
            orig_b, comp_b, ratio = sizes_from_inds(inds, H, W)
            psnr = psnr_of(x_hat, frame, H, W)

            # --- timing ---
            if do_bench:
                def _enc():
                    return net.compress(padded, force_ind=force_ind)

                def _dec():
                    # use the latest inds from a fresh encode
                    ii = net.compress(padded, force_ind=force_ind)
                    payload, meta, tbits = pack_inds(net, ii)
                    d = unpack_inds(payload, meta, tbits, padded.device)
                    return net.decompress(d, force_ind=force_ind)

                enc_mean, enc_std, enc_min, enc_p50, enc_p99 = benchmark_repeated(
                    _enc, warmup=warmup, iters=iters,
                    average_last=avg_last, device=str(device))
                dec_mean, dec_std, dec_min, dec_p50, dec_p99 = benchmark_repeated(
                    _dec, warmup=warmup, iters=iters,
                    average_last=avg_last, device=str(device))
                enc_val, dec_val = enc_mean, dec_mean
            else:
                # single timed pass
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = net.compress(padded, force_ind=force_ind)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                enc_val = (time.perf_counter() - t0) * 1000.0

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                payload, meta, tbits = pack_inds(net, inds)
                d = unpack_inds(payload, meta, tbits, padded.device)
                _ = net.decompress(d, force_ind=force_ind)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dec_val = (time.perf_counter() - t0) * 1000.0
                enc_std = dec_std = 0.0

            row = {
                "dataset":     dataset_name,
                "rate":        force_ind,
                "image":       path.name,
                "H":           H,
                "W":           W,
                "orig_bytes":  orig_b,
                "comp_bytes":  comp_b,
                "ratio":       round(ratio, 4),
                "enc_ms":      round(enc_val, 4),
                "enc_std_ms":  round(enc_std, 4) if do_bench else 0.0,
                "dec_ms":      round(dec_val, 4),
                "dec_std_ms":  round(dec_std, 4) if do_bench else 0.0,
                "psnr_db":     round(psnr, 4),
            }
            per_image_rows.append(row)
            metrics.append(row)

            if do_bench:
                print(f"{idx:>3} | {path.name[:35]:<35} | "
                      f"{orig_b:>10,} | {comp_b:>9,} | {ratio:>7.2f} | "
                      f"{enc_val:>9.3f} | {enc_std:>8.3f} | "
                      f"{dec_val:>9.3f} | {dec_std:>8.3f} | {psnr:>7.3f}")
            else:
                print(f"{idx:>3} | {path.name[:35]:<35} | "
                      f"{orig_b:>10,} | {comp_b:>9,} | {ratio:>7.2f} | "
                      f"{enc_val:>9.2f} | {dec_val:>9.2f} | {psnr:>7.3f}")

        # ---- averages ----
        def m(k): return float(np.mean([x[k] for x in metrics]))
        summary_rows.append({
            "dataset":       dataset_name,
            "rate":          force_ind,
            "num_images":    len(metrics),
            "orig_bytes":    round(m("orig_bytes"), 2),
            "comp_bytes":    round(m("comp_bytes"), 2),
            "ratio":         round(m("ratio"), 4),
            "enc_ms":        round(m("enc_ms"), 4),
            "enc_std_ms":    round(m("enc_std_ms"), 4),
            "dec_ms":        round(m("dec_ms"), 4),
            "dec_std_ms":    round(m("dec_std_ms"), 4),
            "psnr_db":       round(m("psnr_db"), 4),
        })
        print(f"\nAVG rate={force_ind} | ratio={m('ratio'):.2f}x | "
              f"Enc={m('enc_ms'):.3f} ms | Dec={m('dec_ms'):.3f} ms | "
              f"PSNR={m('psnr_db'):.3f} dB\n")

    # ---- write CSVs ----
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = out_dir / f"{dataset_name}_per_image.csv"
    summary_csv   = out_dir / f"{dataset_name}_summary.csv"

    with open(per_image_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader(); w.writerows(per_image_rows)

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    print(f"Saved -> {per_image_csv}")
    print(f"Saved -> {summary_csv}")
    return summary_rows


# ==================================================================
# Main
# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kodak-dir",      type=Path, default=Path("kodak"))
    ap.add_argument("--tecnick-dir",    type=Path, default=Path("tecnick_flat"))
    ap.add_argument("--cityscapes-dir", type=Path,
                    default=Path("/home/common/EF-LIC/datasets/cityscapes/"
                                 "leftImg8bit/val/frankfurt"))
    ap.add_argument("--ckpt-path",      type=Path,
                    default=Path("ckpt/checkpoint.pth.tar"))
    ap.add_argument("--device",         type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir",        type=Path, default=Path("results"))
    ap.add_argument("--datasets",       type=str, nargs="+",
                    default=["kodak", "tecnick", "cityscapes"],
                    choices=["kodak", "tecnick", "cityscapes"])
    ap.add_argument("--max-cityscapes", type=int, default=None)
    ap.add_argument("--max-images",     type=int, default=None,
                    help="Cap ALL datasets at N images (useful for bench).")

    # ---- accelerator ----
    ap.add_argument("--accelerator", type=str, default="none",
                    choices=["none", "torch_compile", "onnx"],
                    help="none | torch_compile | onnx (attempt TRT build)")
    ap.add_argument("--onnx-path",   type=Path, default=Path("eflic.onnx"))
    ap.add_argument("--engine-path", type=Path, default=Path("eflic.trt"))

    # ---- benchmark ----
    ap.add_argument("--bench", type=int, default=0,
                    help="1 to enable 1000-iteration benchmark per image")
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    ap.add_argument("--iters",  type=int, default=MEASURE_ITERS)
    ap.add_argument("--avg-last", type=int, default=AVERAGE_LAST_N)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    print(f"Device      : {device}")
    print(f"Accelerator : {args.accelerator}")
    print(f"Benchmark   : {'on' if args.bench else 'off'}")

    # ---- load model ----
    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    print(f"Checkpoint  : {args.ckpt_path}")

    # ---- optional accelerator ----
    if args.accelerator == "torch_compile":
        net = try_torch_compile(net, device)

    elif args.accelerator == "onnx":
        # Need a dummy input for ONNX export — use first Kodak image
        sample = None
        for cand in [args.kodak_dir, args.tecnick_dir, args.cityscapes_dir]:
            if cand.exists():
                imgs = list_images(cand)
                sample = load_image(imgs[0], device)
                break
        if sample is None:
            print("[accel] No dataset found for ONNX dummy input; skipping")
        else:
            ok = try_onnx_export(net, sample, args.onnx_path)
            if ok:
                build_trt_engine(args.onnx_path, args.engine_path)
                print("[accel] NOTE: this script still runs the PyTorch net "
                      "for evaluation. To use the TRT engine, load "
                      f"{args.engine_path} in your inference code.")

    # ---- dataset loop ----
    dir_map = {
        "kodak":      args.kodak_dir,
        "tecnick":    args.tecnick_dir,
        "cityscapes": args.cityscapes_dir,
    }
    all_summaries = []
    for name in args.datasets:
        d = dir_map[name]
        if not d.exists():
            print(f"[skip] {name}: {d} not found"); continue

        cap = args.max_images
        if name == "cityscapes" and args.max_cityscapes is not None:
            cap = args.max_cityscapes

        summary = evaluate_dataset(
            net, name, d, device, args.out_dir,
            max_images=cap,
            do_bench=bool(args.bench),
            warmup=args.warmup,
            iters=args.iters,
            avg_last=args.avg_last,
        )
        all_summaries.extend(summary)

    # ---- combined CSV + table ----
    if all_summaries:
        combined_csv = args.out_dir / "combined_summary.csv"
        with open(combined_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader(); w.writerows(all_summaries)
        print(f"\nSaved combined summary -> {combined_csv}")

    print(f"\n{'='*120}")
    print(f"{'FINAL SUMMARY':^120}")
    print(f"{'='*120}")
    hdr = (f"{'dataset':<12} | {'rate':>4} | {'N':>5} | "
           f"{'orig(B)':>12} | {'comp(B)':>12} | {'ratio':>8} | "
           f"{'Enc_mean':>9} | {'Enc_std':>8} | "
           f"{'Dec_mean':>9} | {'Dec_std':>8} | {'PSNR':>8}")
    print(hdr); print("-" * len(hdr))
    for r in all_summaries:
        print(f"{r['dataset']:<12} | {r['rate']:>4} | {r['num_images']:>5} | "
              f"{r['orig_bytes']:>12,.0f} | {r['comp_bytes']:>12,.0f} | "
              f"{r['ratio']:>8.2f} | "
              f"{r['enc_ms']:>9.3f} | {r['enc_std_ms']:>8.3f} | "
              f"{r['dec_ms']:>9.3f} | {r['dec_std_ms']:>8.3f} | "
              f"{r['psnr_db']:>8.3f}")


if __name__ == "__main__":
    main()"""
evaluate_datasets.py  —  EF-LIC tensor-size evaluation
                       with optional TensorRT / torch.compile + 1000-iter bench
---------------------------------------------------------------------------------------
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
    pack_inds, unpack_inds, replicate_pad,
    load_checkpoint, load_image,
)

warnings.filterwarnings("ignore")

FORCE_INDS = range(5)
PAD_MULTIPLE = 64

# ------------------------------------------------------------------
# Benchmark settings
# ------------------------------------------------------------------
WARMUP_ITERS   = 100     # discarded
MEASURE_ITERS  = 1000    # total timed iterations
AVERAGE_LAST_N = 900     # of the MEASURE_ITERS, average these


# ------------------------------------------------------------------
# Size helpers
# ------------------------------------------------------------------
def raw_tensor_bytes(inds):
    total = inds["z_inds"].numel() * inds["z_inds"].element_size()
    for t in inds["y_inds"]:
        total += t.numel() * t.element_size()
    return int(total)


def list_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted(p for p in root.rglob("*")
                   if p.suffix.lower() in exts and p.is_file())
    if not files:
        raise RuntimeError(f"No images found in {root.resolve()}")
    return files


# ==================================================================
# Accelerator setup
# ==================================================================
def try_torch_compile(net, device):
    """torch.compile with max-autotune — closest thing to TensorRT that
    works with arbitrary PyTorch modules (RVQ, autoregressive transforms).
    """
    if not hasattr(torch, "compile"):
        print("[accel] torch.compile not available (PyTorch < 2.0)")
        return net

    try:
        print("[accel] Applying torch.compile(mode='max-autotune-no-cudagraphs')...")
        # Note: dynamic=True because our input sizes vary per dataset
        net = torch.compile(net, mode="max-autotune-no-cudagraphs",
                            dynamic=True, fullgraph=False)
        print("[accel] torch.compile succeeded")
        return net
    except Exception as e:
        print(f"[accel] torch.compile failed: {e}")
        return net


def try_onnx_export(net, sample_frame, out_path):
    """Attempt ONNX export for a TensorRT pipeline. Returns True on success.

    EF-LIC contains custom ops (RVQ nearest-neighbour, context transforms),
    so this may fail. Even if it succeeds, TRT will need plugin implementations
    for some ops.
    """
    try:
        print("[accel] Attempting ONNX export...")
        net.eval()
        dummy = sample_frame
        torch.onnx.export(
            net,
            dummy,
            str(out_path),
            input_names=["image"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes={"image": {2: "H", 3: "W"},
                          "output": {2: "H", 3: "W"}},
        )
        print(f"[accel] ONNX exported -> {out_path}")
        return True
    except Exception as e:
        print(f"[accel] ONNX export failed: {e}")
        print("[accel] Falling back to torch.compile / eager.")
        return False


def build_trt_engine(onnx_path, engine_path, fp16=True):
    """Build a TensorRT engine from ONNX. Requires `tensorrt` + `polygraphy`
    installed and matching CUDA version.
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[trt] tensorrt not installed; skipping engine build")
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[trt] ONNX parsing failed:")
            for i in range(parser.num_errors):
                print(f"  {parser.get_error(i)}")
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print("[trt] Building engine (this can take a few minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[trt] Engine build failed")
        return None

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[trt] Engine saved -> {engine_path}")
    return engine_path


# ==================================================================
# Timed benchmark
# ==================================================================
def benchmark_repeated(fn,
                       warmup=WARMUP_ITERS,
                       iters=MEASURE_ITERS,
                       average_last=AVERAGE_LAST_N,
                       device="cuda"):
    """Run fn() warmup times to stabilise, then iters timed runs.
    Returns (mean_ms, std_ms, min_ms, p50_ms, p99_ms) computed over the
    last `average_last` of the timed runs.
    """
    is_cuda = device == "cuda"

    # ---- warm-up (discarded) ----
    for _ in range(warmup):
        fn()

    # ---- timed iterations ----
    times = []
    if is_cuda:
        torch.cuda.synchronize()

    if is_cuda:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start_ev.record()
            fn()
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)

    times = np.asarray(times)
    # Average over the LAST `average_last` samples
    tail = times[-average_last:]
    return (
        float(tail.mean()),
        float(tail.std()),
        float(tail.min()),
        float(np.percentile(tail, 50)),
        float(np.percentile(tail, 99)),
    )


# ==================================================================
# Single-image forward (returns sizes as before)
# ==================================================================
@torch.inference_mode()
def forward_once(net, padded, force_ind):
    inds = net.compress(padded, force_ind=force_ind)
    payload, meta, total_valid_bits = pack_inds(net, inds)
    inds_dec = unpack_inds(payload, meta, total_valid_bits, padded.device)
    x_hat = net.decompress(inds_dec, force_ind=force_ind)
    return x_hat, inds


def sizes_from_inds(inds, H, W):
    orig = H * W * 3
    comp = raw_tensor_bytes(inds)
    return orig, comp, orig / comp


def psnr_of(x_hat, frame, H, W):
    x_hat = x_hat[:, :, :H, :W]
    mse = F.mse_loss(
        ((x_hat + 1.0) * 0.5).clamp(0, 1),
        ((frame  + 1.0) * 0.5).clamp(0, 1),
    ).item()
    return -10.0 * math.log10(mse)


# ==================================================================
# Evaluate one dataset (with benchmark)
# ==================================================================
def evaluate_dataset(net, dataset_name, data_dir, device, out_dir,
                     max_images=None, do_bench=False,
                     warmup=WARMUP_ITERS, iters=MEASURE_ITERS,
                     avg_last=AVERAGE_LAST_N):
    images = list_images(data_dir)
    if max_images is not None and len(images) > max_images:
        images = images[:max_images]

    print(f"\n{'='*90}")
    print(f"Dataset: {dataset_name}   ({len(images)} images)   dir={data_dir}")
    print(f"{'='*90}")
    if do_bench:
        print(f"Benchmark: warmup={warmup}  iters={iters}  "
              f"average last {avg_last}\n")

    per_image_rows = []
    summary_rows   = []

    for force_ind in FORCE_INDS:
        net.prepare_inference_(force_ind=force_ind)

        # warm-up (also primes any compiled kernels)
        warm = load_image(images[0], device)
        warm_pad = replicate_pad(warm, warm.shape[2], warm.shape[3])
        _ = forward_once(net, warm_pad, force_ind)

        metrics = []
        print(f"\n--- force_ind = {force_ind} ---")
        if do_bench:
            header = (f"{'#':>3} | {'file':<35} | {'orig(B)':>10} | "
                      f"{'comp(B)':>9} | {'ratio':>7} | "
                      f"{'Enc_mean':>9} | {'Enc_std':>8} | "
                      f"{'Dec_mean':>9} | {'Dec_std':>8} | {'PSNR':>7}")
        else:
            header = (f"{'#':>3} | {'file':<35} | {'orig(B)':>10} | "
                      f"{'comp(B)':>9} | {'ratio':>7} | "
                      f"{'Enc(ms)':>9} | {'Dec(ms)':>9} | {'PSNR':>7}")
        print(header)
        print("-" * len(header))

        for idx, path in enumerate(images, 1):
            frame = load_image(path, device)
            H, W = frame.shape[2], frame.shape[3]
            padded = replicate_pad(frame, H, W)

            # --- first pass: sizes + PSNR ---
            x_hat, inds = forward_once(net, padded, force_ind)
            orig_b, comp_b, ratio = sizes_from_inds(inds, H, W)
            psnr = psnr_of(x_hat, frame, H, W)

            # --- timing ---
            if do_bench:
                def _enc():
                    return net.compress(padded, force_ind=force_ind)

                def _dec():
                    # use the latest inds from a fresh encode
                    ii = net.compress(padded, force_ind=force_ind)
                    payload, meta, tbits = pack_inds(net, ii)
                    d = unpack_inds(payload, meta, tbits, padded.device)
                    return net.decompress(d, force_ind=force_ind)

                enc_mean, enc_std, enc_min, enc_p50, enc_p99 = benchmark_repeated(
                    _enc, warmup=warmup, iters=iters,
                    average_last=avg_last, device=str(device))
                dec_mean, dec_std, dec_min, dec_p50, dec_p99 = benchmark_repeated(
                    _dec, warmup=warmup, iters=iters,
                    average_last=avg_last, device=str(device))
                enc_val, dec_val = enc_mean, dec_mean
            else:
                # single timed pass
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = net.compress(padded, force_ind=force_ind)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                enc_val = (time.perf_counter() - t0) * 1000.0

                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                payload, meta, tbits = pack_inds(net, inds)
                d = unpack_inds(payload, meta, tbits, padded.device)
                _ = net.decompress(d, force_ind=force_ind)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dec_val = (time.perf_counter() - t0) * 1000.0
                enc_std = dec_std = 0.0

            row = {
                "dataset":     dataset_name,
                "rate":        force_ind,
                "image":       path.name,
                "H":           H,
                "W":           W,
                "orig_bytes":  orig_b,
                "comp_bytes":  comp_b,
                "ratio":       round(ratio, 4),
                "enc_ms":      round(enc_val, 4),
                "enc_std_ms":  round(enc_std, 4) if do_bench else 0.0,
                "dec_ms":      round(dec_val, 4),
                "dec_std_ms":  round(dec_std, 4) if do_bench else 0.0,
                "psnr_db":     round(psnr, 4),
            }
            per_image_rows.append(row)
            metrics.append(row)

            if do_bench:
                print(f"{idx:>3} | {path.name[:35]:<35} | "
                      f"{orig_b:>10,} | {comp_b:>9,} | {ratio:>7.2f} | "
                      f"{enc_val:>9.3f} | {enc_std:>8.3f} | "
                      f"{dec_val:>9.3f} | {dec_std:>8.3f} | {psnr:>7.3f}")
            else:
                print(f"{idx:>3} | {path.name[:35]:<35} | "
                      f"{orig_b:>10,} | {comp_b:>9,} | {ratio:>7.2f} | "
                      f"{enc_val:>9.2f} | {dec_val:>9.2f} | {psnr:>7.3f}")

        # ---- averages ----
        def m(k): return float(np.mean([x[k] for x in metrics]))
        summary_rows.append({
            "dataset":       dataset_name,
            "rate":          force_ind,
            "num_images":    len(metrics),
            "orig_bytes":    round(m("orig_bytes"), 2),
            "comp_bytes":    round(m("comp_bytes"), 2),
            "ratio":         round(m("ratio"), 4),
            "enc_ms":        round(m("enc_ms"), 4),
            "enc_std_ms":    round(m("enc_std_ms"), 4),
            "dec_ms":        round(m("dec_ms"), 4),
            "dec_std_ms":    round(m("dec_std_ms"), 4),
            "psnr_db":       round(m("psnr_db"), 4),
        })
        print(f"\nAVG rate={force_ind} | ratio={m('ratio'):.2f}x | "
              f"Enc={m('enc_ms'):.3f} ms | Dec={m('dec_ms'):.3f} ms | "
              f"PSNR={m('psnr_db'):.3f} dB\n")

    # ---- write CSVs ----
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = out_dir / f"{dataset_name}_per_image.csv"
    summary_csv   = out_dir / f"{dataset_name}_summary.csv"

    with open(per_image_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        w.writeheader(); w.writerows(per_image_rows)

    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    print(f"Saved -> {per_image_csv}")
    print(f"Saved -> {summary_csv}")
    return summary_rows


# ==================================================================
# Main
# ==================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kodak-dir",      type=Path, default=Path("kodak"))
    ap.add_argument("--tecnick-dir",    type=Path, default=Path("tecnick_flat"))
    ap.add_argument("--cityscapes-dir", type=Path,
                    default=Path("/home/common/EF-LIC/datasets/cityscapes/"
                                 "leftImg8bit/val/frankfurt"))
    ap.add_argument("--ckpt-path",      type=Path,
                    default=Path("ckpt/checkpoint.pth.tar"))
    ap.add_argument("--device",         type=str,
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir",        type=Path, default=Path("results"))
    ap.add_argument("--datasets",       type=str, nargs="+",
                    default=["kodak", "tecnick", "cityscapes"],
                    choices=["kodak", "tecnick", "cityscapes"])
    ap.add_argument("--max-cityscapes", type=int, default=None)
    ap.add_argument("--max-images",     type=int, default=None,
                    help="Cap ALL datasets at N images (useful for bench).")

    # ---- accelerator ----
    ap.add_argument("--accelerator", type=str, default="none",
                    choices=["none", "torch_compile", "onnx"],
                    help="none | torch_compile | onnx (attempt TRT build)")
    ap.add_argument("--onnx-path",   type=Path, default=Path("eflic.onnx"))
    ap.add_argument("--engine-path", type=Path, default=Path("eflic.trt"))

    # ---- benchmark ----
    ap.add_argument("--bench", type=int, default=0,
                    help="1 to enable 1000-iteration benchmark per image")
    ap.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    ap.add_argument("--iters",  type=int, default=MEASURE_ITERS)
    ap.add_argument("--avg-last", type=int, default=AVERAGE_LAST_N)
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    print(f"Device      : {device}")
    print(f"Accelerator : {args.accelerator}")
    print(f"Benchmark   : {'on' if args.bench else 'off'}")

    # ---- load model ----
    net = model().to(device).eval()
    net.load_state_dict(load_checkpoint(args.ckpt_path, device), strict=True)
    print(f"Checkpoint  : {args.ckpt_path}")

    # ---- optional accelerator ----
    if args.accelerator == "torch_compile":
        net = try_torch_compile(net, device)

    elif args.accelerator == "onnx":
        # Need a dummy input for ONNX export — use first Kodak image
        sample = None
        for cand in [args.kodak_dir, args.tecnick_dir, args.cityscapes_dir]:
            if cand.exists():
                imgs = list_images(cand)
                sample = load_image(imgs[0], device)
                break
        if sample is None:
            print("[accel] No dataset found for ONNX dummy input; skipping")
        else:
            ok = try_onnx_export(net, sample, args.onnx_path)
            if ok:
                build_trt_engine(args.onnx_path, args.engine_path)
                print("[accel] NOTE: this script still runs the PyTorch net "
                      "for evaluation. To use the TRT engine, load "
                      f"{args.engine_path} in your inference code.")

    # ---- dataset loop ----
    dir_map = {
        "kodak":      args.kodak_dir,
        "tecnick":    args.tecnick_dir,
        "cityscapes": args.cityscapes_dir,
    }
    all_summaries = []
    for name in args.datasets:
        d = dir_map[name]
        if not d.exists():
            print(f"[skip] {name}: {d} not found"); continue

        cap = args.max_images
        if name == "cityscapes" and args.max_cityscapes is not None:
            cap = args.max_cityscapes

        summary = evaluate_dataset(
            net, name, d, device, args.out_dir,
            max_images=cap,
            do_bench=bool(args.bench),
            warmup=args.warmup,
            iters=args.iters,
            avg_last=args.avg_last,
        )
        all_summaries.extend(summary)

    # ---- combined CSV + table ----
    if all_summaries:
        combined_csv = args.out_dir / "combined_summary.csv"
        with open(combined_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader(); w.writerows(all_summaries)
        print(f"\nSaved combined summary -> {combined_csv}")

    print(f"\n{'='*120}")
    print(f"{'FINAL SUMMARY':^120}")
    print(f"{'='*120}")
    hdr = (f"{'dataset':<12} | {'rate':>4} | {'N':>5} | "
           f"{'orig(B)':>12} | {'comp(B)':>12} | {'ratio':>8} | "
           f"{'Enc_mean':>9} | {'Enc_std':>8} | "
           f"{'Dec_mean':>9} | {'Dec_std':>8} | {'PSNR':>8}")
    print(hdr); print("-" * len(hdr))
    for r in all_summaries:
        print(f"{r['dataset']:<12} | {r['rate']:>4} | {r['num_images']:>5} | "
              f"{r['orig_bytes']:>12,.0f} | {r['comp_bytes']:>12,.0f} | "
              f"{r['ratio']:>8.2f} | "
              f"{r['enc_ms']:>9.3f} | {r['enc_std_ms']:>8.3f} | "
              f"{r['dec_ms']:>9.3f} | {r['dec_std_ms']:>8.3f} | "
              f"{r['psnr_db']:>8.3f}")


if __name__ == "__main__":
    main()

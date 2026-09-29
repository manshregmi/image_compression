"""
lic_tcm_wrapper.py
------------------
EF-LIC-style adapter for LIC_TCM.

Handles:
  - CompressAI EntropyBottleneck key renaming (old ↔ new)
  - CDF initialisation via update(force=True) after loading weights
  - TCM decompress returns a dict; extract 'x_hat'
  - Padding to 128 (required by TCM's Swin window sizes)
  - Recursive byte accounting for the compressed bitstream
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models import TCM


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# TCM's main path downsamples 16x and its Swin window is 8 → needs /128.
# TCM's hyperprior path downsamples 32x and its window is 4 → also /128.
PAD_MULTIPLE = 128
HEADER_BYTES = 32


# ------------------------------------------------------------------
# Checkpoint discovery
# ------------------------------------------------------------------
def discover_checkpoints(ckpt_dir: Path, model_size: int = 64):
    found = []
    pattern = f"TCM_MSE_lambda_*_N{model_size}.pth.tar"
    for p in sorted(Path(ckpt_dir).glob(pattern)):
        try:
            lam = float(p.stem.split("lambda_")[1].split("_")[0])
            found.append((lam, p))
        except Exception:
            continue
    found.sort(key=lambda x: x[0])
    return found


# ------------------------------------------------------------------
# Padding — pad to the nearest multiple of PAD_MULTIPLE
# ------------------------------------------------------------------
def replicate_pad(x: torch.Tensor, p: int = PAD_MULTIPLE):
    H, W = x.shape[-2], x.shape[-1]
    new_h = (H + p - 1) // p * p
    new_w = (W + p - 1) // p * p
    pad_h, pad_w = new_h - H, new_w - W
    if pad_h == 0 and pad_w == 0:
        return x, H, W
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x_pad, H, W


# ------------------------------------------------------------------
# Byte accounting
# ------------------------------------------------------------------
def _count_bytes(obj) -> int:
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_count_bytes(x) for x in obj)
    return 0


def _stream_breakdown(strings):
    breakdown = []

    def walk(obj, prefix):
        if isinstance(obj, (bytes, bytearray)):
            breakdown.append((prefix, len(obj)))
        elif isinstance(obj, torch.Tensor):
            breakdown.append((prefix, obj.numel() * obj.element_size()))
        elif isinstance(obj, (list, tuple)):
            for i, x in enumerate(obj):
                child = f"{prefix}[{i}]" if prefix else f"[{i}]"
                walk(x, child)

    if not isinstance(strings, (list, tuple)):
        return [("stream0", _count_bytes(strings))]

    if len(strings) == 2 and isinstance(strings[0], (list, tuple)) \
            and isinstance(strings[1], (list, tuple)):
        for i, s in enumerate(strings[0]):
            breakdown.append((f"y[{i}]", _count_bytes(s)))
        for i, s in enumerate(strings[1]):
            breakdown.append((f"z[{i}]", _count_bytes(s)))
    else:
        walk(strings, "")

    return breakdown


def _naive_bytes(strings) -> int:
    try:
        return sum(len(s[0]) for s in strings)
    except Exception:
        return 0


# ------------------------------------------------------------------
# CompressAI key remapping
# ------------------------------------------------------------------
def _old_to_new(k):
    return (k.replace("_matrix", "matrices.")
             .replace("_bias",   "biases.")
             .replace("_factor", "factors."))

def _new_to_old(k):
    return (k.replace(".matrices.", "._matrix")
             .replace(".biases.",   "._bias")
             .replace(".factors.",  "._factor"))

def _remap_compressai_keys(state, model_keys):
    ckpt_old = any("_matrix" in k for k in state)
    ckpt_new = any(".matrices." in k for k in state)
    model_old = any("_matrix" in k for k in model_keys)
    model_new = any(".matrices." in k for k in model_keys)
    if ckpt_old and model_new:
        print("[LICTCMAdapter] remapping keys: old -> new")
        return {_old_to_new(k): v for k, v in state.items()}
    if ckpt_new and model_old:
        print("[LICTCMAdapter] remapping keys: new -> old")
        return {_new_to_old(k): v for k, v in state.items()}
    return state


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------
class LICTCMAdapter(nn.Module):
    def __init__(self, checkpoint_dir, device="cuda", model_size=64,
                 model_cls=None):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self.model_size = model_size
        self.model_cls = model_cls or TCM

        self.available = discover_checkpoints(self.checkpoint_dir, model_size)
        if not self.available:
            raise FileNotFoundError(
                f"No TCM_MSE_lambda_*_N{model_size}.pth.tar in "
                f"{self.checkpoint_dir}"
            )
        self.available_lambdas = [lam for lam, _ in self.available]
        self.num_rates = len(self.available_lambdas)
        print(f"[LICTCMAdapter] Found {self.num_rates} checkpoints:")
        for lam, p in self.available:
            print(f"  λ={lam:<8}  {p.name}  ({p.stat().st_size/1e6:.0f} MB)")

        self.current_force_ind = None
        self.net = None
        self.prepare_inference_(0)

    def _build_and_load(self, force_ind: int):
        lam, ckpt_path = self.available[force_ind]
        print(f"[LICTCMAdapter] loading {ckpt_path.name}  (λ={lam})")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict):
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        else:
            state = ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}

        model = None
        for kwargs in (dict(N=self.model_size, M=320),
                       dict(N=self.model_size), dict()):
            try:
                print(f"[LICTCMAdapter] constructing "
                      f"{self.model_cls.__name__}({kwargs})")
                model = self.model_cls(**kwargs)
                break
            except TypeError as e:
                print(f"[LICTCMAdapter]   failed: {e}")
                continue
        if model is None:
            raise RuntimeError("Could not construct TCM")

        state = _remap_compressai_keys(state, list(model.state_dict().keys()))

        try:
            missing, unexpected = nn.Module.load_state_dict(
                model, state, strict=False)
        except Exception as e:
            print(f"[LICTCMAdapter] strict=False load failed: {e}")
            model.load_state_dict(state)
            missing, unexpected = [], []

        if missing:
            print(f"[LICTCMAdapter] {len(missing)} missing keys "
                  f"(first 3: {missing[:3]})")
        if unexpected:
            print(f"[LICTCMAdapter] {len(unexpected)} unexpected keys "
                  f"(first 3: {unexpected[:3]})")
        if not missing and not unexpected:
            print("[LICTCMAdapter] all keys matched")

        model.eval().to(self.device)

        try:
            model.update(force=True)
            print("[LICTCMAdapter] update(force=True) done")
        except Exception as e:
            print(f"[LICTCMAdapter] update(force=True) failed: {e}")
            try:
                model.update()
                print("[LICTCMAdapter] update() done")
            except Exception as e2:
                print(f"[LICTCMAdapter] update() also failed: {e2}")

        return model

    @torch.no_grad()
    def prepare_inference_(self, force_ind: int):
        if force_ind != self.current_force_ind:
            lam = self.available_lambdas[force_ind]
            print(f"[LICTCMAdapter] rate {force_ind}  λ={lam}")
            self.net = self._build_and_load(force_ind)
            self.current_force_ind = force_ind

    @torch.no_grad()
    def compress(self, x: torch.Tensor, force_ind: int = None):
        if force_ind is not None and force_ind != self.current_force_ind:
            self.prepare_inference_(force_ind)

        x01 = (x + 1.0) * 0.5
        x_pad, H, W = replicate_pad(x01)

        out = self.net.compress(x_pad)
        strings = out["strings"]
        shape = out["shape"]

        breakdown = _stream_breakdown(strings)
        total_streams = sum(b for _, b in breakdown)
        packed_bytes = total_streams + HEADER_BYTES

        return {
            "strings":          strings,
            "shape":            shape,
            "packed_bytes":     int(packed_bytes),
            "stream_breakdown": breakdown,
            "header_bytes":     HEADER_BYTES,
            "naive_bytes":      _naive_bytes(strings) + HEADER_BYTES,
            "H":                int(H),
            "W":                int(W),
        }

    @torch.no_grad()
    def decompress(self, compressed, force_ind: int = None):
        out = self.net.decompress(compressed["strings"], compressed["shape"])
        if isinstance(out, dict):
            x_hat = out.get("x_hat")
            if x_hat is None:
                for v in out.values():
                    if isinstance(v, torch.Tensor) and v.dim() == 4:
                        x_hat = v
                        break
                if x_hat is None:
                    raise RuntimeError(
                        f"No 4-D tensor in decompress output: {list(out.keys())}")
        else:
            x_hat = out
        H, W = compressed["H"], compressed["W"]
        x_hat = x_hat[:, :, :H, :W]
        return x_hat.clamp(0, 1) * 2.0 - 1.0

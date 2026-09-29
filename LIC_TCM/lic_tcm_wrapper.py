"""
lic_tcm_wrapper.py
------------------
Thin adapter that exposes an EF-LIC-like interface to LIC_TCM so that
existing evaluation/visualization scripts work unchanged.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Adjust these imports to match LIC_TCM's actual module layout
from models.tcm import TCM               # main model class
from models.utils import pad_to_multiple


# Order matters — this is the paper's MSE-optimised lambda list.
LAMBDAS = [0.0025, 0.0035, 0.0067, 0.0130, 0.0250, 0.0500]


class LICTCMAdapter(nn.Module):
    """
    Wraps LIC_TCM so it exposes:
        net.prepare_inference_(force_ind=k)  -> load k-th rate point
        net.compress(x)                      -> dict with bitstreams + shapes
        net.decompress(strings, shape)       -> reconstructed image
    """

    def __init__(self, checkpoint_dir: Path, device="cuda"):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.current_force_ind = 0

        # Load the first rate point by default
        self.net = self._load_for(0)

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------
    def _ckpt_path(self, force_ind: int) -> Path:
        lam = LAMBDAS[force_ind]
        candidates = [
            self.checkpoint_dir / f"TCM_MSE_lambda_{lam}.pth.tar",
            self.checkpoint_dir / f"TCM_MSE_lambda_{lam}.pth",
            self.checkpoint_dir / f"lambda_{lam}.pth.tar",
            self.checkpoint_dir / f"tcm_lambda_{lam}.pth.tar",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"No checkpoint for force_ind={force_ind} (lambda={lam}). "
            f"Tried: {[str(c) for c in candidates]}"
        )

    def _load_for(self, force_ind: int):
        ckpt_path = self._ckpt_path(force_ind)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)

        model = TCM(N=128, M=320)   # adjust N/M if the repo uses different names
        model.load_state_dict(state, strict=False)
        model.eval().to(self.device)
        return model

    # ------------------------------------------------------------------
    # Rate switching (matches EF-LIC's API)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prepare_inference_(self, force_ind: int):
        if force_ind != self.current_force_ind:
            self.net = self._load_for(force_ind)
            self.current_force_ind = force_ind

    # ------------------------------------------------------------------
    # Encode: returns a dict that we treat like EF-LIC's 'inds'
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compress(self, x: torch.Tensor, force_ind: int = None):
        if force_ind is not None and force_ind != self.current_force_ind:
            self.prepare_inference_(force_ind)

        # LIC_TCM expects pixel range [0, 1]
        x_norm = (x + 1.0) * 0.5 if x.min() < 0 else x

        out = self.net.compress(x_norm)

        # CompressAI returns a dict with 'strings' and 'shape'
        strings = out["strings"]
        shape   = out["shape"]

        # Total compressed size (this IS the true compressed size)
        total_bytes = sum(len(s[0]) for s in strings) + 64  # +header

        # We mimic EF-LIC's return format so downstream code just works
        # (no actual VQ indices — we store the bitstream bytes and shapes)
        return {
            "strings":      strings,
            "shape":        shape,
            "packed_bytes": total_bytes,
            "H":            x.shape[-2],
            "W":            x.shape[-1],
        }

    # ------------------------------------------------------------------
    # Decode: takes the dict from compress() and returns the image
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decompress(self, compressed, force_ind: int = None):
        out = self.net.decompress(
            compressed["strings"],
            compressed["shape"],
        )
        # Map back to [-1, 1] for consistency with EF-LIC scripts
        return out.clamp(0, 1) * 2.0 - 1.0

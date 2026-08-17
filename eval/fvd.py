"""FVD (Fréchet Video Distance) with the standard Kinetics-400 I3D torchscript used by StyleGAN-V /
most FVD implementations. Videos: uint8 [N,T,H,W,3] RGB (composite premultiplied RGBA over white first).

    from eval.fvd import fvd
    d = fvd(real_videos, fake_videos, device="cuda")

Notes: I3D wants T>=9 (we use 16), 224x224 input (we resize), and >=~256 videos per set for a
stable estimate — report N and use the same N for real and fake. FVD is only comparable within
this exact setup (I3D weights, resize, N, T).
"""
from __future__ import annotations
import os, urllib.request
import numpy as np
import torch
import torch.nn.functional as F

I3D_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
I3D_PATH = os.path.expanduser("~/.cache/stickdance/i3d_torchscript.pt")
_model = None


def _load(device):
    global _model
    if _model is None:
        os.makedirs(os.path.dirname(I3D_PATH), exist_ok=True)
        if not os.path.exists(I3D_PATH):
            print("downloading I3D torchscript ...", flush=True)
            urllib.request.urlretrieve(I3D_URL, I3D_PATH)
        _model = torch.jit.load(I3D_PATH).eval().to(device)
    return _model


@torch.no_grad()
def features(videos: np.ndarray, device="cuda", bs=16) -> np.ndarray:
    """uint8 [N,T,H,W,3] -> [N,400] I3D features (logits layer, as in the reference implementations)."""
    m = _load(device)
    out = []
    for i in range(0, len(videos), bs):
        x = torch.from_numpy(videos[i:i + bs]).to(device).float()          # [B,T,H,W,3]
        x = x.permute(0, 4, 1, 2, 3)                                        # [B,3,T,H,W]
        B, C, T, H, W = x.shape
        x = F.interpolate(x.reshape(B, C * T, H, W), size=(224, 224), mode="bilinear", align_corners=False).reshape(B, C, T, 224, 224)
        x = x / 127.5 - 1.0
        f = m(x, rescale=False, resize=False, return_features=True)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out, 0)


def frechet(mu1, s1, mu2, s2):
    from scipy import linalg
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def stats(feats):
    return feats.mean(0), np.cov(feats, rowvar=False)


def fvd(real: np.ndarray, fake: np.ndarray, device="cuda") -> float:
    fr, ff = features(real, device), features(fake, device)
    return frechet(*stats(fr), *stats(ff))


def rgba_premult_to_rgb(x: np.ndarray) -> np.ndarray:
    """[N,T,H,W,4] float in [0,1], premultiplied -> uint8 RGB over white."""
    rgb, a = x[..., :3], x[..., 3:4]
    return (np.clip(rgb + (1 - a), 0, 1) * 255).astype(np.uint8)

"""Renderer v2: z-buffer rasteriser -> colour / depth / normal / segment-id, all from one pass.

Every bone is a 3D capsule (round tube). At 4x supersample, per pixel: distance to the
projected axis d, param t along it -> tube depth = lerp(depth_a, depth_b, t) + sqrt(r²-d²)
(nearer = larger), camera-space normal = (dx/r, dy/r, sqrt(1-(d/r)²)). A z-test resolves
occlusion per pixel, so no chain sorting. Head = sphere. Hands = mitten capsule (width follows
palm roll) + thumb sphere, or 3 finger capsules. Feet = shoe capsule with a heel.
Outputs are box-downsampled to 128 (colour coverage-weighted; depth/normal likewise).
"""
from __future__ import annotations
import math
import numpy as np
from PIL import Image
from .skeleton import Body, PARENT, NAMES, THUMB_LEN

SS = 4
SIZE = 128
W = SIZE * SS
INK = (40, 40, 40)
PALETTE = {  # per bone (child joint); unlisted = ink
    "LeftForeArm": (232, 64, 48), "LeftHand": (255, 150, 40), "LeftHandEnd": (255, 150, 40),
    "RightForeArm": (40, 110, 230), "RightHand": (80, 200, 240), "RightHandEnd": (80, 200, 240),
    "LeftLeg": (200, 50, 160), "LeftFoot": (255, 120, 200), "LeftToeBase": (255, 120, 200),
    "RightLeg": (30, 150, 90), "RightFoot": (120, 220, 90), "RightToeBase": (120, 220, 90),
}
SEG_ID = {n: i + 1 for i, n in enumerate(NAMES)}   # 0 = background
DRAWN = [n for n in NAMES if n not in ("Hips", "Head", "LeftHandThumb1", "RightHandThumb1")]
_YY, _XX = np.mgrid[0:W, 0:W].astype(np.float32)
_XX += 0.5; _YY += 0.5


class Raster:
    def __init__(self, m_per_px: float):
        self.m_per_px = m_per_px
        self.depth = np.full((W, W), -np.inf, np.float32)
        self.rgb = np.zeros((W, W, 3), np.float32)
        self.nrm = np.zeros((W, W, 3), np.float32)
        self.seg = np.zeros((W, W), np.uint8)

    def _write(self, y0, y1, x0, x1, mask, z, nx, ny, nz, col, sid):
        zb = self.depth[y0:y1, x0:x1]
        upd = mask & (z > zb)
        zb[upd] = z[upd]
        for c in range(3):
            self.rgb[y0:y1, x0:x1, c][upd] = col[c]
        self.nrm[y0:y1, x0:x1, 0][upd] = nx[upd]
        self.nrm[y0:y1, x0:x1, 1][upd] = ny[upd]
        self.nrm[y0:y1, x0:x1, 2][upd] = nz[upd]
        self.seg[y0:y1, x0:x1][upd] = sid

    def capsule(self, a, b, za, zb_, ra, rb, col, sid):
        """3D capsule between screen points a,b (supersampled px), depths za,zb (m), radii ra,rb (px)."""
        rmax = max(ra, rb)
        x0 = int(max(0, math.floor(min(a[0], b[0]) - rmax - 1))); x1 = int(min(W, math.ceil(max(a[0], b[0]) + rmax + 1)))
        y0 = int(max(0, math.floor(min(a[1], b[1]) - rmax - 1))); y1 = int(min(W, math.ceil(max(a[1], b[1]) + rmax + 1)))
        if x1 <= x0 or y1 <= y0:
            return
        X = _XX[y0:y1, x0:x1]; Y = _YY[y0:y1, x0:x1]
        ux, uy = b[0] - a[0], b[1] - a[1]
        L2 = ux * ux + uy * uy
        t = np.zeros_like(X) if L2 < 1e-6 else np.clip(((X - a[0]) * ux + (Y - a[1]) * uy) / L2, 0.0, 1.0)
        dx = X - (a[0] + t * ux); dy = Y - (a[1] + t * uy)
        d2 = dx * dx + dy * dy
        r = ra + (rb - ra) * t
        r2 = r * r
        mask = d2 <= r2
        if not mask.any():
            return
        rr = np.sqrt(np.maximum(r2 - d2, 0.0))
        z = za + (zb_ - za) * t + rr * self.m_per_px       # tube bulge toward camera
        rs = np.where(r > 1e-6, r, 1.0)
        nx = dx / rs; ny = -dy / rs; nz = rr / rs          # normal: x right, y up, z toward camera
        self._write(y0, y1, x0, x1, mask, z, nx, ny, nz, col, sid)

    def sphere(self, c, zc, r, col, sid):
        self.capsule(c, c, zc, zc, r, r, col, sid)


def rasterize(j2: dict, depth: dict, body: Body, colored: bool = True) -> Raster:
    R = Raster(1.0 / (body.px_per_m * SS))
    j = {k: (x * SS, y * SS) for k, (x, y) in j2.items()}
    w = body.stroke * SS
    for child in DRAWN:
        parent = PARENT[child]
        a, b = j[parent], j[child]
        za, zb = depth[parent], depth[child]
        col = (PALETTE.get(child, INK) if colored else INK)
        sid = SEG_ID[child]
        if child.endswith("HandEnd"):
            _hand(R, j, depth, child, a, b, za, zb, w, col, sid, body)
        elif child.endswith("ToeBase"):
            _foot(R, a, b, za, zb, w, col, sid, body)
        else:
            R.capsule(a, b, za, zb, w / 2, w / 2, col, sid)
    R.sphere(j["Head"], depth["Head"], body.head_r * SS, INK, SEG_ID["Head"])
    return R


def _hand(R, j, depth, child, wrist, tip, zw, zt, w, col, sid, body):
    thumb = child.replace("HandEnd", "HandThumb1")
    ux, uy = tip[0] - wrist[0], tip[1] - wrist[1]
    L = math.hypot(ux, uy)
    tl = THUMB_LEN * body.px_per_m * SS
    if thumb in j:
        lx, ly = (j[thumb][0] - wrist[0]) / tl, (j[thumb][1] - wrist[1]) / tl
        m = min(1.0, math.hypot(lx, ly))
    else:
        lx, ly, m = 0.0, 0.0, 1.0
    if L < 1e-3:
        R.sphere(wrist, zw, w * 0.8, col, sid); return
    ux, uy = ux / L, uy / L
    if body.hand_style == "fingers":
        px, py = (lx / m, ly / m) if m > 1e-3 else (-uy, ux)
        palm = (wrist[0] + ux * L * 0.35, wrist[1] + uy * L * 0.35)
        zp = zw + (zt - zw) * 0.35
        R.sphere(palm, zp, w * 0.65, col, sid)
        for f in (-1, 0, 1):
            base = (palm[0] + px * f * w * 0.45 * m, palm[1] + py * f * w * 0.45 * m)
            end = (tip[0] + px * f * w * 0.75 * m, tip[1] + py * f * w * 0.75 * m)
            R.capsule(base, end, zp, zt, w * 0.25, w * 0.22, col, sid)
        return
    pr = w * (0.55 + 0.55 * m) / 2
    a = (wrist[0] + ux * L * 0.15, wrist[1] + uy * L * 0.15)
    b = (wrist[0] + ux * L * 0.9, wrist[1] + uy * L * 0.9)
    R.capsule(a, b, zw + (zt - zw) * 0.15, zw + (zt - zw) * 0.9, pr, pr, col, sid)
    if thumb in j:
        R.sphere(j[thumb], depth[thumb], w * 0.42, col, sid)


def _foot(R, ankle, toe, za, zt, w, col, sid, body):
    if body.foot_style == "segment":
        R.capsule(ankle, toe, za, zt, w / 2, w / 2, col, sid); return
    ux, uy = toe[0] - ankle[0], toe[1] - ankle[1]
    L = math.hypot(ux, uy)
    if L < 1e-3:
        R.sphere(ankle, za, w * 0.7, col, sid); return
    ux, uy = ux / L, uy / L
    heel = (ankle[0] - ux * w * 0.35, ankle[1] - uy * w * 0.35)
    R.capsule(heel, toe, za, zt, w * 0.625, w * 0.625, col, sid)


# ----------------------------------------------------------------- outputs
def _box(a: np.ndarray) -> np.ndarray:
    if a.ndim == 2:
        return a.reshape(SIZE, SS, SIZE, SS).mean(axis=(1, 3))
    return a.reshape(SIZE, SS, SIZE, SS, a.shape[2]).mean(axis=(1, 3))


def outputs(R: Raster, bg=(0, 0, 0, 0)):
    """-> dict(color: RGBA PIL, depth: float32 [128,128] metres toward camera (nan = bg),
    normal: RGB PIL (n*0.5+0.5), seg: uint8 [128,128] majority bone id, alpha: coverage)."""
    cov = (R.seg > 0).astype(np.float32)
    alpha = _box(cov)
    safe = np.where(alpha > 0, alpha, 1.0)
    rgb = _box(R.rgb * cov[..., None]) / safe[..., None]           # mean colour of covered subpixels
    bg_rgb = np.array(bg[:3], np.float32); bg_a = bg[3] / 255.0
    a_out = alpha + (1 - alpha) * bg_a
    a_safe = np.where(a_out > 0, a_out, 1.0)
    rgb_out = (rgb * alpha[..., None] + bg_rgb * ((1 - alpha) * bg_a)[..., None]) / a_safe[..., None]
    col = np.concatenate([rgb_out, a_out[..., None] * 255], axis=-1)
    color = Image.fromarray(np.clip(col, 0, 255).astype(np.uint8), "RGBA")
    dep = np.where(R.seg > 0, R.depth, 0.0).astype(np.float32)
    depth = _box(dep) / safe
    depth[alpha <= 0] = np.nan
    n = _box(R.nrm * cov[..., None])
    nn = np.linalg.norm(n, axis=-1, keepdims=True); n = n / np.where(nn > 1e-6, nn, 1.0)
    nimg = np.empty((SIZE, SIZE, 3), np.float32)
    nimg[...] = (128, 128, 255)
    nimg[alpha > 0] = (n[alpha > 0] * 0.5 + 0.5) * 255
    normal = Image.fromarray(nimg.astype(np.uint8), "RGB")
    seg = R.seg.reshape(SIZE, SS, SIZE, SS).transpose(0, 2, 1, 3).reshape(SIZE, SIZE, SS * SS)
    counts = np.stack([(seg == i).sum(-1) for i in range(len(NAMES) + 1)], -1)   # [128,128,28]
    counts[..., 0] = 0
    segm = counts.argmax(-1).astype(np.uint8)
    segm[alpha <= 0] = 0
    return dict(color=color, depth=depth, normal=normal, seg=segm, alpha=alpha)


def render(j2: dict, depth: dict, body: Body, colored: bool = True, halo: bool = False,
           bg=(0, 0, 0, 0), halo_rgb=(255, 255, 255)) -> Image.Image:
    """Back-compat: colour image only (halo args ignored — occlusion is per-pixel now)."""
    return outputs(rasterize(j2, depth, body, colored), bg=bg)["color"]


def render_all(j2: dict, depth: dict, body: Body, colored: bool = True, bg=(0, 0, 0, 0)):
    return outputs(rasterize(j2, depth, body, colored), bg=bg)


def depth_to_image(depth: np.ndarray, near=None, far=None) -> Image.Image:
    """visualise: nearer = brighter, bg black."""
    m = ~np.isnan(depth)
    if not m.any(): return Image.new("L", (SIZE, SIZE), 0)
    near = np.nanmax(depth) if near is None else near; far = np.nanmin(depth) if far is None else far
    v = np.zeros_like(depth); v[m] = (depth[m] - far) / max(near - far, 1e-6)
    return Image.fromarray((np.clip(v, 0, 1) * 200 + 55 * m).astype(np.uint8), "L")

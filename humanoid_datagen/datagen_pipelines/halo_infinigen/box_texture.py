"""Procedural cardboard texture for the push box (HALO's pushbox.urdf has only a flat rgba 0.75 0.65 0.45).

Writes six square PNGs for a MuJoCo cube texture on the box geom of datagen_pipelines/halo_infinigen/episodes.py (push along +x, z up):
cardboard base (the HALO rgba, mottling, faint corrugation streaks, darker worn edges); a 48 mm strip of packing tape
along x over the top flap seam, continuing 14 cm down the middle of the +x and -x faces; a white shipping label on
the -y face. MuJoCo cube-file mapping (checked with a probe render, mujoco 3.x): fileright/left = +x/-x,
fileup/down = +y/-y, filefront/back = +z/-z; image axes per face are noted in make().

  python -m humanoid_datagen.datagen_pipelines.halo_infinigen.box_texture [--out humanoid_datagen/assets/pushbox_tex] [--res 512]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

BASE = np.array([0.75, 0.65, 0.45])      # pushbox.urdf "cardboard"
TAPE = np.array([0.88, 0.80, 0.60])      # brown packing tape (lighter, slightly yellow)
BOX = (0.46, 0.50, 0.561264)             # D (x), W (y), H (z), m
FILES = dict(right="px", left="nx", up="py", down="ny", front="top", back="bottom")


def _noise(n, rng, scale):
    """Smooth value noise in [-1, 1] (bilinear upsampled random grid)."""
    g = rng.uniform(-1, 1, (scale + 2, scale + 2))
    t = np.linspace(0, scale, n, endpoint=False)
    i = t.astype(int)
    f = t - i
    a = g[i][:, i] * (1 - f)[None, :] + g[i][:, i + 1] * f[None, :]
    b = g[i + 1][:, i] * (1 - f)[None, :] + g[i + 1][:, i + 1] * f[None, :]
    return a * (1 - f)[:, None] + b * f[:, None]


def _cardboard(n, rng, streaks_vary_along_cols):
    img = np.ones((n, n, 3)) * BASE
    m = 0.06 * _noise(n, rng, 5) + 0.03 * _noise(n, rng, 23)
    s = 0.015 * np.sin(np.linspace(0, 2 * np.pi * 80, n))       # corrugation streaks
    s = s[None, :] if streaks_vary_along_cols else s[:, None]
    img *= (1 + m + s)[..., None]
    u = np.linspace(0, 1, n)
    edge = np.minimum.outer(np.minimum(u, 1 - u), np.minimum(u, 1 - u))
    img *= (1 - 0.22 * np.exp(-edge / 0.02))[..., None]          # darker worn edges
    return img


def _rect(img, rows, cols, colour, rng=None, edge=0.85):
    """Fill a rectangle given as fractional (lo, hi) row and column ranges; darker 1-px border."""
    n = img.shape[0]
    r0, r1 = (np.clip(np.array(rows) * n, 0, n)).astype(int)
    c0, c1 = (np.clip(np.array(cols) * n, 0, n)).astype(int)
    patch = np.ones((r1 - r0, c1 - c0, 3)) * colour
    if rng is not None:
        patch *= (1 + 0.03 * rng.uniform(-1, 1, patch.shape[:2]))[..., None]
    img[r0:r1, c0:c1] = patch
    for sl in ((slice(r0, r0 + 2), slice(c0, c1)), (slice(r1 - 2, r1), slice(c0, c1)),
               (slice(r0, r1), slice(c0, c0 + 2)), (slice(r0, r1), slice(c1 - 2, c1))):
        img[sl] *= edge
    return img


def make(res=512, seed=0):
    rng = np.random.default_rng(seed)
    D, W, H = BOX
    th = 0.048 / 2                                               # tape half width, m
    drop = 0.14                                                  # tape down the +x / -x faces, m
    f = {}
    # top (+z, "front"): image rows run along y, columns along x -> tape along x = a band of rows at y = 0
    f["top"] = _cardboard(res, rng, True)
    _rect(f["top"], (0.5 - th / W, 0.5 + th / W), (0, 1), TAPE, rng)
    # -x ("left"): rows run along y, columns along z with column 0 at the bottom -> tape near the right end
    f["nx"] = _cardboard(res, rng, False)
    _rect(f["nx"], (0.5 - th / W, 0.5 + th / W), (1 - drop / H, 1), TAPE, rng)
    # +x ("right"): rows along y, columns along z with column 0 at the top -> tape near the left end
    f["px"] = _cardboard(res, rng, False)
    _rect(f["px"], (0.5 - th / W, 0.5 + th / W), (0, drop / H), TAPE, rng)
    # -y ("down"): upright image (row 0 at the top), columns along x -> shipping label with a few printed lines
    f["ny"] = _cardboard(res, rng, True)
    lr, lc = (0.30, 0.52), (0.18, 0.52)
    _rect(f["ny"], lr, lc, np.array([0.95, 0.94, 0.90]), rng, edge=0.9)
    for i, w in enumerate((0.8, 0.6, 0.7, 0.4)):
        y = lr[0] + 0.035 + i * 0.045
        _rect(f["ny"], (y, y + 0.015), (lc[0] + 0.03, lc[0] + 0.03 + w * (lc[1] - lc[0] - 0.06)),
              np.array([0.2, 0.2, 0.2]), edge=1.0)
    f["py"] = _cardboard(res, rng, True)
    f["bottom"] = _cardboard(res, rng, True)
    return {k: np.clip(v, 0, 1) for k, v in f.items()}


def write(out, res=512, seed=0):
    import imageio.v2 as iio
    os.makedirs(out, exist_ok=True)
    faces = make(res, seed)
    for key, name in FILES.items():
        iio.imwrite(os.path.join(out, f"{name}.png"), (faces[name] * 255).astype(np.uint8))
    return {key: os.path.join(out, f"{name}.png") for key, name in FILES.items()}


def mjcf_texture(out):
    """MuJoCo <texture> attributes for the six files in `out`."""
    return " ".join(f'file{key}="{os.path.join(out, name + ".png")}"' for key, name in FILES.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="humanoid_datagen/assets/pushbox_tex")
    ap.add_argument("--res", type=int, default=512)
    a = ap.parse_args()
    from humanoid_datagen.datagen_pipelines.halo_infinigen.episodes import rp
    print(write(rp(a.out), a.res))


if __name__ == "__main__":
    main()

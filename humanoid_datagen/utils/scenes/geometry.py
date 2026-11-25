"""Geometry helpers: primitive volumes, mesh volumes and cached CoACD convex decomposition."""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import trimesh


def primitive_volume(g: dict) -> float:
    t, s = g["type"], g.get("size", [])
    if t == "box":
        return 8.0 * s[0] * s[1] * s[2]
    if t == "sphere":
        return 4.0 / 3.0 * math.pi * s[0] ** 3
    if t == "ellipsoid":
        return 4.0 / 3.0 * math.pi * s[0] * s[1] * s[2]
    if t in ("capsule", "cylinder"):
        if g.get("fromto") is not None:
            ft = np.asarray(g["fromto"], float)
            half = 0.5 * np.linalg.norm(ft[3:] - ft[:3])
        else:
            half = s[1]
        vol = math.pi * s[0] ** 2 * 2 * half
        return vol + (4.0 / 3.0 * math.pi * s[0] ** 3 if t == "capsule" else 0.0)
    return 0.0


def load_mesh(path: str, scale=(1.0, 1.0, 1.0)) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    m = m.copy()
    m.apply_scale(np.asarray(scale, float))
    return m


def mesh_volume(path: str, scale=(1.0, 1.0, 1.0)) -> float:
    """Volume of the convex hull (what MuJoCo collides with)."""
    m = load_mesh(path, scale)
    try:
        return float(m.convex_hull.volume)
    except Exception:  # degenerate (flat) meshes
        return 0.0


def coacd_parts(path: str, scale, params: dict, cache_dir: str) -> list[str]:
    """Convex decomposition with CoACD, cached by (file content, scale, params). Returns .obj part paths."""
    with open(path, "rb") as f:
        digest = hashlib.sha1(f.read())
    digest.update(json.dumps([list(map(float, scale)), params], sort_keys=True).encode())
    out = os.path.join(cache_dir, digest.hexdigest()[:16])
    done = os.path.join(out, "parts.json")
    if os.path.isfile(done):
        with open(done) as f:
            return [os.path.join(out, p) for p in json.load(f)]
    import coacd  # optional dependency, only needed when method == coacd

    m = load_mesh(path, scale)
    cm = coacd.Mesh(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64))
    parts = coacd.run_coacd(cm, **params)
    os.makedirs(out, exist_ok=True)
    names = []
    for k, (v, f) in enumerate(parts):
        name = f"part_{k:03d}.obj"
        trimesh.Trimesh(v, f, process=False).export(os.path.join(out, name))
        names.append(name)
    with open(done, "w") as f:
        json.dump(names, f)
    return [os.path.join(out, n) for n in names]

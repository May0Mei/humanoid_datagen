"""Room geometry for robot-start sampling (any task): downward floor rays, inside-the-walls rays and the floor's free
rectangles. The task decides what else a start needs (Task.spot_ok; push: a free corridor, tasks/push/spots.py).
"""
from __future__ import annotations

import mujoco
import numpy as np

HEIGHTS = (0.08, 0.45, 0.9, 1.3)


def _ray(m, d, p, v, geomgroup=None, exclude_body=-1):
    gid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(m, d, np.asarray(p, float), np.asarray(v, float), geomgroup, 1, exclude_body, gid)
    return dist, int(gid[0])


def _floor_ok(m, d, xy, tol, exclude_body=-1):
    dist, g = _ray(m, d, [xy[0], xy[1], 2.0], [0, 0, -1], exclude_body=exclude_body)
    return g >= 0 and abs((2.0 - dist)) < tol


def inside_room(m, d, xy, exclude_body=-1, n_dir=16, reach=15.0, heights=(0.3, 1.0, 1.8, 2.6)):
    """Enclosed by the room: horizontal rays in `n_dir` directions each hit a geom within `reach` m at one of
    `heights` (so a window or doorway in one band does not count as open). Agent floors are often planes larger
    than the walled room, so floor rays alone accept points outside it."""
    for a in np.linspace(0, 2 * np.pi, n_dir, endpoint=False):
        v = [np.cos(a), np.sin(a), 0.0]
        if not any(0 <= r[0] < reach and r[1] >= 0
                   for r in (_ray(m, d, [xy[0], xy[1], h], v, exclude_body=exclude_body) for h in heights)):
            return False
    return True


def floor_boxes(m: mujoco.MjModel, d: mujoco.MjData, min_area: float = 0.5):
    """World xy boxes of floor-like geoms (top within 5 cm of z = 0, at least min_area m2)."""
    out = []
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        a = m.geom_aabb[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            hx, hy = m.geom_size[g, :2]
            if hx <= 0:
                continue
            c, h = d.geom_xpos[g], np.array([hx, hy, 0])
        else:
            c = d.geom_xpos[g] + R @ a[:3]
            h = np.abs(R) @ a[3:]
        if abs(c[2] + h[2]) < 0.05 and 4 * h[0] * h[1] >= min_area:
            out.append((c[:2] - h[:2], c[:2] + h[:2]))
    return out

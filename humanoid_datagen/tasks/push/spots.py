"""Push starts: a robot start (x, y, yaw) is usable when the whole push fits: floor under every point of the path and
a free corridor from `back` m behind the start to the end of the push (start-to-object gap + object depth + push
length + margin), checked with horizontal rays at several heights and three lateral offsets (object half width +
margin, at least the robot's shoulder half width). The same sampler (other seeds) serves training and the test set.
"""
from __future__ import annotations

import mujoco
import numpy as np

from humanoid_datagen.datagen_pipelines.shared.sampler import HEIGHTS, _floor_ok, _ray, floor_boxes


def corridor_ok(m, d, start, yaw, ahead, back, half_width, floor_tol=0.04, step=0.25, exclude_body=-1):
    """Free floor and no obstacle from `back` m behind `start` to `ahead` m in front, `half_width` to each side.
    `exclude_body` (a body id) is ignored by the rays: the test piece that is itself pushed along the corridor."""
    u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    n = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    s0 = np.array([start[0], start[1], 0.0]) - u * back
    L = ahead + back
    for t in np.arange(0, L + 1e-6, step):
        for o in (-half_width, 0.0, half_width):
            if not _floor_ok(m, d, (s0 + u * t + n * o)[:2], floor_tol, exclude_body):
                return False
    for h in HEIGHTS:
        for o in (-half_width, 0.0, half_width):
            p = s0 + n * o + np.array([0, 0, h])
            dist, g = _ray(m, d, p, u, exclude_body=exclude_body)
            if g >= 0 and dist < L:
                return False
    return True


def sample_tasks(room_model: mujoco.MjModel, n: int, rng: np.random.Generator, obj_size, push_len: float,
                 gap=(0.5, 0.9), back: float = 0.4, margin: float = 0.3, max_tries: int = 4000) -> list[dict]:
    """[{start, yaw_deg, object_distance}] with a free corridor for an object of size (D, W, H)."""
    d = mujoco.MjData(room_model)
    mujoco.mj_forward(room_model, d)
    boxes = floor_boxes(room_model, d)
    if not boxes:
        c, e = room_model.stat.center, float(room_model.stat.extent)
        boxes = [(c[:2] - e, c[:2] + e)]
    area = np.array([np.prod(hi - lo) for lo, hi in boxes])
    D, W, _ = obj_size
    hw = max(W / 2 + 0.15, 0.35)
    out = []
    for _ in range(max_tries):
        if len(out) >= n:
            break
        lo, hi = boxes[int(rng.choice(len(boxes), p=area / area.sum()))]
        xy = rng.uniform(lo, hi)
        yaw = rng.uniform(-np.pi, np.pi)
        g = float(rng.uniform(*gap))
        if not _floor_ok(room_model, d, xy, 0.04):
            continue
        if corridor_ok(room_model, d, xy, yaw, g + D + push_len + margin, back, hw):
            out.append(dict(start=[float(xy[0]), float(xy[1])], yaw_deg=float(np.degrees(yaw)),
                            object_distance=g))
    return out

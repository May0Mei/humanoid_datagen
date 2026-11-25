"""HALO planner for the push eval (moved here from envs/closed_loop/planners.py and models/lowlevel/interface.py,
now in cache/, unchanged): the plan container SONIC tracks (MotionChunkCommand) and HaloPlanner, which wraps HALO's
unmodified generator (ml-halo/generator): one ego frame + instruction -> 150-frame plan."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

MOTION_FPS = 30
QPOS_DIM = 36


def quat_slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


@dataclass
class MotionChunkCommand:
    """Reference motion window: ``frames[k]`` is the thor qpos at time ``t0 + k / fps``."""

    frames: np.ndarray                    # (T, 36)
    t0: float = 0.0
    fps: float = MOTION_FPS

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=np.float64)
        if self.frames.ndim != 2 or self.frames.shape[1] != QPOS_DIM:
            raise ValueError(f"motion chunk must be (T, {QPOS_DIM}), got {self.frames.shape}")

    @property
    def num_frames(self) -> int:
        return self.frames.shape[0]

    def frame_at(self, t: float) -> np.ndarray:
        """Interpolated reference qpos at time ``t`` (lerp pos/joints, slerp root quat; clamped)."""
        u = np.clip((t - self.t0) * self.fps, 0.0, self.num_frames - 1)
        i0 = int(np.floor(u))
        i1 = min(i0 + 1, self.num_frames - 1)
        a = u - i0
        f0, f1 = self.frames[i0], self.frames[i1]
        out = (1.0 - a) * f0 + a * f1
        out[3:7] = quat_slerp(f0[3:7], f1[3:7], a)
        return out


class HaloPlanner:
    """HALO generator (ml-halo/generator, unmodified): one ego frame + instruction -> 150-frame plan.

    Plans are yaw-gauged at the current pose: [29 joints, pelvis pos relative to frame 0 with yaw removed, pelvis quat
    with yaw removed]. The first prev_chunk_frames are inpainted with what the previous plan does next, relative to
    its own first frame (HALO's deployment rule); the first call holds the current pose, as its cold-start training
    samples do. Each plan is placed at the robot's measured pose (position and heading).
    """

    def __init__(self, hcfg: dict, log=print):
        import sys

        import torch

        from humanoid_datagen.utils.paths import repo_path, third_party

        sys.path.insert(0, third_party("ml-halo", "generator"))
        from configs.config import load_config
        from models.motion_generator import MotionGenerator, model_config_from

        self.torch = torch
        dev = hcfg["device"]
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if dev == "auto" else dev
        c = load_config(repo_path(hcfg["config"]))
        model = MotionGenerator(model_config_from(c))
        ck = torch.load(repo_path(hcfg["checkpoint"]), map_location="cpu", weights_only=False)
        res = model.load_state_dict(ck["model_state_dict"], strict=False)
        bad = [k for k in res.unexpected_keys if not k.startswith("_normalizer.")] + \
              [k for k in res.missing_keys if not k.startswith("vlm_encoder.")]
        if bad or model.normalizer is None:
            raise RuntimeError(f"checkpoint does not match {hcfg['config']}: {bad[:5]}, normalizer {model.normalizer}")
        self.model = model.to(self.device).eval()
        self.image_hw = tuple(c.data.image_size)
        self.prefix_n = int(c.model.prev_chunk_frames)
        self.fps = 30.0
        self.context_offsets_s = [0.0]
        self.seed0 = int(hcfg["seed"])
        self.plan = None  # (t0, world frames) of the previous plan
        self.prefix_mode = hcfg.get("prefix", "plan")
        if self.prefix_mode not in ("plan", "measured"):
            raise ValueError(f"halo.prefix must be plan or measured, got {self.prefix_mode}")
        self.info = dict(prefix=self.prefix_mode,planner="halo", checkpoint=hcfg["checkpoint"], horizon=int(c.model.trajectory_length),
                         chunk_frame="yaw_gauged", head="dit_flow", context_offsets_s=[0.0], image_hw=list(self.image_hw))

    def reset(self, task: str, seed: int) -> None:
        self.torch.manual_seed(self.seed0 * 1000003 + int(seed))
        self.plan = None

    def _prefix_world(self, cur: np.ndarray, t: float) -> np.ndarray:
        """The next prefix_n world frames: the previous plan from time t on, or a hold of cur."""
        if self.plan is None:
            return np.repeat(cur[None], self.prefix_n, axis=0)
        t0, frames = self.plan
        i = int(round((t - t0) * self.fps)) + np.arange(self.prefix_n)
        return frames[np.minimum(i, len(frames) - 1)]

    def act(self, obs: dict) -> dict:
        if not obs["replan"]:
            return {"chunk": None}
        import torchvision.transforms.functional as TF

        t = time.perf_counter()
        torch = self.torch
        cur = np.asarray(obs["proprio_qpos"], np.float64)
        img = torch.from_numpy(np.asarray(obs["images"][-1]).copy()).permute(2, 0, 1).float() / 255.0
        img = (TF.resize(img, list(self.image_hw)) * 255).to(torch.uint8).float() / 255.0  # = training frames (uint8 .pt)
        world = self._prefix_world(cur, float(obs["time"]))
        prefix = world_to_gauged(world, world[0])
        if self.prefix_mode == "measured" and self.plan is not None:
            prefix = measured_prefix(prefix, cur)
        prefix = self.model.normalizer.normalize(torch.from_numpy(prefix).float())
        pred = self.model.predict(img.to(self.device), obs["instruction"], prev_chunk=prefix.to(self.device))
        # Anchor the plan at the robot's measured pose, not at the previous plan's world frame (world[0]): the prefix
        # carries the previous plan's motion relative to its own frame 0 (inpainting = a smooth hand-over), but the
        # new plan starts where the robot is. Anchoring at world[0] let the plan run ahead of a lagging robot without
        # bound (preview 2026-11-21: plan 1.7-2.0 m ahead of the robot after 12 s, robot crouched in place).
        frames = gauged_to_world(pred.float().cpu().numpy().astype(np.float64), cur)
        frames[0] = cur
        self.plan = (float(obs["time"]), frames)
        return {"chunk": MotionChunkCommand(frames, t0=float(obs["time"]), fps=self.fps), "plan_s": time.perf_counter() - t}


def _yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _rot_z(xy_z: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    out = xy_z.copy()
    out[:, 0], out[:, 1] = c * xy_z[:, 0] - s * xy_z[:, 1], s * xy_z[:, 0] + c * xy_z[:, 1]
    return out


def world_to_gauged(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """World qpos (T, 36) -> HALO frames (T, 36), gauged at qpos ref (ml-halo/generator/data/traj_extract.py)."""
    yaw = _yaw(ref[3:7])
    qz_inv = np.array([np.cos(-yaw / 2), 0, 0, np.sin(-yaw / 2)])
    return np.concatenate([q[:, 7:], _rot_z(q[:, :3] - ref[:3], -yaw), _quat_mul(qz_inv, q[:, 3:7])], axis=1)


def measured_prefix(g: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """halo.prefix = measured: the prefix (HALO frames, gauged at its own frame 0) re-based to start at the robot's
    measured pose cur (world qpos). Frame 0 = cur exactly (joints, pelvis roll/pitch); the gap to the previous plan
    fades out linearly over the prefix, so its last frame is the previous plan's again (the shape of HALO's cold-start
    samples, a hold of the first frame blended into the plan). Pelvis position stays the plan's motion relative to
    frame 0, which already starts at the robot."""
    c = world_to_gauged(np.asarray(cur, np.float64)[None], cur)[0]
    n = len(g)
    w = 1.0 - np.arange(n) / max(n - 1, 1)                 # 1 at frame 0 -> 0 at the last prefix frame
    out = g.copy()
    out[:, :29] += w[:, None] * (c[:29] - g[0, :29])
    for k in range(n):
        out[k, 32:36] = quat_slerp(g[k, 32:36], c[32:36], w[k])
    return out


def gauged_to_world(g: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Inverse of world_to_gauged, with unit quaternions."""
    yaw = _yaw(ref[3:7])
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    quat = _quat_mul(qz, g[:, 32:36])
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    return np.concatenate([ref[:3] + _rot_z(g[:, 29:32], yaw), quat, g[:, :29]], axis=1)



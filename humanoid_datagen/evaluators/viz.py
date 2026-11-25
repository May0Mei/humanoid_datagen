"""Visualize push-eval episodes: the planner's reference motion next to what GEAR-SONIC tracked in physics.

    python humanoid_datagen/scripts/eval.py --config_name eval/push/viz   (GPU: EGL; config key viz: roots, fps, success_only, ids)

Per episode in <results_root>/episodes.jsonl (trajectory npz from evaluators/grscenes.run_episode), writes next to the npz:
  <stem>_viz.mp4    3 panels: third-person TRACKED (SONIC, physics; the object as simulated) | third-person
                    REFERENCE (the active HALO chunk at that time, frame_at as SONIC is fed it; object at its
                    simulated pose) | the robot's ego view (calibrated OAK-1 fisheye, the planner's input camera).
                    Third-person camera: per episode, the azimuth / elevation / distance with the clearest line of
                    sight to the robot and object (ray casts over the episode), kept under the ceiling. Nothing in
                    the scene is hidden or cut away.
  <stem>_plot.png   top-down pelvis paths (tracked vs every chunk), joint tracking error, object push distance
  <stem>_sheet.png  6 time samples of the 3 panels
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import mujoco
import numpy as np

from humanoid_datagen.vla.halo import MotionChunkCommand
from humanoid_datagen.utils.oak_cam import OakCam
from humanoid_datagen.utils.sonic import JOINT_NAMES, robot_free_joint

W, H = 640, 400
HIDE = 5                                   # geom group the default scene option does not draw: cut-away geometry
TARGET_BODIES = ("pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "left_ankle_roll_link",
                 "right_ankle_roll_link")


def cam_pos(lookat, dist, az, el):
    """World position of a MuJoCo free camera (looks along forward = (cos el cos az, cos el sin az, sin el))."""
    a, e = np.radians(az), np.radians(el)
    return np.asarray(lookat) - dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


class LineOfSight:
    """Ray casts from a camera to the robot and the object against the drawn geometry (groups 0-2). A hit on a room
    geom before the target = an occluder; `occluders` repeats the cast with each occluder hidden until clear."""

    def __init__(self, model, data, robot_root: int, obj_body: int):
        self.m, self.d = model, data
        self.mine = {g for g in range(model.ngeom)
                     if model.body_rootid[model.geom_bodyid[g]] in (robot_root, model.body_rootid[obj_body])}
        self.tb = [model.body(n).id for n in TARGET_BODIES if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0]
        self.ob = obj_body
        self.gid = np.zeros(1, np.int32)

    def targets(self):
        d, ob = self.d, self.ob
        return [d.xpos[b] for b in self.tb] + [d.xipos[ob], d.xipos[ob] + [0, 0, 0.25]]

    def occluders(self, pos, max_iter: int = 6) -> set[int]:
        m, d, hidden = self.m, self.d, set()
        grp = np.array([1, 1, 1, 0, 0, 0], np.uint8)
        saved = m.geom_group.copy()
        try:
            for tgt in self.targets():
                v = tgt - pos
                dist = float(np.linalg.norm(v))
                for _ in range(max_iter):
                    x = mujoco.mj_ray(m, d, pos, v / dist, grp, 1, -1, self.gid)
                    g = int(self.gid[0])
                    if x < 0 or x > dist - 0.05 or g in self.mine or m.geom_group[g] == 0:   # clear / target / floor
                        break
                    hidden.add(g)
                    m.geom_group[g] = HIDE
        finally:
            m.geom_group[:] = saved
        return hidden

    def blocked(self, pos) -> int:
        """Targets whose first hit is room geometry (no cut-away)."""
        m, d, n = self.m, self.d, 0
        grp = np.array([1, 1, 1, 0, 0, 0], np.uint8)
        for tgt in self.targets():
            v = tgt - pos
            dist = float(np.linalg.norm(v))
            x = mujoco.mj_ray(m, d, pos, v / dist, grp, 1, -1, self.gid)
            g = int(self.gid[0])
            n += bool(x >= 0 and x < dist - 0.05 and g not in self.mine and m.geom_group[g] != 0)
        return n


def ceiling_z(los: LineOfSight, data, qposes, lookats) -> float:
    """Lowest room geometry straight above the look-at points over the episode (inf when open above)."""
    m, grp, z = los.m, np.array([1, 1, 1, 0, 0, 0], np.uint8), np.inf
    for q, at in zip(qposes, lookats):
        data.qpos[:] = q
        mujoco.mj_kinematics(m, data)
        x = mujoco.mj_ray(m, data, np.asarray(at, float), np.array([0.0, 0.0, 1.0]), grp, 1, -1, los.gid)
        if x >= 0 and int(los.gid[0]) not in los.mine:
            z = min(z, float(at[2]) + x)
    return z


def choose_view(los: LineOfSight, data, qposes, lookats, pref_az: float):
    """(azimuth, elevation, distance, blocked rays): the camera with the fewest blocked robot / object rays over the
    episode, kept 0.15 m under the ceiling (nothing is cut away, so it must see from inside the room). Score = share of
    rays blocked (x3) + distance from pref_az (behind and beside the push) + distance from 3.4 m + elevation off -30:
    a few blocked rays (a chair leg) do not push the camera into a close, flat view."""
    top = ceiling_z(los, data, qposes, lookats) - 0.15
    best = None
    for dist in (5.0, 4.2, 3.4, 2.8, 2.2, 1.7):
        for el in (-12.0, -20.0, -30.0, -40.0, -55.0):
            if max(cam_pos(at, dist, 0.0, el)[2] for at in lookats) > top:
                continue
            for daz in range(0, 360, 15):
                az = pref_az + daz
                n = 0
                for q, at in zip(qposes, lookats):
                    data.qpos[:] = q
                    mujoco.mj_kinematics(los.m, data)
                    n += los.blocked(cam_pos(at, dist, az, el))
                off = min(daz, 360 - daz) / 180.0
                frac = n / max(1, len(qposes) * len(los.targets()))    # share of robot / object rays blocked
                score = 3.0 * frac + 0.5 * off + 0.15 * abs(3.4 - dist) + 0.1 * (el != -30.0)
                if best is None or score < best[0]:
                    best = (score, az, el, dist, n)
    if best is None:      # very low ceiling: closest, flattest view
        return pref_az, -15.0, 1.7, -1
    return best[1], best[2], best[3], best[4]


def reference_at(chunks: list[MotionChunkCommand], t: float) -> np.ndarray:
    """The reference SONIC is given at time t: the latest chunk issued at or before t (evaluators/grscenes' `active`)."""
    c = [c for c in chunks if c.t0 <= t + 1e-9][-1]
    return c.frame_at(t)


def label(img, lines, color=(255, 255, 255)):
    for i, s in enumerate(lines):
        y = 26 + 24 * i
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return img


def episode(task: dict, res: dict, fps: float, tk) -> None:
    from humanoid_datagen.datagen_pipelines.shared.compose import episode_frame
    from humanoid_datagen.datagen_pipelines.shared.run import load_traj
    z = load_traj(res["trajectory"])
    yaw_deg = episode_frame(task)[1]
    qpos, t = z["qpos"], z["t"]
    chunks = [MotionChunkCommand(f, t0=float(t0), fps=30.0) for t0, f in zip(z["chunk_t0"], z["chunk_frames"])]
    model, _, _, info = tk.compose_task(task)
    fj = robot_free_joint(model)
    qa = model.jnt_qposadr[fj.id]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES])
    oak = OakCam(model, "oak1")                      # edits oak1 + offscreen size before the renderer is made
    rnd = mujoco.Renderer(model, H, W)
    data = mujoco.MjData(model)
    obj0, d = z["obj0"], z["direction"]
    step = max(1, int(round(1.0 / fps / (t[1] - t[0]))))
    # the camera follows the tracked robot (both panels: the reference's offset from the robot stays visible)
    look, looks = None, []
    for k in range(0, len(t), step):
        tgt = np.array([*(qpos[k][qa:qa + 2] + 0.5 * d), 0.6])
        look = tgt if look is None else 0.85 * look + 0.15 * tgt       # smoothed follow
        looks.append(look)
    los = LineOfSight(model, data, model.jnt_bodyid[fj.id], model.body(info["object_body"]).id)
    sel = slice(None, None, max(1, int(round(0.5 * fps))))              # view choice: every 0.5 s
    az, el, dist, nblk = choose_view(los, data, qpos[::step][sel], looks[sel], yaw_deg + 50.0)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = dist, el, az
    print(f"[viz] {task['task_id']}: view az {(az - yaw_deg) % 360:.0f} deg from push, el {el:.0f}, "
          f"dist {dist:.1f} m ({nblk} blocked rays; nothing cut away)", flush=True)

    def third(q, at):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        cam.lookat[:] = at
        rnd.update_scene(data, camera=cam)
        return rnd.render().copy()

    disp = (z["obj_pos"][:, :2] - obj0[:2]) @ d
    frames, ref_traj, jerr = [], [], []
    for k in range(len(t)):
        ref = reference_at(chunks, float(t[k]))
        ref_traj.append(ref)
        jerr.append(np.degrees(np.sqrt(np.mean((qpos[k][qadr] - ref[7:]) ** 2))))
        if k % step:
            continue
        look = looks[k // step]
        a = third(qpos[k], look)
        qr = qpos[k].copy()
        qr[qa:qa + 7], qr[qadr] = ref[:7], ref[7:]
        b = third(qr, look)
        data.qpos[:] = qpos[k]
        mujoco.mj_forward(model, data)
        e = cv2.resize(oak.render(data), (W, H), interpolation=cv2.INTER_AREA)
        ci = sum(c.t0 <= t[k] + 1e-9 for c in chunks)
        label(a, ["TRACKED (GEAR-SONIC, physics)", f"t={t[k]:.2f}s  push={disp[k]:+.2f}m"])
        label(b, ["REFERENCE (HALO chunk)", f"chunk {ci}/{len(chunks)}  joint err {jerr[-1]:.1f} deg"], (120, 220, 255))
        label(e, ["EGO (OAK-1, planner input)", task["instruction"]])
        frames.append(np.concatenate([a, b, e], axis=1))
    rnd.close()
    oak.close()
    stem = res["trajectory"][:-4]
    import imageio.v2 as imageio
    imageio.mimsave(stem + "_viz.mp4", frames, fps=fps, macro_block_size=8)
    idx = np.linspace(0, len(frames) - 1, 6).round().astype(int)
    imageio.imwrite(stem + "_sheet.png", np.concatenate([cv2.resize(frames[i], (W * 3 // 2, H // 2)) for i in idx], 0))
    plot(task, res, z, np.array(ref_traj), np.array(jerr), disp, chunks, qa, stem + "_plot.png")


def plot(task, res, z, ref, jerr, disp, chunks, qa, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, qpos = z["t"], z["qpos"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for i, c in enumerate(chunks):
        ax[0].plot(c.frames[:, 0], c.frames[:, 1], color=plt.cm.viridis(i / max(len(chunks) - 1, 1)), lw=0.8, alpha=0.6)
    ax[0].plot(ref[:, 0], ref[:, 1], "--", color="tab:blue", lw=1.5, label="reference fed to SONIC")
    ax[0].plot(qpos[:, qa], qpos[:, qa + 1], color="tab:red", lw=2, label="tracked pelvis")
    ax[0].plot(z["obj_pos"][:, 0], z["obj_pos"][:, 1], color="tab:brown", lw=2, label="object")
    ax[0].scatter(*z["obj0"][:2], color="tab:brown", marker="s")
    ax[0].scatter(qpos[0, qa], qpos[0, qa + 1], color="k", marker="o", label="start")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].legend(fontsize=7)
    ax[0].set_title("top-down: thin = each HALO chunk (150 frames, colour = time)")
    ax[1].plot(t, jerr, color="tab:purple")
    for c in chunks:
        ax[1].axvline(c.t0, color="0.85", lw=0.5, zorder=0)
    ax[1].set_xlabel("t (s)")
    ax[1].set_ylabel("joint RMS error, tracked vs reference (deg)")
    ax[1].set_title("tracking error (grey = replans)")
    pe = np.linalg.norm(qpos[:, qa:qa + 2] - ref[:, :2], axis=1)
    ax1b = ax[1].twinx()
    ax1b.plot(t, pe, color="tab:orange", lw=1)
    ax1b.set_ylabel("pelvis xy error (m)", color="tab:orange")
    ax[2].plot(t, disp, color="tab:brown", label="object push distance")
    ax[2].axhline(0.3, ls="--", color="k", lw=0.8, label="success 0.3 m")
    ax[2].plot(t, qpos[:, qa + 2], color="tab:red", lw=1, label="tracked pelvis z")
    ax[2].plot(t, ref[:, 2], "--", color="tab:blue", lw=1, label="reference pelvis z")
    ax[2].set_xlabel("t (s)")
    ax[2].legend(fontsize=7)
    fig.suptitle(f"{res['task_id']} s{res['seed']}  {task['instruction']}  success={res['success']} "
                 f"push={res['push_dist']:.2f}m tilt={res['max_tilt_deg']:.0f}° fell={res['fell']}")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def run(C: dict, tk) -> None:
    """Review videos of saved eval episodes (eval mode `viz`): C["viz"] = {roots, fps, success_only, ids}."""
    from humanoid_datagen.utils.paths import repo_path
    V = C["viz"]
    T = {t["task_id"]: t for t in map(json.loads, open(repo_path(C["tasks"]["file"])))}
    for root in V["roots"]:
        recs = [json.loads(l) for l in open(os.path.join(repo_path(root), "episodes.jsonl"))]
        if V.get("success_only"):
            recs = [r for r in recs if r["success"]]
        if V.get("ids") is not None:
            want = set(V["ids"])
            recs = [r for r in recs if f"{r['task_id']}:{r['seed']}" in want]
        for r in recs:
            episode(T[r["task_id"]], r, float(V.get("fps", 25.0)), tk)
            print(f"[viz] {r['task_id']} s{r['seed']} -> {r['trajectory'][:-4]}_viz.mp4", flush=True)

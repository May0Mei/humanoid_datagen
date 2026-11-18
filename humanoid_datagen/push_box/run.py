"""Physics rollout of a staged push-box scene (docs/push_box.md, step 4).

settle: the robot is held kinematically at frame 0 while the box drops and comes to rest;
track:  GEAR-SONIC tracks the Kimodo reference at 50 Hz (PD at 200 Hz) with the box simulated;
kinematic: the same reference replayed kinematically (robot = infinite-mass collider), for comparison.
Every rollout saves the raw trajectory so success criteria can be rescored later.
"""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np

from .sonic import Reference, SonicTracker, set_robot_pose, qmat

DECIM = 4  # 200 Hz physics / 50 Hz policy


def _box_ids(model, body="box"):
    b = model.body(body).id
    j = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    return b, model.jnt_qposadr[j], model.jnt_dofadr[j]


def _object_geom_mask(model, b):
    """Geoms of body b and its descendants."""
    mask = np.zeros(model.ngeom, bool)
    for g in range(model.ngeom):
        p = model.geom_bodyid[g]
        while p > 0:
            if p == b:
                mask[g] = True
                break
            p = model.body_parentid[p]
    return mask


# Fall rule, shared by the agent tools, datagen acceptance and the eval: the robot fell when its head (helmet) drops
# below FALL_HEAD_Z. Squat pushes keep the head >= ~0.6 m; real falls end with the head at ~0.05-0.1 m.
HEAD_BODY = "helmet_assembly_link"
FALL_HEAD_Z = 0.4
# Success rule, identical in training (agent simulate / finalize, datagen acceptance) and test (eval_push).
PASS = dict(min_push=0.3, max_tilt_deg=20.0)


def head_z(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Head height for each row of qpos."""
    d = mujoco.MjData(model)
    hb = model.body(HEAD_BODY).id
    out = np.empty(len(qpos))
    for i, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(model, d)
        out[i] = d.xpos[hb, 2]
    return out


def _hand_box_force(model, data, obj_mask, robot_geom_mask):
    f = np.zeros(6)
    tot = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if (obj_mask[c.geom1] and robot_geom_mask[c.geom2]) or (obj_mask[c.geom2] and robot_geom_mask[c.geom1]):
            mujoco.mj_contactForce(model, data, i, f)
            tot += abs(f[0])
    return tot


def rollout(model: mujoco.MjModel, ref: Reference, mode: str = "track", settle_s: float = 0.5,
            tail_s: float = 1.0, record_every: int = 1, object_body: str = "box") -> dict:
    """mode: 'track' (SONIC physics) or 'kinematic' (replay). Returns the trajectory dict."""
    data = mujoco.MjData(model)
    bid, bq, bd = _box_ids(model, object_body)
    box_geom = _object_geom_mask(model, bid)
    rmask = model.geom_contype == 2
    set_robot_pose(model, data, ref, 0)
    tr = SonicTracker(model, ref)
    rq = np.r_[np.arange(tr.base_q, tr.base_q + 7), tr.qadr]
    rd = np.r_[np.arange(tr.base_d, tr.base_d + 6), tr.dadr]
    q0 = data.qpos[rq].copy()
    # settle the box with the robot frozen
    for _ in range(int(settle_s / model.opt.timestep)):
        data.qpos[rq] = q0
        data.qvel[rd] = 0
        mujoco.mj_step(model, data)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    box0 = data.qpos[bq:bq + 7].copy()

    tr.reset(data)
    n_ctrl = ref.T + int(tail_s * 50)
    rec = {k: [] for k in ("t", "qpos", "box_pos", "box_quat", "hand_box_force", "ref_frame")}
    fpos = np.zeros(n_ctrl)
    for k in range(n_ctrl):
        i = min(k, ref.T - 1)
        if mode == "track":
            tr.act(data)
            for _ in range(DECIM):
                data.qfrc_applied[tr.dadr] = tr.pd_torque(data)
                mujoco.mj_step(model, data)
        else:
            # kinematic replay: robot pose and velocity written before each substep
            i1 = min(k + 1, ref.T - 1)
            for s in range(DECIM):
                a = (s + 1) / DECIM
                data.qpos[tr.base_q:tr.base_q + 3] = ref.root_pos[i] * (1 - a) + ref.root_pos[i1] * a
                data.qpos[tr.base_q + 3:tr.base_q + 7] = ref.root_quat[i1]
                data.qpos[tr.qadr] = ref.joint_pos[i] * (1 - a) + ref.joint_pos[i1] * a
                data.qvel[tr.base_d:tr.base_d + 3] = (ref.root_pos[i1] - ref.root_pos[i]) * 50
                data.qvel[tr.base_d + 3:tr.base_d + 6] = 0
                data.qvel[tr.dadr] = ref.joint_vel[i]
                data.qacc_warmstart[:] = 0
                mujoco.mj_step(model, data)
        fpos[k] = _hand_box_force(model, data, box_geom, rmask)
        if k % record_every == 0:
            rec["t"].append((k + 1) * DECIM * model.opt.timestep)
            rec["qpos"].append(data.qpos.copy())
            rec["box_pos"].append(data.xpos[bid].copy())
            rec["box_quat"].append(data.xquat[bid].copy())
            rec["hand_box_force"].append(fpos[k])
            rec["ref_frame"].append(min(k + 1, ref.T - 1))
    out = {k: np.array(v) for k, v in rec.items()}
    out["box0"] = box0
    out["robot_qadr"] = tr.qadr
    out["base_q"] = tr.base_q
    return out


def metrics(traj: dict, ref: Reference, push_dir, model) -> dict:
    d = np.array([push_dir[0], push_dir[1]])
    bp = traj["box_pos"]
    disp = (bp[:, :2] - traj["box0"][:2]) @ d
    up = qmat(traj["box_quat"])[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    q = traj["qpos"]
    bq = int(traj["base_q"])
    pel_z = q[:, bq + 2]
    hz = head_z(model, q)
    fr = traj["ref_frame"]
    jerr = np.abs(q[:, traj["robot_qadr"]] - ref.joint_pos[fr]).mean(1)
    rerr = np.linalg.norm(q[:, bq:bq + 2] - ref.root_pos[fr, :2], axis=1)
    fell = bool((hz < FALL_HEAD_Z).any())
    in_push = traj["hand_box_force"] > 1.0
    return dict(
        box_push_dist=float(disp[-1]), box_max_tilt_deg=float(tilt.max()), box_lateral_drift=float(
            abs((bp[-1, :2] - traj["box0"][:2]) @ np.array([-d[1], d[0]]))),
        robot_fell=fell, min_pelvis_z=float(pel_z.min()), min_head_z=float(hz.min()),
        joint_err_mean=float(jerr.mean()), joint_err_max=float(jerr.max()),
        root_xy_err_mean=float(rerr.mean()), root_xy_err_final=float(rerr[-1]),
        contact_frac=float(in_push.mean()), hand_box_force_mean_in_contact=float(
            traj["hand_box_force"][in_push].mean() if in_push.any() else 0.0),
        hand_box_force_max=float(traj["hand_box_force"].max()),
    )


def render(model: mujoco.MjModel, traj: dict, path: str, camera: str, size=(640, 360), fps=50, stride=1):
    import imageio.v2 as imageio
    data = mujoco.MjData(model)
    r = mujoco.Renderer(model, height=size[1], width=size[0])
    opt = mujoco.MjvOption()
    with imageio.get_writer(path, fps=fps // stride, codec="libx264", quality=8) as w:
        for k in range(0, len(traj["qpos"]), stride):
            data.qpos[:] = traj["qpos"][k]
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=camera, scene_option=opt)
            w.append_data(r.render())
    r.close()


def save(traj: dict, out_dir: str, name: str, m: dict):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"{name}_traj.npz"), **traj)
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)

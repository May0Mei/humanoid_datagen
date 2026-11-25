"""Physics rollout of a staged scene with the GEAR-SONIC tracker (any task; a task's metrics read the trajectory).

settle: the robot is held kinematically at frame 0 while the object drops and comes to rest;
track:  GEAR-SONIC tracks the Kimodo reference at 50 Hz (PD at 200 Hz) with the object simulated;
kinematic: the same reference replayed kinematically (robot = infinite-mass collider), for comparison.
Trajectory (rollout): t, qpos, ref_frame, robot_qadr, base_q, and with an object: obj_pos, obj_quat (per step), obj0
(its settled qpos), hand_obj_force (robot-object contact normal force). Every rollout keeps the raw trajectory so
success criteria can be rescored later; load_traj reads older files too (box_* keys, push_dir).
"""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np

from humanoid_datagen.utils.sonic import Reference, SonicTracker, set_robot_pose, qmat

DECIM = 4  # 200 Hz physics / 50 Hz policy


def _object_ids(model, body: str):
    """Body id, qpos and dof address of the object's free joint."""
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


def _contact_force(model, data, obj_mask, robot_geom_mask):
    """Total normal force between the robot and the object."""
    f = np.zeros(6)
    tot = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if (obj_mask[c.geom1] and robot_geom_mask[c.geom2]) or (obj_mask[c.geom2] and robot_geom_mask[c.geom1]):
            mujoco.mj_contactForce(model, data, i, f)
            tot += abs(f[0])
    return tot


def rollout(model: mujoco.MjModel, ref: Reference, mode: str = "track", settle_s: float = 0.5,
            tail_s: float = 1.0, record_every: int = 1, object_body: str | None = None) -> dict:
    """mode: 'track' (SONIC physics) or 'kinematic' (replay). object_body: the free object to record (None: robot
    only). Returns the trajectory dict (module docstring)."""
    data = mujoco.MjData(model)
    if object_body is not None:
        bid, bq, bd = _object_ids(model, object_body)
        obj_geom = _object_geom_mask(model, bid)
    rmask = model.geom_contype == 2
    set_robot_pose(model, data, ref, 0)
    tr = SonicTracker(model, ref)
    rq = np.r_[np.arange(tr.base_q, tr.base_q + 7), tr.qadr]
    rd = np.r_[np.arange(tr.base_d, tr.base_d + 6), tr.dadr]
    q0 = data.qpos[rq].copy()
    # settle the object with the robot frozen
    for _ in range(int(settle_s / model.opt.timestep)):
        data.qpos[rq] = q0
        data.qvel[rd] = 0
        mujoco.mj_step(model, data)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    obj0 = data.qpos[bq:bq + 7].copy() if object_body is not None else None

    tr.reset(data)
    n_ctrl = ref.T + int(tail_s * 50)
    rec = {k: [] for k in ("t", "qpos", "ref_frame") + (("obj_pos", "obj_quat", "hand_obj_force") if object_body else ())}
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
        if object_body is not None:
            fpos[k] = _contact_force(model, data, obj_geom, rmask)
        if k % record_every == 0:
            rec["t"].append((k + 1) * DECIM * model.opt.timestep)
            rec["qpos"].append(data.qpos.copy())
            rec["ref_frame"].append(min(k + 1, ref.T - 1))
            if object_body is not None:
                rec["obj_pos"].append(data.xpos[bid].copy())
                rec["obj_quat"].append(data.xquat[bid].copy())
                rec["hand_obj_force"].append(fpos[k])
    out = {k: np.array(v) for k, v in rec.items()}
    if object_body is not None:
        out["obj0"] = obj0
    out["robot_qadr"] = tr.qadr
    out["base_q"] = tr.base_q
    return out


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


OLD_KEYS = {"box_pos": "obj_pos", "box_quat": "obj_quat", "box0": "obj0", "hand_box_force": "hand_obj_force",
            "push_dir": "direction"}


def load_traj(path: str) -> dict:
    """A saved trajectory (.npz) as a dict with the current keys; files written before the rename (datagen traj.npz:
    box_*; eval trajectories: push_dir) get their keys mapped."""
    z = dict(np.load(path, allow_pickle=True))
    for old, new in OLD_KEYS.items():
        if old in z and new not in z:
            z[new] = z.pop(old)
    return z

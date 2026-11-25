"""Push task success criteria: the pass rule and rollout metrics (shared by the agent tools, datagen acceptance and the
eval), and hand-contact attribution: which robot parts moved the push object (datagen acceptance, eval rescoring).

Replays a trajectory's qpos (robot + object) with collisions limited to robot <-> object and a `MARGIN` contact margin
on the object, and credits every step's forward object displacement (along the push direction) to the parts
touching it that step:
  hand  = wrist links (the G1 has no hand colliders: wrist_pitch / wrist_yaw are the hand); `both_m` is the part of
          the hand credit with both hands touching
  arm   = elbow / shoulder (no hand touching)
  leg   = hip / knee / ankle (no hand or arm touching)
  body  = torso / pelvis only
  none  = nothing touching (the object coasting after a shove)
Hand rule (RULE): hands moved it >= min_hand_m, legs + torso < max_leg_body_m, and both hands touching for at least
min_both_frac of the hand credit.
"""
from __future__ import annotations

import copy

import mujoco
import numpy as np

from humanoid_datagen.datagen_pipelines.shared.run import FALL_HEAD_Z, head_z
from humanoid_datagen.utils.sonic import Reference, qmat

# The pass rule itself (push >= min_push, tilt <= max_tilt_deg, no fall) is PushTask.pass_rule, set in config.


def metrics(traj: dict, ref: Reference, push_dir, model) -> dict:
    d = np.array([push_dir[0], push_dir[1]])
    bp = traj["obj_pos"]
    disp = (bp[:, :2] - traj["obj0"][:2]) @ d
    up = qmat(traj["obj_quat"])[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    q = traj["qpos"]
    bq = int(traj["base_q"])
    pel_z = q[:, bq + 2]
    hz = head_z(model, q)
    fr = traj["ref_frame"]
    jerr = np.abs(q[:, traj["robot_qadr"]] - ref.joint_pos[fr]).mean(1)
    rerr = np.linalg.norm(q[:, bq:bq + 2] - ref.root_pos[fr, :2], axis=1)
    fell = bool((hz < FALL_HEAD_Z).any())
    in_push = traj["hand_obj_force"] > 1.0
    return dict(
        push_dist=float(disp[-1]), max_tilt_deg=float(tilt.max()), lateral_drift=float(
            abs((bp[-1, :2] - traj["obj0"][:2]) @ np.array([-d[1], d[0]]))),
        robot_fell=fell, min_pelvis_z=float(pel_z.min()), min_head_z=float(hz.min()),
        joint_err_mean=float(jerr.mean()), joint_err_max=float(jerr.max()),
        root_xy_err_mean=float(rerr.mean()), root_xy_err_final=float(rerr[-1]),
        contact_frac=float(in_push.mean()), hand_force_mean_in_contact=float(
            traj["hand_obj_force"][in_push].mean() if in_push.any() else 0.0),
        hand_force_max=float(traj["hand_obj_force"].max()),
    )

MARGIN = 0.02
RULE = dict(min_hand_m=0.3, max_leg_body_m=0.1, min_both_frac=0.0)
ORDER = ("hand", "arm", "leg", "body")


def part(name: str) -> str:
    if "wrist" in name:
        return "hand"
    if "elbow" in name or "shoulder" in name:
        return "arm"
    if "hip" in name or "knee" in name or "ankle" in name:
        return "leg"
    return "body"


def attribution_model(model: mujoco.MjModel, object_body: str = "push_object", robot_contype: int = 2):
    """A copy of `model` with collisions limited to robot <-> object; returns (model, object geom mask,
    {robot geom: (part, side)}) with side 'l' / 'r' for the hands (None otherwise)."""
    m = copy.deepcopy(model)
    ob = m.body(object_body).id
    kids = {ob}
    for b in range(m.nbody):
        p = b
        while p > 0:
            if p in kids:
                kids.add(b)
                break
            p = m.body_parentid[p]
    obj = np.array([m.geom_bodyid[g] in kids and bool(m.geom_contype[g] or m.geom_conaffinity[g])
                    for g in range(m.ngeom)])
    rob = m.geom_contype == robot_contype
    for g in range(m.ngeom):
        if not (obj[g] or rob[g]):
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
    m.geom_margin[obj] = MARGIN
    gpart = {}
    for g in np.where(rob)[0]:
        n = m.body(m.geom_bodyid[g]).name
        p = part(n)
        gpart[int(g)] = (p, ("l" if n.startswith("left") else "r") if p == "hand" else None)
    return m, obj, gpart


def attribute(m, obj, gpart, qpos: np.ndarray, obj_xy: np.ndarray, push_dir, obj0_xy=None) -> dict:
    """qpos (T, nq) and the object's xy (T, 2) per step; obj0_xy: its xy before step 0 (default: step 0's)."""
    d = np.asarray(push_dir, float)[:2]
    s = np.asarray(obj_xy)[:, :2] @ d
    s0 = float(np.asarray(obj0_xy)[:2] @ d) if obj0_xy is not None else s[0]
    disp = np.diff(s, prepend=s0)
    data = mujoco.MjData(m)
    acc = dict(hand=0.0, arm=0.0, leg=0.0, body=0.0, none=0.0)
    both = 0.0
    frames = dict(hand_l=0, hand_r=0, both=0)
    for k in range(len(qpos)):
        data.qpos[:] = qpos[k]
        mujoco.mj_kinematics(m, data)
        mujoco.mj_collision(m, data)
        parts, sides = set(), set()
        for i in range(data.ncon):
            c = data.contact[i]
            if not (obj[c.geom1] or obj[c.geom2]):
                continue
            g = int(c.geom2 if obj[c.geom1] else c.geom1)
            if g in gpart:
                p, sd = gpart[g]
                parts.add(p)
                if sd:
                    sides.add(sd)
        frames["hand_l"] += "l" in sides
        frames["hand_r"] += "r" in sides
        frames["both"] += len(sides) == 2
        who = next((p for p in ORDER if p in parts), "none")
        if disp[k] > 0:
            acc[who] += float(disp[k])
            if len(sides) == 2:
                both += float(disp[k])
    out = {f"{p}_m": v for p, v in acc.items()}
    out.update(both_m=both, frames=len(qpos), **{f"{k}_frames": v for k, v in frames.items()})
    return out


def passes(r: dict, rule: dict | None = None) -> bool:
    R = dict(RULE, **(rule or {}))
    return bool(r["hand_m"] >= R["min_hand_m"] and r["leg_m"] + r["body_m"] < R["max_leg_body_m"]
                and r["both_m"] >= R["min_both_frac"] * r["hand_m"])

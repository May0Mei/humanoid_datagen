"""Stage the box against a Kimodo push motion (docs/push_box.md, step 3).

Kinematic analysis of the reference in its own frame (frame-0 root at the origin):
  * hand geoms = the G1 rubber-hand collision capsules;
  * push direction d = horizontal root travel during the push (fallback: root heading);
  * contact frame t_c = first frame where both hands are extended forward (>= `extend_frac` of their
    peak forward reach relative to the pelvis) and at similar height;
  * the box's near face is put at the hands' front at t_c (minus `preload`), centred laterally between
    the hands, made tall enough that the palms land below its top edge, wide enough for both palms,
    and deep enough that it slides instead of tipping (half depth >= tip_safety * friction * contact height).
The kinematic push distance is how far the hands advance past that face afterwards.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .sonic import JOINT_NAMES, Reference, set_robot_pose

HAND_GEOMS = ("left_hand_collision", "right_hand_collision")


def _geom_extreme(model, data, gid, d):
    """Farthest point of a collision geom along unit vector d (capsule / sphere / box)."""
    c = data.geom_xpos[gid]
    R = data.geom_xmat[gid].reshape(3, 3)
    t = model.geom_type[gid]
    s = model.geom_size[gid]
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        ax = R[:, 2]
        e = c + np.sign(ax @ d or 1.0) * ax * s[1]
        return e + d * s[0]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return c + d * s[0]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return c + R @ (np.sign(R.T @ d) * s[:3])
    return c


def robot_fk(model: mujoco.MjModel, ref: Reference):
    """Per-frame world positions of hand fronts (filled later per d), pelvis, and all collision geoms."""
    data = mujoco.MjData(model)
    gids = [i for i in range(model.ngeom) if model.geom_contype[i] == 2]
    out = {"pelvis": [], "geom_xpos": [], "geom_xmat": []}
    for i in range(ref.T):
        set_robot_pose(model, data, ref, i)
        out["pelvis"].append(data.xpos[model.body("pelvis").id].copy())
        out["geom_xpos"].append(data.geom_xpos[gids].copy())
        out["geom_xmat"].append(data.geom_xmat[gids].copy())
    return gids, {k: np.array(v) for k, v in out.items()}


def stage(model: mujoco.MjModel, ref: Reference, box_half: np.ndarray, extend_frac: float = 0.85,
          preload: float = 0.01, min_clear_top: float = 0.08, friction: float = 0.35,
          tip_safety: float = 1.4) -> dict:
    data = mujoco.MjData(model)
    hand_ids = [model.geom(n).id for n in HAND_GEOMS]
    pel = model.body("pelvis").id
    T = ref.T
    # heading at each frame
    yaw = 2 * np.arctan2(ref.root_quat[:, 3], ref.root_quat[:, 0])

    # pass 1: hand centres relative to pelvis in the heading frame
    hands = np.zeros((T, 2, 3))
    pelvis = np.zeros((T, 3))
    for i in range(T):
        set_robot_pose(model, data, ref, i)
        pelvis[i] = data.xpos[pel]
        hands[i] = data.geom_xpos[hand_ids]
    fwd = np.stack([np.cos(yaw), np.sin(yaw)], 1)
    rel = hands[:, :, :2] - pelvis[:, None, :2]
    reach = np.einsum("tkc,tc->tk", rel, fwd)            # forward reach of each hand
    both = reach.min(1)
    level = np.abs(hands[:, 0, 2] - hands[:, 1, 2]) < 0.15
    thr = extend_frac * both.max()
    cand = np.where((both >= thr) & level)[0]
    if len(cand) == 0:
        cand = np.array([int(both.argmax())])
    tc = int(cand[0])

    # push direction: root travel from contact to the end, else heading at contact
    travel = ref.root_pos[-1, :2] - ref.root_pos[tc, :2]
    d2 = travel / np.linalg.norm(travel) if np.linalg.norm(travel) > 0.15 else fwd[tc]
    d = np.array([d2[0], d2[1], 0.0])
    lat = np.array([-d[1], d[0], 0.0])

    # pass 2: hand front along d, every robot collision geom's extreme along d
    rob = [i for i in range(model.ngeom) if model.geom_contype[i] == 2]
    hand_front = np.zeros(T)
    other_front = np.zeros(T)
    for i in range(T):
        set_robot_pose(model, data, ref, i)
        hand_front[i] = max(_geom_extreme(model, data, g, d) @ d for g in hand_ids)
        other_front[i] = max(_geom_extreme(model, data, g, d) @ d for g in rob if g not in hand_ids
                             and model.geom_bodyid[g] not in _hand_chain(model))

    s = hand_front[tc] - preload                          # near-face offset along d
    face = np.maximum.accumulate(np.maximum(hand_front, s))  # kinematic box face (pushed by hands only)
    face[:tc] = s
    push_dist = float(face[-1] - s)
    # contact height and lateral centre at t_c
    zc = float(hands[tc, :, 2].mean())
    ylat = float(hands[tc, :, :2].mean(0) @ lat[:2])
    half = np.array(box_half, float)
    half[2] = max(half[2], (zc + min_clear_top) / 2)
    hand_span = abs(float((hands[tc, 0, :2] - hands[tc, 1, :2]) @ lat[:2]))
    half[1] = max(half[1], hand_span / 2 + 0.08)
    # slide, don't topple: a push at height zc tips the box once friction*zc > half depth
    half[0] = max(half[0], tip_safety * friction * zc)
    centre = d[:2] * (s + half[0]) + lat[:2] * ylat
    box_yaw = float(np.arctan2(d[1], d[0]))
    # other body parts must stay behind the moving face
    intrusion = float(np.max(other_front - face))
    return dict(t_contact=tc / 50.0, frame_contact=tc, push_dir=d[:2].tolist(), box_center=centre.tolist(),
                box_yaw=box_yaw, box_half=half.tolist(), contact_height=zc, kinematic_push_dist=push_dist,
                body_intrusion=intrusion, hand_front=hand_front.tolist(), face=face.tolist())


def _hand_chain(model):
    """Bodies of the forearms / wrists (allowed to touch the box)."""
    names = [n for n in (model.body(i).name for i in range(model.nbody))
             if any(k in n for k in ("wrist", "elbow", "hand", "rubber"))]
    return {model.body(n).id for n in names}

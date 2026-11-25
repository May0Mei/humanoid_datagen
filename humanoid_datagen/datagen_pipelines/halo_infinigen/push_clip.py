"""The HALO-baseline push clip: a Kimodo stand-in for HALO's motion-capture push recording.

HALO (arXiv 2611.06908, App. A3, Table IV) builds every Push episode as a generated walk ++ one recorded push of a
0.46 x 0.50 x 0.56 m box. The recording is not released, so we make one clip with Kimodo, with HALO's own Kimodo
helpers (ml-halo/datagen/motion_gen/kimodo, imported, not modified). The version in use (v4) starts bent at the box:
  1. author: the contact pose (crouch-lean, palms level on the box's near face), IK-solved from `hinge`;
  2. clip: Kimodo between two keyframes only, the contact pose at the box -> the same pose push_m further, with the
     text prompt (Kimodo chooses the steps; HALO composes its generated walk-up ++ this clip);
  3. render: the box is kinematic (HALO does not say how its box moves): it stays until the palms reach its near
     face, then its face follows the palms' furthest forward position; third-person and oak1 views side by side;
  4. track: the clip under GEAR-SONIC physics, as a check.
v1-v3 (deep half squat; hip hinge with pinned footsteps; upright walk-up + bend in place) did not survive GEAR-SONIC
tracking (docs/push_box.md, "HALO data-generation baseline"); their code and configs are in cache/push_oneoffs/
of the main repo (halo_push/push_clip_v1-v4.py, configs/push_box/halo_push_clip*.yaml).

Runs (configs/datagen/push/halo_infinigen/):
  clip.yaml        kimodo env: encode, author, clip      python humanoid_datagen/scripts/datagen.py --config_name datagen/push/halo_infinigen/clip
  clip_check.yaml  base env:   render, track (SONIC)     python humanoid_datagen/scripts/datagen.py --config_name datagen/push/halo_infinigen/clip_check
Output: <out_dir>/ (contact_pose.json, clip/clip_<k>.csv, clip/clip_<k>.mp4, track/). One-off: its output
(push_clip_contact/clip/clip_0.csv) is what the HALO-baseline episodes use (halo_infinigen/episodes.yaml).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from humanoid_datagen.utils.paths import REPO_ROOT as REPO, repo_path
from humanoid_datagen.datagen_pipelines.shared.kimodo import halo_path as _halo_path
from humanoid_datagen.datagen_pipelines.shared.base import Stage
FPS = 30


def rp(p: str) -> str:
    return repo_path(p)






# ------------------------------------------------------------------------------------------------ kimodo env
def encode(C):
    from humanoid_datagen.datagen_pipelines.shared.kimodo import encode as kimodo_encode
    kimodo_encode([C["contact_prompt"], C["clip_prompt"]], os.path.join(rp(C["out_dir"]), "text_emb.npz"), "cpu")


def _load_kimodo(C):
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.skeleton import G1Skeleton34
    from humanoid_datagen.datagen_pipelines.shared.kimodo import load
    model, _ = load(os.path.join(rp(C["out_dir"]), "text_emb.npz"), "cpu")
    return model, G1Skeleton34(), MujocoQposConverter(G1Skeleton34())


def clip(C, k):
    """Keyframed clip sample k (one per call): the contact pose at the box -> the contact pose push_m further; the text
    prompt and these two keyframes only (Kimodo chooses the steps)."""
    from kimodo.tools import seed_everything
    _halo_path()
    from motion_gen.kimodo.constraints import (build_constraint_object, canonicalize, compute_canon_angle,
                                               undo_canon_g1_qpos)
    from motion_gen.kimodo.segments import world_qpos_to_kf

    model, skel, conv = _load_kimodo(C)
    cp = json.load(open(os.path.join(rp(C["out_dir"]), "contact_pose.json")))
    kf_stand, kf_contact, kf_end = (np.asarray(cp[k], np.float64) for k in ("kf_stand", "kf_contact", "kf_end"))
    n_push = int(round(C["push_s"] * FPS))
    N = n_push + 1
    # keyframes in Kimodo's canonical frame of the standing pose (as motion_gen/kimodo/segments.py does)
    js, rs = zip(*[world_qpos_to_kf(k, conv) for k in (kf_stand, kf_contact, kf_end)])
    canon = compute_canon_angle(js[0], use_g1=True)
    jc, rc = zip(*[canonicalize(j, r, canon) for j, r in zip(js, rs)])
    origin = jc[0][0][[0, 2]].copy()
    jc = [j - np.array([origin[0], 0.0, origin[1]], np.float32) for j in jc]
    cons = [build_constraint_object([(0, jc[1], rc[1]), (n_push, jc[2], rc[2])], skel)]

    seed_everything(int(C["seed"]) + 1000 + k)
    res = model(prompts=[C["clip_prompt"]], num_frames=[N], constraint_lst=cons,
                num_denoising_steps=int(C["steps"]), cfg_weight=list(C["cfg"]), return_numpy=True,
                post_processing=bool(C.get("post_processing", False)))   # Kimodo's foot-skate correction
    q = np.asarray(conv.dict_to_qpos(res, device="cpu", root_quat_w_first=True, numpy=True)).reshape(N, -1)
    out = os.path.join(rp(C["out_dir"]), "clip")
    os.makedirs(out, exist_ok=True)
    w = undo_canon_g1_qpos(q.astype(np.float64), origin, canon)
    w[:, :2] += kf_contact[:2] - w[0, :2]        # Kimodo starts at its canonical origin; frame 0 = the contact keyframe
    np.savetxt(os.path.join(out, f"clip_{k}.csv"), w, delimiter=",", fmt="%.6f")
    json.dump(dict(frames=dict(arrive=0, contact=0, end=n_push, hands=[]), fps=FPS),
              open(os.path.join(out, "timing.json"), "w"), indent=1)
    print(f"saved clip {k} -> {out}", flush=True)



# ------------------------------------------------------------------------------------------------ mujoco side
def _yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def set_root(q, xy, yaw):
    """Copy of a [36] qpos with root xy and yaw set (pitch / roll kept), as HALO's sit_keyframes.set_root_pose."""
    from scipy.spatial.transform import Rotation as R
    p = np.asarray(q, np.float64).copy()
    dq = R.from_euler("z", yaw - _yaw(p[3:7]))
    r = dq * R.from_quat([p[4], p[5], p[6], p[3]])
    x = r.as_quat()
    p[3:7] = [x[3], x[0], x[1], x[2]]
    p[0], p[1] = xy
    return p


class Body:
    """FK of HALO's THOR G1: palms, knees, ankles, pelvis per qpos."""

    def __init__(self, mjcf):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        sid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.palms = [sid("left_palm"), sid("right_palm")]
        self.front = [bid(n) for n in ("left_knee_link", "right_knee_link", "left_ankle_roll_link",
                                       "right_ankle_roll_link")]
        self.feet = [bid("left_ankle_roll_link"), bid("right_ankle_roll_link")]

    def __call__(self, q):
        self.d.qpos[:] = q
        self.mj.mj_kinematics(self.m, self.d)
        return dict(palms=self.d.site_xpos[self.palms].copy(), front=self.d.xpos[self.front].copy(),
                    feet=self.d.xpos[self.feet].copy(), pelvis=self.d.qpos[:3].copy())


TOE = 0.13  # ankle -> toe tip, along the foot


def _save_contact(C, q, P, face, front, info):
    """contact_pose.json: standing / contact / push-end keyframes; q, P (palms), face, front in the robot frame."""
    D, W, H = C["box_size"]
    face = face + float(C.get("box_gap", 0.0))                     # box face this far beyond the palms
    y0 = -P[:, 1].mean()                                           # centre the palms on the box (y = 0)
    a = float(C["approach_m"])
    stand = np.load(rp(C["standing_pose"])).astype(np.float64)
    kf_stand = set_root(stand, (0.0, y0), 0.0)
    kf_contact = set_root(q, (a, y0), 0.0)
    kf_end = set_root(q, (a + float(C["push_m"]), y0), 0.0)
    rec = dict(**info, palms_robot_frame=P.round(4).tolist(), palm_height=float(P[:, 2].mean()),
               box_face_x=float(a + face), box_center=[float(a + face + D / 2), 0.0, float(H / 2)],
               knee_foot_front_robot_frame=float(front),
               kf_stand=kf_stand.tolist(), kf_contact=kf_contact.tolist(), kf_end=kf_end.tolist())
    json.dump(rec, open(os.path.join(rp(C["out_dir"]), "contact_pose.json"), "w"), indent=1)
    print(f"contact pose: {info}, palms at {P[:, 2].round(3)} m, "
          f"{face:.2f} m ahead of the pelvis; knees/toes {face - front:.2f} m behind the face", flush=True)


def author(C):
    """Hand-authored contact pose (as HALO's sit task commits authored sit keyframes): a hip hinge solved by least
    squares on HALO's THOR G1, symmetric left/right from the standing pose. Feet flat at the standing ankle height,
    palms level at `hinge.palm_z` and `hinge.reach` ahead of the toes, centre of mass 5 cm ahead of the ankles,
    knees near `hinge.knee`, pelvis pitch <= `hinge.max_pitch`."""
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R
    h = C["hinge"]
    body = Body(rp(C["robot_mjcf"]))
    m, d = body.m, body.d
    J = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]  # noqa: E731
    q0 = set_root(np.load(rp(C["standing_pose"])), (0.0, 0.0), 0.0)
    q0[3:7] = [1, 0, 0, 0]
    for n in ("hip_pitch", "knee", "ankle_pitch", "shoulder_pitch", "elbow", "wrist_pitch"):
        q0[J(f"right_{n}_joint")] = q0[J(f"left_{n}_joint")]
    for n in ("hip_roll", "hip_yaw", "ankle_roll", "shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_yaw"):
        q0[J(f"right_{n}_joint")] = -q0[J(f"left_{n}_joint")]
    q0[J("waist_yaw_joint")] = q0[J("waist_roll_joint")] = 0.0

    def fk(q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)

    fk(q0)
    ank = body.feet
    z_ank, foot_x0 = d.xpos[ank, 2].mean(), d.xmat[ank[0]].reshape(3, 3)[:, 0].copy()

    def build(x):
        th, hp, kn, ap, wp, sp, sr, el, wr = x
        q = q0.copy()
        q[3:7] = R.from_euler("y", th).as_quat(scalar_first=True)
        for side, sg in (("left", 1), ("right", -1)):
            for n, v in (("hip_pitch", hp), ("knee", kn), ("ankle_pitch", ap), ("shoulder_pitch", sp),
                         ("shoulder_roll", sg * sr), ("elbow", el), ("wrist_pitch", wr)):
                q[J(f"{side}_{n}_joint")] = v
        q[J("waist_pitch_joint")] = wp
        fk(q)
        q[2] += z_ank - d.xpos[ank, 2].mean()                      # feet back on the floor
        fk(q)
        return q

    def res(x):
        build(x)
        P, A = d.site_xpos[body.palms], d.xpos[ank]
        toe = (A[:, 0] + TOE).max()
        return np.r_[10 * (P[:, 2] - h["palm_z"]), 10 * (P[:, 0].min() - toe - h["reach"]),
                     5 * (P[:, 1] - [h["palm_y"], -h["palm_y"]]), 10 * (d.xmat[ank[0]].reshape(3, 3)[2, 0] - foot_x0[2]),
                     10 * (d.subtree_com[1][0] - (A[:, 0].mean() + 0.05)), 10 * (P[0, 0] - P[1, 0]),
                     x[2] - h["knee"], 0.5 * (x[7] - 0.5), 0.5 * x[8], 0.5 * (x[6] - 0.15)]

    lo = [-0.2, -2.5, 0.0, -0.8, -0.5, -3.0, -0.3, -0.5, -1.5]
    hi = [float(h["max_pitch"]), 0.5, 2.0, 0.5, 0.52, 1.0, 1.0, 2.0, 1.5]
    r = least_squares(res, np.clip([0.6, -0.8, 0.35, -0.1, 0.3, -1.2, 0.2, 0.5, 0.0], lo, hi), bounds=(lo, hi))
    q = build(r.x)
    b = body(q)
    P = b["palms"]
    face = P[:, 0].max()
    front = max(b["front"][:, 0].max(), (b["feet"][:, 0] + TOE).max())
    names = ("pelvis_pitch", "hip_pitch", "knee", "ankle_pitch", "waist_pitch", "shoulder_pitch", "shoulder_roll",
             "elbow", "wrist_pitch")
    _save_contact(C, q, P, face, front, dict(authored=dict(zip(names, np.round(r.x, 4).tolist())),
                                             pelvis_z=float(q[2]), solve_cost=float(r.cost)))


def box_track(Q, body, face0):
    """Kinematic box: the near face stays at face0 until the palms reach it, then follows their furthest reach."""
    lead = np.array([body(q)["palms"][:, 0].max() for q in Q])
    return np.maximum(face0, np.maximum.accumulate(lead)), lead


def metrics(Q, body, face, lead, tm, C):
    fc, fe = tm["frames"]["contact"], tm["frames"]["end"]
    fr = [body(q) for q in Q]
    ph = np.array([b["palms"][:, 2] for b in fr])
    feet = np.array([b["feet"] for b in fr])
    on = feet[:, :, 2] < feet[:, :, 2].min(axis=0) + 0.015      # planted: ankle within 1.5 cm of its lowest
    v = np.linalg.norm(np.diff(feet[:, :, :2], axis=0), axis=-1) * FPS
    skate = float(np.mean(v[on[1:]])) if on[1:].any() else 0.0
    gap = face[fc:fe + 1] - lead[fc:fe + 1]                       # > 0: palms behind the face (lost contact)
    return dict(box_push_m=float(face[-1] - face[0]),
                palm_height_push=[float(ph[fc:fe + 1].min()), float(ph[fc:fe + 1].max())],
                palm_gap_push_max=float(gap.max()), frac_push_in_contact=float(np.mean(gap < 0.03)),
                pelvis_min=float(Q[:, 2].min()), foot_skate_mps=skate,
                contact_frame_palm_gap=float(face[0] - lead[fc]))


def render(C, k):
    """Render clip k (one per call: the login node kills long CPU jobs); metrics merged into clip/metrics.json."""
    import glob
    import imageio.v2 as imageio
    import mujoco
    out = os.path.join(rp(C["out_dir"]), "clip")
    cp = json.load(open(os.path.join(rp(C["out_dir"]), "contact_pose.json")))
    tm = json.load(open(os.path.join(out, "timing.json")))
    body = Body(rp(C["robot_mjcf"]))
    D, W, H = C["box_size"]
    y0 = 0.0                                                       # pick centred the palms on y = 0
    spec = mujoco.MjSpec.from_file(rp(C["robot_mjcf"]))
    bx = spec.worldbody.add_body(name="pushbox", mocap=True, pos=[cp["box_center"][0], y0, H / 2])
    bx.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[D / 2, W / 2, H / 2], rgba=[0.75, 0.65, 0.45, 1],
                contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[1.5, -2.0, 4.0], dir=[0, 0.4, -1], diffuse=[0.7, 0.7, 0.7])
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 800
    m = spec.compile()
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 400, 640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 3.2, -18.0, 120.0
    mp = os.path.join(out, "metrics.json")
    allm = json.load(open(mp)) if os.path.exists(mp) else {}
    for f in [os.path.join(out, f"clip_{k}.csv")]:
        Q = np.loadtxt(f, delimiter=",")
        face, lead = box_track(Q, body, cp["box_face_x"])
        allm[os.path.basename(f)] = mm = metrics(Q, body, face, lead, tm, C)
        with imageio.get_writer(f[:-4] + ".mp4", fps=FPS, codec="libx264", quality=8, macro_block_size=8) as wr:
            for t, q in enumerate(Q):
                d.qpos[:] = q
                d.mocap_pos[0] = [face[t] + D / 2, y0, H / 2]
                mujoco.mj_forward(m, d)
                cam.lookat[:] = [0.5 * (q[0] + face[t]), y0, 0.5]
                r.update_scene(d, camera=cam)
                a = r.render()
                r.update_scene(d, camera="oak1")
                wr.append_data(np.concatenate([a, r.render()], axis=1))
        print(f"{os.path.basename(f)}: {json.dumps(mm)}", flush=True)
    r.close()
    json.dump(allm, open(mp, "w"), indent=1)


def track(C, k, tk):
    """GEAR-SONIC (our shared frozen tracker, humanoid_datagen/utils/sonic.py) on each clip, with the push box as a
    free 1 kg physics body on an open floor; same rollout and metrics as the push datagen (datagen_pipelines/shared/run.py)."""
    import glob
    import warnings
    import mujoco
    from humanoid_datagen.datagen_pipelines.shared.compose import ROBOT_CONAFF, ROBOT_CONTYPE, _robot_spec
    from humanoid_datagen.datagen_pipelines.shared.run import render as run_render, rollout, save
    from humanoid_datagen.utils.sonic import Reference
    cp = json.load(open(os.path.join(rp(C["out_dir"]), "contact_pose.json")))
    D, W, H = C["box_size"]
    sp = mujoco.MjSpec()
    sp.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=[0.55, 0.55, 0.58], rgb2=[0.42, 0.42, 0.45], width=512, height=512)
    sp.add_material(name="grid", textures=["", "grid"], texrepeat=[100, 100])
    sp.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1], material="grid")
    sp.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                           diffuse=[0.8, 0.8, 0.8])
    sp.visual.headlight.ambient = [0.4, 0.4, 0.4]
    sp.visual.global_.offwidth, sp.visual.global_.offheight = 960, 540
    sp.option.timestep = 0.005
    cx = float(cp["box_center"][0])
    b = sp.worldbody.add_body(name="box", pos=[cx, 0.0, H / 2 + 0.002])
    b.add_freejoint()
    b.add_geom(name="box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[D / 2, W / 2, H / 2], mass=float(C["box_mass"]),
               friction=[float(C["box_friction"]), 0.005, 0.0001], rgba=[0.75, 0.65, 0.45, 1])
    if "box_floor_friction" in C:
        # MuJoCo combines geom frictions by max (floor default 1.0); pin the box-floor coefficient explicitly
        mu = float(C["box_floor_friction"])
        sp.add_pair(geomname1="box", geomname2="floor", friction=[mu, mu, 0.005, 0.0001, 0.0001])
    sp.worldbody.add_camera(name="side", pos=[cx - 0.2, -3.2, 1.6], xyaxes=[1, 0, 0, 0, 0.325, 0.946])
    rs = _robot_spec()
    for g in rs.geoms:
        if g.contype or g.conaffinity:
            g.contype, g.conaffinity = ROBOT_CONTYPE, ROBOT_CONAFF
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp.worldbody.add_frame().attach_body(rs.body("pelvis"), "", "")
        model = sp.compile()
    out = os.path.join(rp(C["out_dir"]), "track")
    mp = os.path.join(out, "metrics.json")
    allm = json.load(open(mp)) if os.path.exists(mp) else {}
    for f in [os.path.join(rp(C["out_dir"]), "clip", f"clip_{k}.csv")]:
        name = os.path.basename(f)[:-4]
        ref = Reference.from_qpos(np.loadtxt(f, delimiter=","), FPS)
        res = {}
        for mode in ("track", "kinematic"):
            tr = rollout(model, ref, mode, settle_s=0.5, tail_s=1.0, object_body="box")
            mm = tk.metrics(tr, ref, [1.0, 0.0], model)
            mm["accepted"] = bool((not mm["robot_fell"]) and mm["push_dist"] >= 0.3 and mm["max_tilt_deg"] <= 20)
            save(tr, out, f"{name}_{mode}", mm)
            if mode == "track":
                run_render(model, tr, os.path.join(out, f"{name}_sonic.mp4"), "side", size=(960, 540), stride=2)
            res[mode] = mm
        allm[name] = res
        t, k = res["track"], res["kinematic"]
        print(f"{name}: SONIC push {t['push_dist']:.2f} m (kinematic {k['push_dist']:.2f}), fell={t['robot_fell']}, "
              f"tilt {t['max_tilt_deg']:.0f} deg, joint err {t['joint_err_mean']:.3f} rad, accepted={t['accepted']}",
              flush=True)
    json.dump(allm, open(mp, "w"), indent=1)


def _shard(C: dict, n: int) -> list[int]:
    """Item indices of this process: range(n)[shard::num_shards] (or SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT)."""
    sh = int(os.environ.get("SLURM_ARRAY_TASK_ID", C.get("shard", 0)))
    ns = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", C.get("num_shards", 1)))
    return list(range(n))[sh::ns]


class HaloClip(Stage):
    """Datagen stage `clip` of the HALO-Infinigen pipeline: C["run_steps"] in order (encode, author: once; clip, render,
    track: for each sample index of this shard, of clip_samples)."""

    def run(self) -> None:
        C, tk = self.cfg, self.task
        os.makedirs(rp(C["out_dir"]), exist_ok=True)
        for step in C["run_steps"]:
            if step in ("encode", "author"):
                dict(encode=encode, author=author)[step](C)
            else:
                for k in _shard(C, int(C["clip_samples"])):
                    if step == "track":
                        track(C, k, tk)
                    else:
                        dict(clip=clip, render=render)[step](C, k)

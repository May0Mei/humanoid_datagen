"""HALO-baseline push episodes: HALO's Push recipe ("walk (best of 10) ++ recorded push", 9-11-frame bridge) with
the v4 Kimodo clip in place of HALO's unreleased recording (docs/push_box.md, "HALO data-generation baseline").

HALO's own code is imported, not modified (ml-halo/datagen):
  gen     scene-free frame like tasks/sit.py: the pushbox at the origin, push along +x; the robot starts spawn_dist
          (pelvis -> box centre) behind it, facing it. Walk: unicycle plan to the clip's first pelvis position
          (motion_gen/unicycle.py), Kimodo walk_qpos_from_traj, best of walk_batch ranked by HALO's walk gate
          (check_worker.check_one, prefix key, as generate_sit_traj ranks its walks; HALO ranks Push walks by gait
          similarity to its recording, which we do not have). Bridge: gen_segment (full-body keyframes walk end ->
          clip frame 0). Joined with seam_blend. The box is kinematic: it stays until the palms reach its near face,
          then the face follows the palms' furthest reach (no rotation). The whole episode is run through
          motion_check (task push_box: HALO's default + tier thresholds) and the verdict recorded.
          With `approach_prompt` in the config, walk + bridge are replaced by ONE Kimodo call (approach()): full-body
          keyframes HALO standing pose -> clip frame 0, root path = the unicycle plan, bend_frames free frames for the
          stand -> bend; best of walk_batch by the whole-episode gate. HALO's 9-11-frame bridge assumes a recording
          that starts with a walk; our clip starts bent, so the transition needs ~1 s.
  render  one episode into one Infinigen room with HALO's MuJoCo backend: build_scene_with_robot, a placement search
          like se2_fit (robot path clear by 0.35 m, plus the box's path clear by its footprint radius), place
          _trajectory, select_infinigen_tpv, OAK pinhole -> calibrated fisheye (fisheye_warp). The box is a mocap
          body. Clean render (no cam_fx). Writes tpv.mp4, fpv.mp4 (fisheye OAK) and side_by_side.mp4.

Runs (configs/datagen/push/halo_infinigen/):
  episodes.yaml         kimodo env, GPU: encode, gen    python humanoid_datagen/scripts/datagen.py --config_name datagen/push/halo_infinigen/episodes
  episodes_render.yaml  base env, EGL:   render         python humanoid_datagen/scripts/datagen.py --config_name datagen/push/halo_infinigen/episodes_render
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

from humanoid_datagen.utils.paths import REPO_ROOT as REPO, repo_path, third_party
from humanoid_datagen.datagen_pipelines.shared.kimodo import halo_path as _halo_path  # noqa: E402
from humanoid_datagen.datagen_pipelines.shared.base import Stage
HALO = third_party("ml-halo", "datagen")
FPS = 30
ROBOT_RADIUS = 0.35   # HALO solve_path_placement default


def rp(p: str) -> str:
    return repo_path(p)






# ------------------------------------------------------------------------------------------------ clip geometry
def clip_in_box_frame(C):
    """The clip with the box centre at the origin and the push along +x; box near-face x at frame 0."""
    from humanoid_datagen.datagen_pipelines.halo_infinigen.push_clip import Body
    Q = np.loadtxt(rp(C["clip_csv"]), delimiter=",")
    Q[:, 2] += float(C.get("clip_lift", 0.0))         # clears HALO's foot_sink gate (v4 clip_0 sinks 4.3 cm)
    body = Body(rp(C["robot_mjcf"]))
    P = body(Q[0])["palms"]
    D = float(C["box_size"][0])
    face = float(P[:, 0].max())                       # box face at the palms of frame 0 (the clip pushes along +x)
    centre = np.array([face + D / 2, float(P[:, 1].mean())])
    Q = Q.copy()
    Q[:, :2] -= centre
    return Q, -D / 2


def box_track(Q, body, face0, D):
    """Kinematic box centre x per frame: the near face stays at face0 until the palms reach it, then follows the
    palms' furthest forward reach (push_clip.box_track); robot frame has the push along +x."""
    lead = np.array([body(q)["palms"][:, 0].max() for q in Q])
    face = np.maximum(face0, np.maximum.accumulate(lead))
    return face + D / 2


# ------------------------------------------------------------------------------------------------ kimodo env
def encode(C):
    from humanoid_datagen.datagen_pipelines.shared.kimodo import encode as kimodo_encode
    kimodo_encode([C[k] for k in ("walk_prompt", "bridge_prompt", "approach_prompt") if C.get(k)],
                  os.path.join(rp(C["out_dir"]), "text_emb.npz"), "cpu")


def _standing_world(spawn):
    """HALO's standing keyframe qpos (single_kf_standpose.npz) with the pelvis at spawn, facing +x (yaw 0)."""
    from scipy.spatial.transform import Rotation as R
    _halo_path()
    q = np.load(os.path.join(HALO, "motion_gen", "kimodo", "keyframes", "single_kf_standpose.npz"))["qpos"]
    q = np.asarray(q, np.float64).copy()
    w, x, y, z = q[3:7]
    e = R.from_quat([x, y, z, w]).as_euler("ZYX")
    nq = R.from_euler("ZYX", [0.0, e[1], e[2]]).as_quat()
    q[3:7] = [nq[3], nq[0], nq[1], nq[2]]
    q[:2] = spawn
    return q


def gen(C, k, device):
    import torch
    from kimodo import load_model
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.skeleton import G1Skeleton34
    from humanoid_datagen.datagen_pipelines.halo_infinigen.push_clip import Body
    from humanoid_datagen.datagen_pipelines.shared.kimodo import CachedText, approach
    _halo_path()
    from common import placement as _pl
    from motion_gen import unicycle as _uni
    from motion_gen.check_worker import check_one
    from motion_gen.kimodo.motion_gen import seam_blend
    from motion_gen.kimodo.segments import gen_segment
    from motion_gen.kimodo.walk import _g1_standing_keyframe, walk_qpos_from_traj
    from motion_gen.motion_check.config import CheckConfig
    from motion_gen.motion_check.episode import MotionEpisode
    from motion_gen.motion_check.run import run_episode
    from motion_gen.tasks.sit import canonical_scene
    from kimodo.tools import seed_everything

    torch.set_num_threads(6)
    out = os.path.join(rp(C["out_dir"]), "episodes", f"episode_{k:04d}")
    os.makedirs(out, exist_ok=True)
    seed = int(C["seed"]) * 1000 + k
    model = load_model("Kimodo-G1-RP-v1", device=device,
                       text_encoder=CachedText(os.path.join(rp(C["out_dir"]), "text_emb.npz")))
    skel = G1Skeleton34()
    conv = MujocoQposConverter(G1Skeleton34())
    standing_kf = _g1_standing_keyframe(skel, model=model, cache_path=rp(C["standing_kf"]))

    clip, face0 = clip_in_box_frame(C)
    D, W, H = C["box_size"]
    stop = clip[0, :2].copy()                                   # walk ends at the clip's first pelvis position
    spawn = np.array([-float(C["spawn_dist"]), 0.0])            # HALO: pelvis 1.3 m behind the box, facing it
    scene = canonical_scene()
    occ = _pl.OccGrid(grid=scene.occ.grid.copy(), x_min=scene.occ.x_min, y_min=scene.occ.y_min,
                      resolution=scene.occ.resolution)
    _pl.add_box(occ, (0.0, 0.0), (D / 2, W / 2), 0.0)
    dil = _pl.dilate(occ, 0.30)
    path = _pl.plan_path_astar(dil, tuple(spawn), tuple(stop))
    corridor = _pl.simplify_path(path, n_waypoints=20) if path is not None and len(path) >= 2 else [tuple(spawn), tuple(stop)]
    plan = _uni.simulate_corridor(tuple(spawn), 0.0, corridor, occ=scene.occ, robot_radius=0.30,
                                  v_max=float(C["walk_speed"]))
    print(f"[ep{k}] walk plan {len(plan)} frames, {np.linalg.norm(plan[-1, :2] - plan[0, :2]):.2f} m", flush=True)

    body = Body(rp(C["robot_mjcf"]))
    cfg = CheckConfig.for_task("push_box", tier=C["tier"])

    def episode_check(q, phases, name):
        rep = run_episode(MotionEpisode(episode_id=name, qpos=q, task_type="push_box",
                                        meta={"task_type": "push_box", "phases": phases}), cfg, use_mujoco=True)
        return dict(accept=bool(rep.verdict.accept), blocking=list(rep.verdict.blocking),
                    score=float(rep.verdict.score))

    if C.get("approach_prompt"):
        # approach: one Kimodo call standing -> walk -> bend into clip frame 0; best of walk_batch by the whole-episode
        # push_box gate (fewest blocking checks, then score)
        stand = _standing_world(spawn)
        apps, T = approach(C, model, skel, conv, plan, stand, clip[0], seed, device)
        cands = []
        for i, a in enumerate(apps):
            cat = np.concatenate([a[:-1], clip], axis=0)          # the last approach frame is the clip's frame 0
            q = seam_blend(cat.copy(), [len(a) - 1], k=int(C["seam_k"]))
            ph = {"WALK": [0, T], "BRIDGE": [T, len(a) - 1], "PUSH": [len(a) - 1, len(q)]}
            v = episode_check(q, ph, f"ep{k}_s{i}")
            cands.append((q, ph, v))
            print(f"[ep{k}] sample {i}: accept={v['accept']} blocking={v['blocking']} score={v['score']:.3f}",
                  flush=True)
        wi = min(range(len(cands)), key=lambda i: (len(cands[i][2]["blocking"]), -cands[i][2]["score"]))
        qpos, phases, verdict = cands[wi]
        seams = [phases["PUSH"][0]]
        bx = np.zeros(len(qpos))
        bx[phases["PUSH"][0]:] = box_track(qpos[phases["PUSH"][0]:], body, face0, D)
        box = np.stack([bx, np.zeros(len(qpos)), np.full(len(qpos), H / 2)], axis=1)
        np.save(os.path.join(out, "g1_qpos.npy"), qpos.astype(np.float32))
        np.save(os.path.join(out, "box_pos.npy"), box.astype(np.float32))
        np.save(os.path.join(out, "plan.npy"), plan.astype(np.float32))
        meta = dict(task_type="push_box", mode="approach", phases=phases, seams=seams, total_frames=len(qpos),
                    pick=int(wi), batch=len(cands), n_accept=int(sum(c[2]["accept"] for c in cands)),
                    gate=verdict, box_push_m=float(box[-1, 0] - box[0, 0]), spawn=spawn.tolist(), seed=seed,
                    clip=C["clip_csv"], instruction="Push the cardboard box forward.")
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"[ep{k}] {len(qpos)} frames ({phases}); pick {wi} ({meta['n_accept']}/{len(cands)} accepted); "
              f"gate accept={verdict['accept']} blocking={verdict['blocking']}; box pushed {meta['box_push_m']:.2f} m",
              flush=True)
        return

    # walk: best of walk_batch by HALO's walk gate (generate_sit_traj ranks its walks the same way)
    seed_everything(seed)
    walks = walk_qpos_from_traj(plan[:, 0], plan[:, 1], plan[:, 2], 0.0, model, skel, standing_kf, end_yaw=0.0,
                                num_denoising_steps=int(C["steps"]), cfg_weight=float(C["cfg"][0]),
                                prompt=C["walk_prompt"], device=device, num_samples=int(C["walk_batch"]))
    walks = walks[None] if walks.ndim == 2 else walks
    reps = [check_one((i, walks[i], {"WALK": [0, walks.shape[1]]}, C["tier"])) for i in range(len(walks))]
    END = ("sits_at_end", "seat_contact")

    def key(r):
        b = [x for x in r["blocking"] if x not in END]
        sk = r["checks"].get("foot_skate", {}).get("meta", {}).get("frac_planted_skating")
        return (len(b), float("inf") if sk is None else float(sk), -float(r["score"]))
    wi = min(range(len(reps)), key=lambda i: key(reps[i]))
    walk = np.asarray(walks[wi], np.float64)

    # bridge: full-body keyframes walk end -> clip frame 0
    bseed = seed + 500
    seed_everything(bseed)
    bridge = gen_segment(model, skel, conv, C["bridge_prompt"], int(C["bridge_frames"]), walk[-1], clip[0],
                         bseed, device, int(C["steps"]), cfg=tuple(C["cfg"]), turn_mode="fullbody")
    bridge = np.asarray(bridge, np.float64)
    cat = np.concatenate([walk, bridge[1:-1], clip], axis=0)    # bridge endpoints are the walk end / clip start
    seams = [len(walk), len(walk) + len(bridge) - 2]
    qpos = seam_blend(cat.copy(), seams, k=int(C["seam_k"]))
    phases = {"WALK": [0, seams[0]], "BRIDGE": [seams[0], seams[1]], "PUSH": [seams[1], len(qpos)]}

    # kinematic box (frames before the push: the box at rest)
    bx = np.zeros(len(qpos))                                    # box centre at the origin until the push
    bx[phases["PUSH"][0]:] = box_track(qpos[phases["PUSH"][0]:], body, face0, D)
    box = np.stack([bx, np.zeros(len(qpos)), np.full(len(qpos), H / 2)], axis=1)

    verdict = episode_check(qpos, phases, f"ep{k}")
    np.save(os.path.join(out, "g1_qpos.npy"), qpos.astype(np.float32))
    np.save(os.path.join(out, "box_pos.npy"), box.astype(np.float32))
    np.save(os.path.join(out, "plan.npy"), plan.astype(np.float32))
    meta = dict(task_type="push_box", phases=phases, seams=seams, total_frames=len(qpos), walk_pick=int(wi),
                walk_batch=int(C["walk_batch"]), walk_pick_blocking=[x for x in reps[wi]["blocking"] if x not in END],
                walk_n_pass=int(sum(1 for r in reps if not [x for x in r["blocking"] if x not in END])),
                gate=verdict, box_push_m=float(box[-1, 0] - box[0, 0]), spawn=spawn.tolist(), seed=seed,
                clip=C["clip_csv"], instruction="Push the cardboard box forward.")
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"[ep{k}] {len(qpos)} frames ({phases}); walk pick {wi} ({meta['walk_n_pass']}/{len(walks)} pass); "
          f"gate accept={verdict['accept']} blocking={verdict['blocking']}; box pushed {meta['box_push_m']:.2f} m",
          flush=True)


# ------------------------------------------------------------------------------------------------ render
def _place(C, qpos, box, analysis, clearance, cell, seed):
    """se2_fit-style search (HALO solve_path_placement) with the box's path also clear; returns (spawn_xy, yaw)."""
    _halo_path()
    from render.synthetic_scene.robot_placement import _heading0
    D, W, _ = C["box_size"]
    box_r = 0.5 * math.hypot(D, W) + 0.10                      # HALO target_footprint_radius rule
    heading0 = _heading0(qpos)
    origin = qpos[0, :2].copy()
    rel_path = qpos[::3, :2] - origin
    rel_box = box[::10, :2] - origin
    ny, nx = clearance.shape
    rng = np.random.default_rng(seed)
    cand = np.argwhere(clearance >= ROBOT_RADIUS)
    order = rng.permutation(len(cand))[:4000]
    yaws = rng.permutation(24) * (2.0 * math.pi / 24)

    def ok(pts, r, sx, sy, Rz):
        w = pts @ Rz.T + np.array([sx, sy])
        i, j = np.floor(w[:, 0] / cell).astype(int), np.floor(w[:, 1] / cell).astype(int)
        return i.min() >= 0 and i.max() < nx and j.min() >= 0 and j.max() < ny and clearance[j, i].min() >= r

    for ci in order:
        j, i = cand[ci]
        sx, sy = (i + 0.5) * cell, (j + 0.5) * cell
        for yaw in yaws:
            dyaw = yaw - heading0
            Rz = np.array([[math.cos(dyaw), -math.sin(dyaw)], [math.sin(dyaw), math.cos(dyaw)]])
            if ok(rel_path, ROBOT_RADIUS, sx, sy, Rz) and ok(rel_box, box_r, sx, sy, Rz):
                return [float(sx), float(sy)], float(yaw)
    return None


def _ffmpeg_with_x264(root):
    """Put imageio's bundled ffmpeg (has libx264; the system one does not) first on PATH as `ffmpeg`."""
    import imageio_ffmpeg
    d = os.path.join(root, "_bin")
    os.makedirs(d, exist_ok=True)
    link = os.path.join(d, "ffmpeg")
    if not os.path.exists(link):
        os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), link)
    if not os.environ.get("PATH", "").startswith(d + os.pathsep):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _robot_base_mjcf(C, work):
    """The robot MJCF handed to HALO's make_robot_with_cameras: `robot_render_mjcf` (default HALO's THOR G1 with
    backpack) with, if `oak_calib` is set, the oak1 camera pose replaced by the calibration's helmet-frame pose."""
    import re
    if not C.get("robot_render_mjcf"):
        return None
    xml = open(rp(C["robot_render_mjcf"])).read()
    if C.get("oak_calib"):
        cal = json.load(open(rp(C["oak_calib"])))
        # calib_run/result.json (one hand-eye run) or assets/robots/g1/oak1_calibration.json (the adopted calibration)
        cam = cal["mujoco_camera_in_helmet"] if "mujoco_camera_in_helmet" in cal else \
            cal["extrinsics"]["mujoco_camera_in_helmet_assembly_link"]
        xml, n = re.subn(r'(<camera\s+name="oak1"\s+)pos="[^"]*"\s+xyaxes="[^"]*"',
                         rf'\1pos="{cam["pos"]}" xyaxes="{cam["xyaxes"]}"', xml)
        assert n == 1, "oak1 camera not found in robot_render_mjcf"
    os.makedirs(work, exist_ok=True)
    out = os.path.join(work, "robot_base.xml")
    open(out, "w").write(xml)
    return out


def _scene_with_robot(C, scene_dir, work, offwidth, offheight):
    """HALO build_scene_with_robot (same room parts, headlight and include), with our robot base MJCF if set."""
    from render.synthetic_scene import build_infinigen_mjcf as B
    base = _robot_base_mjcf(C, work)
    if base is None:
        return B.build_scene_with_robot(scene_dir, work, offwidth=offwidth, offheight=offheight)
    parts = B.split_obj(os.path.join(scene_dir, "asset", "scan.obj"), os.path.join(work, "_mjparts"))
    assets, geoms = B.geometry_assets_and_geoms(parts)
    robot = B.make_robot_with_cameras(os.path.join(work, "g1_29dof_ego.xml"), base_mjcf=base,
                                      meshes_dir=rp(C["robot_meshes"]))
    out = os.path.join(work, "scene_with_robot.xml")
    open(out, "w").write(f"""<mujoco model="infinigen_scene_with_robot">
  <compiler meshdir="/" texturedir="/" angle="radian" autolimits="true"/>
  <include file="{os.path.basename(robot)}"/>
  <visual>
    <headlight ambient="1 1 1" diffuse="0 0 0" specular="0 0 0"/>
    <global offwidth="{offwidth}" offheight="{offheight}"/>
    <quality offsamples="4"/>
    <map znear="0.005" zfar="5.0"/>
  </visual>
  <asset>
{assets}
  </asset>
  <worldbody>
{geoms}
  </worldbody>
</mujoco>
""")
    return out


def scene_of(C, k):
    n = C.get("episodes_per_scene")
    return C["scenes"][k // int(n)] if n else C["scenes"][k % len(C["scenes"])]


def render(C, k):
    import imageio.v2 as imageio
    import mujoco
    _halo_path()
    from render.cameras import load_oak_intrinsics
    from render.driver_common import load_scene
    from render.fisheye_warp import warp_pinhole_mp4_to_fisheye_mp4
    from render.mujoco.driver import _add_backing_floor
    from render.synthetic_scene.robot_placement import place_trajectory, rigid_place_xy, _heading0
    from render.tpv import select_infinigen_tpv

    intr = load_oak_intrinsics(rp(C["oak_intrinsics"]) if C.get("oak_intrinsics") else None)
    ep = os.path.join(rp(C["out_dir"]), "episodes", f"episode_{k:04d}")
    scene_name = scene_of(C, k)
    scene_dir = os.path.join(rp(C["scene_root"]), scene_name)
    qpos = np.load(os.path.join(ep, "g1_qpos.npy")).astype(np.float64)
    box = np.load(os.path.join(ep, "box_pos.npy")).astype(np.float64)
    scene = load_scene(scene_dir)
    floor_z = scene["floor_z"]
    sol = _place(C, qpos, box, scene["analysis"], scene["clearance"], scene["cell"], seed=1000 + k)
    if sol is None:
        raise SystemExit(f"no placement for episode {k} in {scene_name}")
    spawn_xy, spawn_yaw = sol
    dyaw = spawn_yaw - _heading0(qpos)
    box_w = rigid_place_xy(box[:, :2], qpos, spawn_xy, spawn_yaw)
    qw = place_trajectory(qpos, spawn_xy + [floor_z], spawn_yaw, floor_z)
    eye, target = select_infinigen_tpv(scene["analysis"], np.concatenate([qw[:, :2], box_w]), floor_z)

    D, W, H = C["box_size"]
    work = os.path.join(rp(C["out_dir"]), "_mjscene", scene_name)
    pr = intr["pinhole_render"]
    oak_w, oak_h = pr["width"], pr["height"]
    xml = open(_scene_with_robot(C, scene_dir, work, offwidth=max(1600, C["width"], oak_w),
                                 offheight=max(1000, C["height"], oak_h))).read()
    r, g, b, a = C["box_rgba"]
    if C.get("box_texture"):        # cardboard + packing tape cube texture (datagen_pipelines/halo_infinigen/box_texture.py)
        from humanoid_datagen.datagen_pipelines.halo_infinigen.box_texture import mjcf_texture, write
        tex = rp(C["box_texture"])
        if not os.path.exists(os.path.join(tex, "top.png")):
            write(tex)
        mat = (f'    <texture name="pushbox_tex" type="cube" {mjcf_texture(tex)}/>\n'
               f'    <material name="pushbox_mat" texture="pushbox_tex"/>\n  </asset>')
    else:
        mat = f'    <material name="pushbox_mat" rgba="{r} {g} {b} {a}"/>\n  </asset>'
    xml = xml.replace("</asset>", mat, 1)
    xml = xml.replace("</worldbody>", f'    <body name="pushbox" mocap="true" pos="0 0 0"><geom type="box" '
                      f'size="{D / 2} {W / 2} {H / 2}" material="pushbox_mat" contype="0" conaffinity="0"/></body>\n'
                      "  </worldbody>", 1)
    xml = _add_backing_floor(xml, floor_z)
    xp = os.path.join(work, f"episode_{k:04d}.xml")
    open(xp, "w").write(xml)
    model = mujoco.MjModel.from_xml_path(xp)
    model.vis.global_.fovy = 61.3                               # HALO: Isaac's TPV vertical FoV
    data = mujoco.MjData(model)
    ren = mujoco.Renderer(model, height=C["height"], width=C["width"])
    oak = mujoco.Renderer(model, height=oak_h, width=oak_w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    fwd = np.array(target) - np.array(eye)
    cam.distance = float(np.linalg.norm(fwd))
    fwd /= cam.distance
    cam.lookat[:] = target
    cam.elevation = float(np.degrees(np.arcsin(fwd[2])))
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    ego = mujoco.MjvOption()
    ego.geomgroup[1] = 0                                        # HALO: hide the robot's own meshes in head cameras
    bq = np.array([math.cos(dyaw / 2), 0, 0, math.sin(dyaw / 2)])
    tpv, oakf = [], []
    for t in range(len(qw)):
        data.qpos[:36] = qw[t]
        data.mocap_pos[0] = [box_w[t, 0], box_w[t, 1], floor_z + H / 2]
        data.mocap_quat[0] = bq
        mujoco.mj_forward(model, data)
        ren.update_scene(data, camera=cam)
        tpv.append(ren.render().copy())
        oak.update_scene(data, camera="oak1", scene_option=ego)
        oakf.append(oak.render().copy())
    imageio.mimsave(os.path.join(ep, "tpv.mp4"), tpv, fps=FPS, macro_block_size=1)
    imageio.mimsave(os.path.join(ep, "oak_pinhole.mp4"), oakf, fps=FPS, macro_block_size=1)
    _ffmpeg_with_x264(rp(C["out_dir"]))                         # fisheye_warp pipes to `ffmpeg ... libx264`
    warp_pinhole_mp4_to_fisheye_mp4(os.path.join(ep, "oak_pinhole.mp4"), os.path.join(ep, "fpv.mp4"),
                                    intr, fps=FPS)
    fpv = imageio.mimread(os.path.join(ep, "fpv.mp4"), memtest=False)
    from PIL import Image
    hh = C["height"]
    side = []
    for a_, b_ in zip(tpv, fpv):
        b_ = np.asarray(Image.fromarray(np.asarray(b_)[..., :3]).resize(
            (int(round(b_.shape[1] * hh / b_.shape[0])) // 2 * 2, hh)))
        side.append(np.concatenate([a_, b_], axis=1))
    imageio.mimsave(os.path.join(ep, "side_by_side.mp4"), side, fps=FPS, macro_block_size=1)
    meta = json.load(open(os.path.join(ep, "meta.json")))
    meta["render"] = dict(scene=scene_name, spawn_xy=spawn_xy, spawn_yaw=spawn_yaw, tpv_eye=eye, tpv_target=target,
                          floor_z=floor_z)
    json.dump(meta, open(os.path.join(ep, "meta.json"), "w"), indent=1)
    print(f"[ep{k}] rendered {len(qw)} frames in {scene_name} -> {ep}/{{tpv,fpv,side_by_side}}.mp4", flush=True)


def _shard(C: dict, n: int) -> list[int]:
    """Item indices of this process: range(n)[shard::num_shards] (or SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT)."""
    sh = int(os.environ.get("SLURM_ARRAY_TASK_ID", C.get("shard", 0)))
    ns = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", C.get("num_shards", 1)))
    return list(range(n))[sh::ns]


class HaloEpisodes(Stage):
    """Datagen stage `episodes` of the HALO-Infinigen pipeline: C["run_steps"] in order (encode: once; gen, render: for
    each episode index of this shard, of `episodes`)."""

    def run(self) -> None:
        C, tk = self.cfg, self.task
        os.makedirs(rp(C["out_dir"]), exist_ok=True)
        for step in C["run_steps"]:
            if step == "encode":
                encode(C)
                continue
            for k in _shard(C, int(C["episodes"])):
                if step == "gen":
                    gen(C, k, C["device"])
                else:
                    render(C, k)

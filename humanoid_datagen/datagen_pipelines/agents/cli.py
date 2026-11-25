"""push-scene agent tools. Run from the run dir:  python -m humanoid_datagen.datagen_pipelines.agents.cli <tool> [args]

Every tool prints one JSON object. Paths are relative to the run dir (cwd).
  keyframes  --video prior/video.mp4 [--n 12]        -> prior_frames/sheet.png + prior_frames/fNN.png
  motions                                            -> motions/summary.json + motions/<m>_sheet.png
  compose    [--task task.json] [--room room.xml]    staging + static checks (fast; no physics)
  render     [--task ..] [--room ..] [--frames start,contact,end] [--cameras a,b]  -> renders/rNN/sheet.png
  simulate   [--task ..] [--room ..]                 GEAR-SONIC + kinematic rollouts -> sims/sNN/ (metrics, videos, sheets)
  finalize   [--task ..] [--room ..] --assessment assessment.json   -> final/, result.json
  room_from_layout --layout layout.json [--out room.xml]   optional quick room from a JSON layout (primitives)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from humanoid_datagen.tasks.load import task_from_env   # the session's task ($HDG_TASK, set by agents/batch.py)



def _ffmpeg() -> str:
    """imageio's bundled ffmpeg: the system one on della decodes h264 only through libopenh264, which fails."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"

def _out(d):
    print(json.dumps(d, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def _next_dir(root, prefix):
    os.makedirs(root, exist_ok=True)
    k = 1
    while os.path.exists(os.path.join(root, f"{prefix}{k:02d}")):
        k += 1
    p = os.path.join(root, f"{prefix}{k:02d}")
    os.makedirs(p)
    return p


def _log(tool, args, res):
    with open("tool_log.jsonl", "a") as f:
        f.write(json.dumps({"tool": tool, "args": vars(args), "ok": "error" not in res}, default=str) + "\n")


def _sheet(images, path, cols=4, labels=None):
    from PIL import Image, ImageDraw
    w, h = images[0].shape[1], images[0].shape[0]
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(images):
        t = Image.fromarray(im)
        if labels:
            ImageDraw.Draw(t).text((6, 4), labels[i], fill=(255, 255, 0))
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    sheet.save(path)
    return path


# ------------------------------------------------------------------------------------------------ keyframes
def keyframes(a):
    os.makedirs("prior_frames", exist_ok=True)
    n = a.n
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.video],
                               capture_output=True, text=True).stdout.strip() or 5)
    frames = []
    from PIL import Image
    for i in range(n):
        t = dur * (i + 0.5) / n
        p = f"prior_frames/f{i:02d}.png"
        subprocess.run([_ffmpeg(), "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", a.video, "-frames:v", "1", p], check=True)
        frames.append(p)
    ims = [np.array(Image.open(p).convert("RGB").resize((480, 270))) for p in frames]
    sheet = _sheet(ims, "prior_frames/sheet.png", labels=[f"{os.path.basename(p)} t={dur * (i + .5) / n:.1f}s" for i, p in enumerate(frames)])
    return dict(sheet=sheet, stills=frames, duration_s=dur)


# ------------------------------------------------------------------------------------------------ motions
def motions(a):
    import mujoco
    from humanoid_datagen.datagen_pipelines.shared.run import rollout, head_z, FALL_HEAD_Z
    from humanoid_datagen.datagen_pipelines.shared.compose import robot_only_model, robot_floor_model
    from humanoid_datagen.utils.sonic import Reference, set_robot_pose
    stage = task_from_env().stage                     # the push agent's tools: the push staging
    rm = robot_only_model()
    out = {}
    csvs = sorted(glob.glob("motions/*.csv"))
    far = robot_floor_model()  # the official G1 (as in compose / datagen / eval) on an open floor
    for csv in csvs:
        name = os.path.splitext(os.path.basename(csv))[0]
        ref = Reference.from_qpos(np.loadtxt(csv, delimiter=","), 30.0)
        st = stage(rm, ref, [0.1, 0.1, 0.1], friction=0.0, tip_safety=0.0)
        # tracker on flat ground, no object: does the motion stand on its own?
        tr = rollout(far, ref, "track", settle_s=0.0, tail_s=0.5)
        hz = head_z(far, tr["qpos"])
        travel = ref.root_pos[-1, :2] - ref.root_pos[0, :2]
        # sheet: robot alone at 8 frames, side view
        d = mujoco.MjData(rm)
        r = mujoco.Renderer(rm, 240, 320)
        cam = mujoco.MjvCamera()
        ims, labs = [], []
        for i in np.linspace(0, ref.T - 1, 8).astype(int):
            set_robot_pose(rm, d, ref, i)
            cam.lookat[:] = d.xpos[rm.body("pelvis").id]
            cam.distance, cam.azimuth, cam.elevation = 2.6, np.degrees(np.arctan2(*fwd_of(ref, i)[::-1])) - 90, -12
            r.update_scene(d, cam)
            ims.append(r.render())
            labs.append(f"{name} t={i / 50:.1f}s" + (" CONTACT" if abs(i - st["frame_contact"]) < 6 else ""))
        r.close()
        sheet = _sheet(ims, f"motions/{name}_sheet.png", labels=labs)
        out[name] = dict(duration_s=round(ref.T / 50, 2), t_contact_s=st["t_contact"],
                         palm_height_m=round(st["contact_height"], 3),
                         hands_push_dist_m=round(st["kinematic_push_dist"], 2),
                         root_travel_m=round(float(np.linalg.norm(travel)), 2),
                         stands_alone_under_tracker=bool(hz.min() > FALL_HEAD_Z),
                         sheet=sheet)
    json.dump(out, open("motions/summary.json", "w"), indent=1)
    return dict(motions=out, note="palm_height = where the palms meet the object's near face; pick an object "
                "taller than that. stands_alone=false: the motion leans on the object (it may still work with it).")


def fwd_of(ref, i):
    q = ref.root_quat[i]
    yaw = 2 * np.arctan2(q[3], q[0])
    return np.array([np.cos(yaw), np.sin(yaw)])


# ------------------------------------------------------------------------------------------------ compose / render
def _compose(a):
    from humanoid_datagen.datagen_pipelines.shared.compose import compose
    return compose(".", a.task, a.room)


def compose_tool(a):
    m, spec, ref, info = _compose(a)
    return dict(info, hint="checks.ok must be true before simulate; object_pose 'auto' puts the near face on the palms")


def _cams(model, names):
    import mujoco
    allc = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    if names:
        return [c for c in names.split(",") if c in allc or c in ("top", "follow")]
    return [c for c in allc if c] + ["top"]


def _render_view(model, data, r, cam, info):
    import mujoco
    if cam == "top":
        # hide ceilings / lamps (geoms entirely above 1.8 m) and look straight down at the floor area
        R = data.geom_xmat.reshape(-1, 3, 3)
        aabb = model.geom_aabb
        lo = data.geom_xpos[:, 2] + np.einsum("gk,gk->g", R[:, 2, :], aabb[:, :3]) \
            - np.einsum("gk,gk->g", np.abs(R[:, 2, :]), aabb[:, 3:])
        high = lo > 1.8
        rgba = model.geom_rgba.copy()
        model.geom_rgba[high, 3] = 0
        floor = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
        half = model.geom_size[floor[0], :2] if floor and model.geom_size[floor[0], 0] > 0 else np.array([4.0, 4.0])
        c = mujoco.MjvCamera()
        c.lookat[:] = [0, 0, 0]
        fovy = np.radians(model.vis.global_.fovy)
        c.distance = float(max(half[1], half[0] * 360 / 640) / np.tan(fovy / 2) * 1.05)
        c.azimuth, c.elevation = 90, -90
        r.update_scene(data, c)
        im = r.render()
        model.geom_rgba[:] = rgba
        return im
    elif cam == "follow":
        c = mujoco.MjvCamera()
        c.lookat[:] = data.xpos[model.body("pelvis").id]
        d = np.array(info["direction"])
        c.distance, c.elevation = 3.2, -18
        c.azimuth = np.degrees(np.arctan2(d[1], d[0])) + 120
        r.update_scene(data, c)
    else:
        r.update_scene(data, cam)
    return r.render()


def render(a):
    import mujoco
    from humanoid_datagen.utils.sonic import set_robot_pose
    m, spec, ref, info = _compose(a)
    out = _next_dir("renders", "r")
    d = mujoco.MjData(m)
    frames = {"start": 0, "contact": info["contact_frame"], "end": ref.T - 1}
    want = [f for f in a.frames.split(",") if f in frames]
    r = mujoco.Renderer(m, 360, 640)
    ims, labs = [], []
    for f in want:
        set_robot_pose(m, d, ref, frames[f])
        for cam in _cams(m, a.cameras):
            ims.append(_render_view(m, d, r, cam, info))
            labs.append(f"{f} | {cam}")
    r.close()
    sheet = _sheet(ims, os.path.join(out, "sheet.png"), cols=min(4, len(_cams(m, a.cameras))), labels=labs)
    return dict(sheet=sheet, checks=info["checks"], note="robot posed kinematically at each frame; object at its staged pose")


# ------------------------------------------------------------------------------------------------ simulate
def _simulate(a, out):
    import imageio.v2 as imageio
    import mujoco
    from humanoid_datagen.datagen_pipelines.shared.run import rollout, save
    tk = task_from_env()
    m, spec, ref, info = _compose(a)
    task = json.load(open(a.task))
    obj = info["object_body"]
    res = {}
    cams = [c for c in _cams(m, a.cameras) if c != "top"] if a.cameras else ["oak1"] + [
        c for c in _cams(m, None) if c not in ("top", "follow", "oak1")][:2]
    for mode in ("track", "kinematic"):
        tr = rollout(m, ref, mode, settle_s=task.get("settle_s", 0.5), tail_s=1.0, object_body=obj)
        mm = tk.metrics(tr, ref, info["direction"], m)
        save(tr, out, mode, mm)
        res[mode] = mm
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, 368, 640)
        sheet_ims, labs = [], []
        for cam in cams:
            path = os.path.join(out, f"{mode}_{cam}.mp4")
            with imageio.get_writer(path, fps=25, codec="libx264", quality=7) as w:
                for k in range(0, len(tr["qpos"]), 2):
                    d.qpos[:] = tr["qpos"][k]
                    mujoco.mj_forward(m, d)
                    im = _render_view(m, d, r, cam, info)
                    w.append_data(im)
                    if mode == "track" and k % 40 == 0 and len(sheet_ims) < 24:
                        sheet_ims.append(im)
                        labs.append(f"{cam} t={tr['t'][k]:.1f}s")
        r.close()
        if mode == "track":
            _sheet(sheet_ims, os.path.join(out, "track_sheet.png"), cols=4, labels=labs)
    t = res["track"]
    status = "pass" if (info["checks"]["ok"] and tk.passed(t)) else "fail"   # same rule as datagen acceptance and the eval
    summ = dict(status=status, track=t, kinematic=res["kinematic"], checks=info["checks"],
                object_pose=info["object_pose"], sheet=os.path.join(out, "track_sheet.png"),
                videos=sorted(glob.glob(os.path.join(out, "*.mp4"))), pass_rule=tk.pass_rule)
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    return summ, (m, spec, ref, info)


def simulate(a):
    return _simulate(a, _next_dir("sims", "s"))[0]


def finalize(a):
    if os.path.exists("final"):
        shutil.rmtree("final")
    os.makedirs("final")
    summ, (m, spec, ref, info) = _simulate(a, "final")
    for f in (a.room, a.task, a.assessment):
        shutil.copy(f, os.path.join("final", os.path.basename(f)))
    # assets referenced by the room (meshes / textures in the run dir)
    for sub in ("assets", "textures", "meshes"):
        if os.path.isdir(sub):
            shutil.copytree(sub, os.path.join("final", sub), dirs_exist_ok=True)
    q = np.load("final/track_traj.npz")
    bq = int(q["base_q"])
    sel = np.r_[np.arange(bq, bq + 7), q["robot_qadr"]]
    t = q["t"] - q["t"][0]
    idx = np.round(np.arange(0, t[-1], 1 / 30.0) * 50).astype(int)
    np.savetxt("final/tracked_motion.csv", q["qpos"][idx][:, sel], delimiter=",")
    ass = json.load(open(a.assessment))
    iters = len(glob.glob("sims/s*"))
    res = dict(status=summ["status"], finalized_by="cli finalize", track=summ["track"],
               kinematic=summ["kinematic"], checks=summ["checks"], object_pose=summ["object_pose"],
               assessment=ass, iterations=iters, final_dir="final")
    json.dump(res, open("result.json", "w"), indent=1)
    return res


def room_from_layout(a):
    from humanoid_datagen.datagen_pipelines.agents import scene
    if not a.layout:
        return dict(format=scene.__doc__, note="push.start_xy / third_camera keys are ignored here; set the start in "
                    "task.json and add cameras yourself. The box becomes the push_object body.")
    return dict(room=scene.room_xml_from_layout(json.load(open(a.layout)), a.out))


def main():
    ap = argparse.ArgumentParser(prog="datagen_pipelines.agents.cli")
    sub = ap.add_subparsers(dest="tool", required=True)
    p = sub.add_parser("keyframes"); p.add_argument("--video", default="prior/video.mp4"); p.add_argument("--n", type=int, default=12)
    sub.add_parser("motions")
    for name in ("compose", "render", "simulate", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--task", default="task.json")
        p.add_argument("--room", default="room.xml")
        p.add_argument("--cameras", default=None, help="comma list; default: all room cameras + oak1 + top + follow")
        if name == "render":
            p.add_argument("--frames", default="start,contact,end")
        if name == "finalize":
            p.add_argument("--assessment", required=True)
    p = sub.add_parser("room_from_layout"); p.add_argument("--layout", default=None); p.add_argument("--out", default="room.xml")
    a = ap.parse_args()
    fn = {"keyframes": keyframes, "motions": motions, "compose": compose_tool, "render": render,
          "simulate": simulate, "finalize": finalize, "room_from_layout": room_from_layout}[a.tool]
    try:
        res = fn(a)
    except Exception as e:  # report, don't crash: the agent acts on the message
        import traceback
        res = dict(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc().splitlines()[-6:])
    _log(a.tool, a, res)
    _out(res)
    sys.exit(0 if "error" not in res else 1)


if __name__ == "__main__":
    main()

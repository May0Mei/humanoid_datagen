"""Physics demos for VLA training, shared by every pipeline that produces rooms (datagen stage `demos`).

    python humanoid_datagen/scripts/datagen.py --config_name datagen/push/agents/demos

Per finalized agent scene (ours: outputs/push_box/scenes, text: scenes_text; both arms run the same code):
  spots    `n_spots` push spots (motion frame-0 pelvis xy, push yaw): the agent's own spot plus spots sampled in the
           room (task.spot_ok; push: floor and a free corridor from behind the standing start, `travel` m
           behind the motion, to `corridor_push_len` m beyond the object's far face, as the test tasks), all of it inside
           the walls (sampler.inside_room). Spots of an earlier run (spots.json) are kept, so its demos stay valid;
           kept spots outside the room are replaced at the same index and their demos moved to _replaced_spots/. The agent's spot is replaced
           by a sampled one when its corridor is blocked. New spots differ from earlier ones by >= `spot_min_sep`
           m or >= `spot_min_dyaw` deg. Written to <out>/<scene>/spots.json.
  motions  every motion of the prior with a lead-in (priors/<pid>/leadin/m<k>.csv = stand -> lead-in -> motion).
  demos    spot x motion: jitter as datagen v1 (start +-start_jitter_m sideways, push yaw +-yaw_jitter_deg,
           the object's masses and per-geom friction x U(1-j, 1+j), relative to the agent's). The motion is staged at the spot as in
           v1 (object under the palms); lead-in ++ motion is moved rigidly with it. Static checks: palms on the
           object's face; no robot penetration with the room while standing and walking in. When the bend reaches
           past the motion's frame-0 hands into the object, the object moves ahead along the push direction until
           the walk-in clears it (<= `max_object_shift` m; the push then starts with that gap). SONIC
           tracks the whole episode from standing; accepted with the task's pass rule (task.passed) and demo filter (task.accept_demo) plus no hand-object force before the motion starts. Up to `max_tries` jitter
           draws per (spot, motion).
  render   accepted demos: output_oak1.mp4 (oak_cam: calibrated OAK-1, HALO pinhole -> K-B fisheye, 1280x800,
           30 fps), output.csv (tracked qpos, 30 fps, Kimodo layout, rows = video frames), prompt.txt, task.json,
           metrics.json, traj.npz (raw 50 Hz trajectory, for rescoring); output_tpv.mp4 (third person) for the
           first accepted demo of every spot.
<out>/<scene>/index.jsonl ({demo_id, accepted, ...}) as vla/export.py reads it; written when the
scene is done (scenes with an index are skipped; demos already on disk are reused).
"""
from __future__ import annotations

import glob
import json
import os
import time
import zlib

import mujoco
import numpy as np

from humanoid_datagen.datagen_pipelines.shared.compose import (episode_frame, ROBOT_CONTYPE, _is_desc, compose_room, model_extent, robot_only_model,
                                              room_spec)
from humanoid_datagen.datagen_pipelines.shared.run import load_traj, rollout
from humanoid_datagen.datagen_pipelines.shared.base import Stage
from humanoid_datagen.datagen_pipelines.shared.sampler import floor_boxes, inside_room
from humanoid_datagen.utils.sonic import Reference, set_robot_pose
from humanoid_datagen.utils.paths import repo_path

FPS = 30.0


# ------------------------------------------------------------------------------------------------ scene + motions
def load_scene(run_dir: str, C: dict, tk) -> dict:
    """tk: the task module (tasks/load.py)."""
    task0 = json.load(open(os.path.join(run_dir, "final", "task.json")))
    obj = task0.get("object_body", "push_object")
    m0 = room_spec(os.path.join(run_dir, "room.xml")).compile()
    pid = json.load(open(os.path.join(run_dir, "_driver", "policy.json")))["prior_id"]
    pd = os.path.join(repo_path(C["priors_root"]), pid)
    category = json.load(open(os.path.join(pd, "meta.json")))["object"]
    mass0, mu0 = _object_physics(m0, obj)
    ext = model_extent(m0, obj)
    rm = robot_only_model()
    motions = []
    allow = None
    if C.get("motion_allow"):              # {scene: [motion names]}: only these motions (hand-rule refill)
        allow = set(json.load(open(repo_path(C["motion_allow"]))).get(os.path.basename(run_dir.rstrip("/")), []))
    for js in sorted(glob.glob(os.path.join(pd, "leadin", "m*.json")), key=lambda p: int(os.path.basename(p)[1:-5])):
        if allow is not None and os.path.basename(js)[:-5] not in allow:
            continue
        meta = json.load(open(js))
        Q = np.loadtxt(js[:-5] + ".csv", delimiter=",")
        ms = int(meta["motion_start"])
        clip = Q[ms:]
        fr = tk.motion_frame(rm, Reference.from_qpos(clip, FPS), ext["half"])
        motions.append(dict(name=os.path.basename(js)[:-5], Q=Q, ms=ms, clip=clip, dir=fr["dir"],
                            ahead=fr["reach"], push=fr["travel"], meta=meta))
    return dict(tk=tk, run_dir=run_dir, scene=os.path.basename(run_dir.rstrip("/")), task0=task0, obj=obj, model0=m0,
                mass0=mass0, mu0=mu0, ext=ext, pid=pid, category=category, motions=motions)


def _object_physics(m, obj):
    """Total mass and the largest sliding friction among the object's colliding geoms (what touches the floor)."""
    b = m.body(obj).id
    sub = {b} | {i for i in range(m.nbody) if _is_desc(m, i, b)}
    mus = [m.geom_friction[g, 0] for g in range(m.ngeom)
           if m.geom_bodyid[g] in sub and (m.geom_contype[g] or m.geom_conaffinity[g])]
    return float(model_extent(m, obj)["mass"]), float(max(mus))


# ------------------------------------------------------------------------------------------------ step 2: spots
def sample_spots(S: dict, C: dict, rng, keep: list | None = None) -> dict:
    """`keep`: spots already used (an earlier run's spots.json); kept in order, more sampled to reach n_spots."""
    m = S["model0"]
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ob = m.body(S["obj"]).id
    # free floor up to the object's far face plus corridor_push_len beyond it (= the test's corridor rule), not the
    # motion's whole push: success needs 0.3 m, and some Kimodo motions push 3-6 m
    ahead = max(mm["ahead"] for mm in S["motions"]) + float(C["corridor_push_len"]) + float(C["corridor_margin"])
    back = float(C["travel"]) + float(C["back_margin"])

    def inside(xy, yaw):      # the whole corridor inside the walls (agent floor planes extend past them)
        u = np.array([np.cos(yaw), np.sin(yaw)])
        return all(inside_room(m, d, np.asarray(xy) + u * t, ob) for t in np.arange(-back, ahead + 1e-6, 0.5))

    def clear(xy, yaw):
        return S["tk"].spot_ok(m, d, xy, yaw, S["ext"], ahead, back, exclude_body=ob) and inside(xy, yaw)

    def distinct(xy, yaw, spots):
        for s in spots:
            s_xy, s_yaw = episode_frame(s)
            dy = abs((np.degrees(yaw) - s_yaw + 180) % 360 - 180)
            if np.linalg.norm(np.asarray(xy) - s_xy) < float(C["spot_min_sep"]) and \
                    dy < float(C["spot_min_dyaw"]):
                return False
        return True

    t0 = S["task0"]
    a_xy, a_yaw = episode_frame(t0)
    a_xy, a_yaw = np.asarray(a_xy, float), np.radians(a_yaw)
    agent_clear = bool(clear(a_xy, a_yaw))
    # kept spots outside the room (sampled before the inside check) are replaced in place, same index
    spots = [s_ if inside(episode_frame(s_)[0], np.radians(episode_frame(s_)[1])) else None for s_ in (keep or [])]
    replaced = [i for i, s_ in enumerate(spots) if s_ is None]
    if agent_clear and not any(s_ and s_["source"] == "agent" for s_ in spots) and \
            (None in spots or len(spots) < int(C["n_spots"])):
        a = dict(start=a_xy.tolist(), yaw_deg=float(np.degrees(a_yaw)), source="agent")
        if None in spots:
            spots[spots.index(None)] = a
        else:
            spots.append(a)
    boxes = floor_boxes(m, d)
    if not boxes:
        c, e = m.stat.center, float(m.stat.extent)
        boxes = [(c[:2] - e, c[:2] + e)]
    area = np.array([np.prod(hi - lo) for lo, hi in boxes])
    tries = 0
    while (None in spots or len(spots) < int(C["n_spots"])) and tries < int(C["spot_max_tries"]):
        tries += 1
        lo, hi = boxes[int(rng.choice(len(boxes), p=area / area.sum()))]
        xy, yaw = rng.uniform(lo, hi), rng.uniform(-np.pi, np.pi)
        if clear(xy, yaw) and distinct(xy, yaw, [s_ for s_ in spots if s_]):
            new_ = dict(start=[float(xy[0]), float(xy[1])], yaw_deg=float(np.degrees(yaw)),
                        source="sampled")
            if None in spots:
                spots[spots.index(None)] = new_
            else:
                spots.append(new_)
    if None in spots:                    # unfilled: cut there, so later spots keep their index (their demos go too)
        k = spots.index(None)
        replaced = sorted(set(replaced) | set(range(k, len(spots))))
        spots = spots[:k]
    return dict(spots=spots, agent_spot_clear=agent_clear, tries=tries, ahead=ahead, back=back,
                replaced=replaced)


# ------------------------------------------------------------------------------------------------ step 4: one demo
def scale_object_physics(spec, obj, fm: float, ff: float):
    """Scale the object's masses by fm and every colliding geom's sliding friction by ff. Relative, per geom: the
    agents give objects several collision geoms with different friction (e.g. a slippery side the hands touch
    and a grippy base on the floor; datagen v1's _scale_object_physics set them all to one value)."""
    def walk(b):
        for g in b.geoms:
            if g.mass > 0:
                g.mass = g.mass * fm
            elif g.contype or g.conaffinity:
                g.density = g.density * fm
            if g.contype or g.conaffinity:
                g.friction = [g.friction[0] * ff, g.friction[1], g.friction[2]]
        for c in b.bodies:
            walk(c)
    walk(spec.body(obj))


def leadin_penetrations(model, ref: Reference, upto: int, obj_body: int, step: int = 3) -> tuple[list[str], float]:
    """Robot penetrations at 50 Hz frames 0..upto (standing + walk-in): (room geoms penetrated by > 1 cm, the deepest
    robot-object contact distance, negative = penetration)."""
    d = mujoco.MjData(model)
    rob = model.geom_contype == ROBOT_CONTYPE
    room, obj_min = set(), 0.0
    for i in list(range(0, upto, step)) + [max(upto - 1, 0)]:
        set_robot_pose(model, d, ref, i)
        mujoco.mj_forward(model, d)
        for c in d.contact[:d.ncon]:
            if not (rob[c.geom1] ^ rob[c.geom2]):
                continue
            g = int(c.geom2 if rob[c.geom1] else c.geom1)
            if _is_desc(model, int(model.geom_bodyid[g]), obj_body):
                obj_min = min(obj_min, float(c.dist))
            elif c.dist < -0.01 and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE:
                room.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}")
    return sorted(room), obj_min


def build(S, spot, mm, rng, C):
    """Jittered, composed episode: (model, info, ref, ms50, jitter) or (None, reason)."""
    s_xy, s_yaw = episode_frame(spot)
    y = s_yaw + rng.uniform(-1, 1) * float(C["yaw_jitter_deg"])
    yr = np.radians(y)
    lat = rng.uniform(-1, 1) * float(C["start_jitter_m"])
    start = np.asarray(s_xy, float) + lat * np.array([-np.sin(yr), np.cos(yr)])
    j = float(C["physics_jitter"])
    fm, ff = float(rng.uniform(1 - j, 1 + j)), float(rng.uniform(1 - j, 1 + j))
    mass, mu = S["mass0"] * fm, S["mu0"] * ff          # mu: the floor-contact (max) friction, for the record
    spec = room_spec(os.path.join(S["run_dir"], "room.xml"))
    scale_object_physics(spec, S["obj"], fm, ff)
    xml = os.path.join(S["run_dir"], "_room_datagen.xml")   # asset paths are relative to the run dir
    open(xml, "w").write(spec.to_xml())
    task = dict({k: v for k, v in S["task0"].items() if k not in ("robot_start", "push_yaw_deg")},
                start=start.tolist(), yaw_deg=float(y), object_pose="auto")
    task.pop("motion", None)
    jit = dict(start=start.tolist(), yaw_deg=float(y), lateral=float(lat), mass=mass, mu=mu, mass_scale=fm,
               friction_scale=ff)
    ref0 = Reference.from_qpos(mm["clip"], FPS)
    try:
        model, _, _, info = compose_room(xml, task, S["tk"], S["run_dir"], ref0=ref0)
    except Exception as e:  # noqa: BLE001
        return None, f"compose: {e}", jit
    if not info["checks"]["palms_on_object_face"]:
        return None, f"checks: {info['checks']}", jit
    # lead-in ++ motion: the rigid transform that staged the bare motion (frame ms -> start, motion direction -> yaw)
    rot = yr - np.arctan2(mm["dir"][1], mm["dir"][0])
    R2 = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    Q, ms = mm["Q"], mm["ms"]
    ref = Reference.from_qpos(Q, FPS).transformed(start - R2 @ (Q[ms, :2] - Q[0, :2]), rot)
    ms50 = int(round(ms / FPS * 50))
    upto = max(ms50 - int(C["seam_skip_s"] * 50), 1)
    # the bend can reach past the motion's frame-0 hands (motions that start with the palms on the object): move the
    # object ahead along the push direction until the walk-in clears it (<= max_object_shift)
    u, shift = np.array([np.cos(yr), np.sin(yr)]), 0.0
    pose0 = list(info["object_pose"])
    for _ in range(4):
        room, omin = leadin_penetrations(model, ref, upto, model.body(info["object_body"]).id)
        if room:
            return None, f"lead-in penetrations: {room}", jit
        if omin >= -0.01:
            break
        shift += -omin + 0.01
        if shift > float(C["max_object_shift"]):
            return None, f"lead-in reaches {shift:.2f} m into the object", jit
        task["object_pose"] = [pose0[0] + shift * u[0], pose0[1] + shift * u[1], pose0[2]]
        model, _, _, info = compose_room(xml, task, S["tk"], S["run_dir"], ref0=ref0)
    else:
        return None, f"lead-in still touches the object after {shift:.2f} m", jit
    jit["object_shift"] = float(shift)
    return (model, info, ref, ms50), None, jit


def simulate(model, info, ref, ms50, C, tk):
    tr = rollout(model, ref, "track", settle_s=0.5, tail_s=float(C["tail_s"]), object_body=info["object_body"])
    mm = tk.metrics(tr, ref, info["direction"], model)
    fr = np.asarray(tr["ref_frame"])
    f = np.asarray(tr["hand_obj_force"])
    pre = fr < ms50 - int(C["seam_skip_s"] * 50)
    mm["max_force_before_motion"] = float(f[pre].max()) if pre.any() else 0.0
    mm["motion_start_s"] = ms50 / 50.0
    ok = tk.passed(mm) and mm["max_force_before_motion"] <= float(C["max_pre_force"])
    if ok:                                # the task's demo filter (push: the hand rule)
        chk = tk.accept_demo(model, tr, info)
        if len(chk) > 1:
            mm["demo_check"] = chk
        ok = bool(chk["pass"])
    return tr, mm, bool(ok)


def reuse_demo(S, C, od, demo_id, ddir, rj):
    """Hand-rule refill: a demo of the `reuse_root` run (same spots, same seeded draws) is not simulated again. An
    accepted one is re-checked with the hand rule from its traj.npz: pass -> <demo>/ with links to the old files and
    its own row.json; fail -> _rejected/<demo>.json. A rejected one is copied as it is. None: not in that run."""
    src = os.path.join(repo_path(C["reuse_root"]), S["scene"], demo_id)
    src_rj = os.path.join(repo_path(C["reuse_root"]), S["scene"], "_rejected", demo_id + ".json")
    if os.path.exists(os.path.join(src, "row.json")):
        row = json.load(open(os.path.join(src, "row.json")))
        t = json.load(open(os.path.join(src, "task.json")))
        ti = t["info"]
        task = dict(object_body=ti["object_body"], object_pose=ti["object_pose"])
        task["start"], task["yaw_deg"] = episode_frame(ti)
        model, _, _, info = compose_room(os.path.join(S["run_dir"], "room.xml"), task, S["tk"], S["run_dir"])
        info = dict(info, direction=ti.get("direction", ti.get("push_dir")))
        z = load_traj(os.path.join(src, "traj.npz"))
        hc = S["tk"].accept_demo(model, dict(qpos=z["qpos"], obj_pos=z["obj_pos"]), info)
        row = dict(row, demo_check=hc, reused_from=src)
        if hc["pass"]:
            os.makedirs(ddir, exist_ok=True)
            for f in os.listdir(src):
                if f != "row.json" and not os.path.exists(os.path.join(ddir, f)):
                    os.symlink(os.path.join(src, f), os.path.join(ddir, f))
            json.dump(row, open(os.path.join(ddir, "row.json"), "w"), indent=1, default=float)
        else:
            row["accepted"] = False
            row["reason"] = "demo_check"
            os.makedirs(os.path.dirname(rj), exist_ok=True)
            json.dump(row, open(rj, "w"), indent=1, default=float)
        return row, hc["pass"] and os.path.exists(os.path.join(src, "output_tpv.mp4"))
    if os.path.exists(src_rj):
        row = json.load(open(src_rj))
        os.makedirs(os.path.dirname(rj), exist_ok=True)
        json.dump(row, open(rj, "w"), indent=1, default=float)
        return row, False
    return None


# ------------------------------------------------------------------------------------------------ step 5: render
def _writer(path):
    import imageio.v2 as imageio
    return imageio.get_writer(path, fps=int(FPS), codec="libx264", quality=int(_writer.quality), macro_block_size=1)


_writer.quality = 7


def write_demo(out_dir, model, tr, mm, task_meta, instr, tpv: bool):
    from humanoid_datagen.utils.oak_cam import OakCam
    os.makedirs(out_dir, exist_ok=True)
    t = np.asarray(tr["t"]) - tr["t"][0]
    idx = np.minimum(np.round(np.arange(0, t[-1] + 1e-9, 1 / FPS) * 50).astype(int), len(t) - 1)
    d = mujoco.MjData(model)
    cam = OakCam(model)
    tp = None
    if tpv:
        names = [model.camera(i).name for i in range(model.ncam) if model.camera(i).name != "oak1"]
        if names:
            tp = mujoco.Renderer(model, 600, 960)
            tpc = next((c for c in ("overview_cam", "corner", "side") if c in names), names[0])
            w2 = _writer(os.path.join(out_dir, "output_tpv.mp4"))
    w1 = _writer(os.path.join(out_dir, "output_oak1.mp4"))
    for k in idx:
        d.qpos[:] = tr["qpos"][k]
        mujoco.mj_forward(model, d)
        w1.append_data(cam.render(d))
        if tp is not None:
            tp.update_scene(d, camera=tpc)
            w2.append_data(tp.render())
    w1.close()
    cam.close()
    if tp is not None:
        w2.close()
        tp.close()
    q = np.asarray(tr["qpos"])
    bq = int(tr["base_q"])
    sel = np.r_[np.arange(bq, bq + 7), tr["robot_qadr"]]
    np.savetxt(os.path.join(out_dir, "output.csv"), q[idx][:, sel], delimiter=",", fmt="%.7g")
    open(os.path.join(out_dir, "prompt.txt"), "w").write(instr + "\n")
    json.dump(dict(task_meta, instruction=instr, frames=len(idx), fps=FPS,
                   motion_start_frame=int(round(mm["motion_start_s"] * FPS))),
              open(os.path.join(out_dir, "task.json"), "w"), indent=1, default=float)
    json.dump(mm, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1, default=float)
    np.savez_compressed(os.path.join(out_dir, "traj.npz"), **tr)


# ------------------------------------------------------------------------------------------------ driver
def run_demo(S, C, od, si, spot, mm, instr, tpv, prior_rejected):
    """One (spot, motion) demo: up to `max_tries` jitter draws. Accepted -> <demo>/row.json; rejected ->
    _rejected/<demo>.json. Either is reused on resume (the draws are seeded by the demo id, so a rerun repeats them)."""
    sc = S["scene"]
    demo_id = f"{sc}__s{si}_{mm['name']}"
    ddir = os.path.join(od, demo_id)
    rj = os.path.join(od, "_rejected", demo_id + ".json")
    if os.path.exists(os.path.join(ddir, "row.json")):                # resumed (written last)
        return json.load(open(os.path.join(ddir, "row.json"))), os.path.exists(os.path.join(ddir, "output_tpv.mp4"))
    if os.path.exists(rj):
        return json.load(open(rj)), False
    if demo_id in prior_rejected:                                     # rejected in a run before _rejected/ existed
        return prior_rejected[demo_id], False
    if C.get("reuse_root"):
        got = reuse_demo(S, C, od, demo_id, ddir, rj)
        if got is not None:
            print(f"[v2] {demo_id}: reused, accepted={got[0]['accepted']}", flush=True)
            return got
    drng = np.random.default_rng([int(C["seed"]), zlib.crc32(demo_id.encode())])
    t = time.time()
    tries = []
    row = dict(demo_id=demo_id, accepted=False, spot=si, motion=mm["name"], category=S["category"])
    for k in range(int(C["max_tries"])):
        built, why, jit = build(S, spot, mm, drng, C)
        if built is None:
            tries.append(dict(jit, reason=why))
            continue
        model, info, ref, ms50 = built
        tr, met, ok = simulate(model, info, ref, ms50, C, S["tk"])
        tries.append(dict(jit, push=met["push_dist"], fell=met["robot_fell"],
                          tilt=met["max_tilt_deg"], pre_force=met["max_force_before_motion"], ok=ok))
        if ok:
            meta = dict(source="agent_v2", scene_id=sc, demo_id=demo_id, run_dir=S["run_dir"], prior=S["pid"],
                        spot=dict(spot, index=si), motion=mm["meta"]["motion"], leadin=f"leadin/{mm['name']}.csv",
                        jitter=jit, agent_mass=S["mass0"], agent_mu=S["mu0"], target=dict(category=S["category"]),
                        info={k2: v for k2, v in info.items() if k2 != "checks"}, attempt=k,
                        physics_jitter=float(C["physics_jitter"]))
            write_demo(ddir, model, tr, met, meta, instr, tpv=tpv)
            row.update(accepted=True, push=met["push_dist"], tilt=met["max_tilt_deg"], attempt=k)
            break
    row["tries"] = tries
    if row["accepted"]:
        json.dump(row, open(os.path.join(ddir, "row.json"), "w"), indent=1, default=float)
    else:
        os.makedirs(os.path.dirname(rj), exist_ok=True)
        json.dump(row, open(rj, "w"), indent=1, default=float)
    print(f"[v2] {demo_id}: accepted={row['accepted']} after {len(tries)} tries "
          f"[{time.time() - t:.0f}s] {tries[-1].get('reason', '') if not row['accepted'] else ''}", flush=True)
    return row, row["accepted"] and tpv


def extra_spot(S, C, sc, spots):
    """One more spot for the top-up, separation relaxed step by step (crowded rooms fit few distinct spots)."""
    for f in C["topup_sep_relax"]:
        C2 = dict(C, n_spots=len(spots) + 1, spot_min_sep=float(C["spot_min_sep"]) * f,
                  spot_min_dyaw=float(C["spot_min_dyaw"]) * f, spot_max_tries=int(C["topup_spot_tries"]))
        rng = np.random.default_rng([int(C["seed"]), zlib.crc32(sc.encode()), 1, len(spots), int(f * 100)])
        sp = sample_spots(S, C2, rng, keep=spots)
        if len(sp["spots"]) > len(spots) and not sp["replaced"]:
            return dict(sp["spots"][-1], source="topup", sep_factor=f)
    return None


def run_scene(run_dir: str, C: dict, out_root: str, tk):
    S = load_scene(run_dir, C, tk)
    sc = S["scene"]
    od = os.path.join(out_root, sc)
    os.makedirs(od, exist_ok=True)
    rng = np.random.default_rng([int(C["seed"]), zlib.crc32(sc.encode())])
    if not S["motions"]:
        print(f"[v2] {sc}: no lead-in motions for {S['pid']}", flush=True)
        return []
    prior_rejected = {}
    if os.path.exists(os.path.join(od, "index.jsonl")):
        for line in open(os.path.join(od, "index.jsonl")):
            r = json.loads(line)
            if not r["accepted"]:
                prior_rejected[r["demo_id"]] = r
    sj = os.path.join(od, "spots.json")
    if not os.path.exists(sj) and C.get("reuse_root"):             # same spots as the reused run
        rsj = os.path.join(repo_path(C["reuse_root"]), sc, "spots.json")
        if os.path.exists(rsj):
            json.dump(json.load(open(rsj)), open(sj, "w"), indent=1)
    old = json.load(open(sj)) if os.path.exists(sj) else None
    # base spots: the first n_spots (later ones are top-up spots, kept as they are)
    base_keep = [s_ for s_ in (old or {}).get("spots", []) if s_.get("source") != "topup"]
    extra = [s_ for s_ in (old or {}).get("spots", []) if s_.get("source") == "topup"]
    if extra:      # topped up before: the base spots are settled (re-sampling them could shift top-up indices)
        sp = dict({k: v for k, v in old.items() if k != "spots"}, spots=base_keep, replaced=[])
    else:
        sp = sample_spots(S, C, rng, keep=base_keep or None)
    for si in sp["replaced"]:          # demos at a replaced (outside-the-room) spot: moved aside, re-generated
        stale = sorted(glob.glob(os.path.join(od, f"{sc}__s{si}_*")))
        if stale:
            bin_ = os.path.join(od, "_replaced_spots", time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(bin_, exist_ok=True)
            for p in stale:
                os.rename(p, os.path.join(bin_, os.path.basename(p)))
            print(f"[v2] {sc}: spot {si} was outside the room; moved {len(stale)} demos to {bin_}", flush=True)
    sp["spots"] = sp["spots"] + extra
    json.dump(sp, open(sj, "w"), indent=1)
    print(f"[v2] {sc} ({S['pid']}, {S['category']}): {len(sp['spots'])} spots ({len(extra)} top-up; agent spot clear: "
          f"{sp['agent_spot_clear']}), {len(S['motions'])} motions", flush=True)
    instr = tk.instruction(S["category"])
    by_name = {mm["name"]: mm for mm in S["motions"]}
    rows = []
    for si, spot in enumerate(sp["spots"]):
        tpv_done = False
        names = spot["motions"] if spot.get("source") == "topup" else [mm["name"] for mm in S["motions"]]
        for name in [n for n in names if n in by_name]:          # (motion_allow may drop some)
            row, tpv = run_demo(S, C, od, si, spot, by_name[name], instr, not tpv_done, prior_rejected)
            tpv_done = tpv_done or tpv
            rows.append(row)

    # top-up: more spots until `target_per_scene` accepted; at each new spot the live motions with the fewest
    # accepted demos go first; a motion with no accepted demo after `dead_after` attempts is dropped
    target = int(C["target_per_scene"])
    acc = sum(r["accepted"] for r in rows)
    while acc < target and sum(s_.get("source") == "topup" for s_ in sp["spots"]) < int(C["topup_max_spots"]):
        na = {n: sum(r["accepted"] for r in rows if r["motion"] == n) for n in by_name}
        nt = {n: sum(1 for r in rows if r["motion"] == n) for n in by_name}
        live = sorted((n for n in by_name if na[n] > 0 or nt[n] < int(C["dead_after"])),
                      key=lambda n: (na[n], nt[n], int(n[1:])))
        if not live:
            print(f"[v2] {sc}: top-up stops, no live motion", flush=True)
            break
        spot = extra_spot(S, C, sc, sp["spots"])
        if spot is None:
            print(f"[v2] {sc}: top-up stops, no further spot fits", flush=True)
            break
        spot["motions"] = []
        sp["spots"].append(spot)
        si = len(sp["spots"]) - 1
        tpv_done = False
        for n in live[:target - acc + 3]:      # a few spare: some will be rejected; the rest go to the next spot
            if acc >= target:
                break
            spot["motions"].append(n)
            json.dump(sp, open(sj, "w"), indent=1)     # planned before it runs, so a resume repeats the same demo
            row, tpv = run_demo(S, C, od, si, spot, by_name[n], instr, not tpv_done, prior_rejected)
            tpv_done = tpv_done or tpv
            rows.append(row)
            acc += row["accepted"]
        json.dump(sp, open(sj, "w"), indent=1)
    with open(os.path.join(od, "index.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    json.dump(dict(target=target, accepted=acc, spots=len(sp["spots"]),
                   topup_spots=sum(s_.get("source") == "topup" for s_ in sp["spots"])),
              open(os.path.join(od, "done.json"), "w"), indent=1)
    print(f"[v2] {sc}: {acc}/{len(rows)} accepted, {len(sp['spots'])} spots", flush=True)
    return rows


class PhysicsDemos(Stage):
    """Datagen stage `demos`: every passed scene of this shard (`shard` / `num_shards`, or SLURM_ARRAY_TASK_ID /
    SLURM_ARRAY_TASK_COUNT) not done yet."""

    def run(self) -> None:
        C, tk = self.cfg, self.task
        _writer.quality = int(C.get("video_quality", 7))
        out_root = repo_path(C["out_root"])
        runs = sorted(d for d in glob.glob(os.path.join(repo_path(C["scenes_root"]), "*"))
                      if os.path.isfile(os.path.join(d, "result.json")) and ".attempt" not in d
                      and json.load(open(os.path.join(d, "result.json"))).get("status") == "pass")
        if C.get("scenes"):
            runs = [r for r in runs if os.path.basename(r) in set(C["scenes"])]
        sh = int(os.environ.get("SLURM_ARRAY_TASK_ID", C.get("shard", 0)))
        ns = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", C.get("num_shards", 1)))
        for rd in runs[sh::ns]:
            dj = os.path.join(out_root, os.path.basename(rd), "done.json")
            if os.path.exists(dj) and json.load(open(dj))["target"] == int(C["target_per_scene"]):
                continue
            run_scene(rd, C, out_root, tk)

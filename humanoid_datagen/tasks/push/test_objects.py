"""Test objects for the GRScenes push evaluation: objects no arm trains on. Built once with
eval/push/build_grscenes.yaml (eval mode `build`, task.build_test_set); episodes compose with compose_task.

Two sources, both with mass / friction drawn from the test-only ranges (`test_physics` of the build config):
  grscenes  furniture that already stands on the floor of the home (chairs, armchairs, nightstands, desks, tables,
            stools, boxes, baskets), of a training type by category + size; made free-moving in place (its own
            convex colliders, `free_body`), the robot starts `gap` m behind it with a free corridor beyond it;
  library   held-out THOR / Objaverse assets per training type (`held_out.split_file`: picked, reviewed and curated
            once; excluded from the agents' library), placed by the independent sampler.
Pieces are pushed in place only when the home leaves a free run behind and beyond them (rare); otherwise they are
relocated to free floor of the same home by the sampler (task["placement"]).
"""
from __future__ import annotations

import glob
import json
import os
import queue
import sys
import time

import mujoco
import numpy as np

from humanoid_datagen.utils.paths import repo_path

def _library(C: dict):
    """The library module pointed at the full index (the agents' copy excludes the held-out assets)."""
    from humanoid_datagen.datagen_pipelines.shared import library
    h = C["held_out"]
    d = repo_path(h["library_dir"])
    if not os.path.isfile(os.path.join(d, "meta.jsonl")):
        library.export_index(repo_path(h["library_index"]), d)
    library.DATA = d
    return library
# GRScenes category -> training type, decided by the piece's size (D, W, H = footprint and height in m)
LIMITS = dict(min_foot=0.25, max_foot=1.6, min_h=0.35, max_h=1.3)


def _gr_type(cat: str, D: float, W: float, H: float) -> str | None:
    f = max(D, W)
    if cat == "chair":
        return "chair"                      # not a training type: reported as its own split
    if cat == "armchair":
        return "armchair"
    if cat == "nightstand":
        return "nightstand"
    if cat == "desk":
        return "desk"
    if cat == "table":
        if H >= 0.65 and f >= 0.9:
            return "dining_table"
        if H <= 0.75 and f <= 0.7:
            return "side_table"
    if cat == "coffee_table" and f <= 0.8:
        return "side_table"
    if cat == "stool" and H <= 0.55:
        return "ottoman"
    if cat == "box" and 0.3 <= H <= 0.9 and 0.3 <= f <= 0.9:
        return "cardboard_box"
    if cat == "basket" and H >= 0.4:
        return "laundry_hamper"
    return None


GRSCENES_CATEGORIES = ("chair", "armchair", "nightstand", "desk", "table", "coffee_table", "stool", "box", "basket")
TRAINING_TYPES = ("cardboard_box", "wardrobe_box", "beer_case", "plastic_crate", "side_table", "dining_table", "desk",
                  "ottoman", "laundry_hamper", "storage_chest", "office_chair", "laundry_cart", "suitcase",
                  "rolling_bookshelf", "armchair", "bench", "nightstand", "pet_crate")


def physics_ranges(C: dict) -> dict:
    return C["test_physics"]


def draw_physics(ranges: dict, typ: str, rng: np.random.Generator) -> dict:
    r = ranges[typ]
    return dict(mass=float(rng.uniform(*r["mass"])), mu=float(rng.uniform(*r["mu"])))




# ------------------------------------------------------------------------------------------------ free-moving bodies
def _walk(b):
    yield b
    for c in b.bodies:
        yield from _walk(c)


# Contact impedance of the test objects: MuJoCo's default. The library assets ship solimp 0.998 0.998 (near-rigid,
# with priority 1, so it wins every contact); with their convex hulls resting on a few points that gave solver
# impulse spikes that launched ~1 in 4 library objects with nothing touching them (2026-11-21, tmp/object_settle).
SOLIMP = [0.9, 0.95, 0.001, 0.5, 2.0]


def free_body(spec: mujoco.MjSpec, body: str, mass: float, mu: float) -> dict:
    """Give an existing (static) body a free joint, `mass` kg over its colliding geoms by volume, friction `mu`,
    contact solimp SOLIMP. Visual geoms (contype = conaffinity = 0) are massless. Returns {mass, n_colliders}."""
    b = spec.body(body)
    if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in b.joints):
        b.add_freejoint(name=f"{body}_free")
    col = []
    for bb in _walk(b):
        for g in bb.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                g.mass = 0.0
                g.density = 0.0
            else:
                g.mass = float("nan")   # unset: density decides
                g.density = 1000.0
                g.friction = [mu, 0.005, 0.0001]
                g.solimp = SOLIMP
                col.append(g)
    if not col:
        raise ValueError(f"body {body!r} has no colliding geoms")
    m = spec.compile()
    vol = float(m.body_subtreemass[m.body(body).id]) / 1000.0     # m^3 of the colliders
    for g in col:
        g.density = mass / vol
    return dict(mass=mass, n_colliders=len(col), volume_m3=round(vol, 4))


def attach_library_asset(spec: mujoco.MjSpec, asset_xml: str, name: str = "push_object", quat=None) -> None:
    """Library asset (library.export: origin bottom centre, front +x) inside a new body `name`, no joint yet.
    `quat` (wxyz, from the split entry): the asset's orientation inside the body, e.g. a flat-lying suitcase stood
    upright (the composer puts the lowest collider point on the floor whatever the rotation)."""
    child = mujoco.MjSpec.from_file(asset_xml)
    top = child.worldbody.first_body()
    body = spec.worldbody.add_body(name=name)
    frame = body.add_frame()
    if quat is not None:
        frame.quat = [float(v) for v in quat]
    frame.attach_body(top, f"{name}/", "")


def prepare_spec(task: dict):
    """Hook for compose.compose_room: make the task's object free-moving with its test physics."""
    obj = task["object"]

    def prep(spec: mujoco.MjSpec):
        if task["source"] == "library":
            attach_library_asset(spec, repo_path(obj["asset"]), task["object_body"], obj.get("quat"))
        free_body(spec, task["object_body"], float(obj["mass"]), float(obj["mu"]))
    return prep


def compose_task(task: dict, tk, timestep: float = 0.005):
    """Exact object geometry: lowest collider vertex on the floor, centre of mass on the push line (auto poses)."""
    from humanoid_datagen.datagen_pipelines.shared.compose import compose_room
    return compose_room(repo_path(task["room"]), task, tk, prepare=prepare_spec(task), timestep=timestep,
                        exact_object=True)


def settle_check(task: dict, sc: dict, tk, on_settled=None) -> tuple[bool, dict]:
    """Simulate the eval's settle on a drawn task and keep it only if the object rests free on the floor
    (docs/eval_push_issues.md section 2). Robot pinned at its standing start as in the eval; the object settles
    `settle_s`, then is held `hold_s` more. Rejected when: it tilts more than `max_settle_tilt_deg` while settling
    (rocks onto its legs, folds over); it moves more than `max_drift_m` or turns more than `max_drift_tilt_deg`
    during the hold (still sliding / rocking); or it touches anything but the floor (furniture, walls, the robot)
    at the start or end. In-place furniture (explicit pose): the start is first moved sideways so the push line
    runs through the centre of mass (edits `task`). on_settled(model, data, info) is called at the end of the
    settle, i.e. at the eval's episode start (preview renders). Returns (ok, stats)."""
    from humanoid_datagen.evaluators.grscenes import place_robot, settle_held, tilt_deg
    model, _, _, info = compose_task(task, tk)
    data = mujoco.MjData(model)
    ob = model.body(info["object_body"]).id
    sub = {ob} | {i for i in range(model.nbody) if _is_desc(model, i, ob)}
    mujoco.mj_forward(model, data)
    if task.get("placement") == "in_place":
        u = np.array(info["direction"])
        n = np.array([-u[1], u[0]])
        st = np.array(task["start"], float)
        st = st + n * float(n @ (data.subtree_com[ob][:2] - st))
        task["start"] = [float(st[0]), float(st[1])]
    q0, qa, da, qadr, dadr = place_robot(model, data, task)

    def touching():
        out = set()
        for c in data.contact[:data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if (b1 in sub) == (b2 in sub):
                continue
            if abs(c.frame[2]) > 0.9 and c.pos[2] < 0.03:      # floor: horizontal contact at floor level
                continue
            out.add(model.body(b2 if b1 in sub else b1).name)
        return sorted(out)

    t0 = touching()
    p0, q0o = data.xpos[ob].copy(), data.xquat[ob].copy()
    settle_held(model, data, q0, qa, da, qadr, dadr, float(sc["settle_s"]))
    p1, q1 = data.xpos[ob].copy(), data.xquat[ob].copy()
    if on_settled is not None:
        on_settled(model, data, info)
    settle_held(model, data, q0, qa, da, qadr, dadr, float(sc["hold_s"]))
    p2, q2 = data.xpos[ob].copy(), data.xquat[ob].copy()
    t1 = touching()
    stats = dict(settle_tilt_deg=round(float(tilt_deg(q1, q0o)), 2), settle_move=round(float(np.linalg.norm(p1 - p0)), 4),
                 drift_m=round(float(np.linalg.norm(p2 - p1)), 4), drift_tilt_deg=round(float(tilt_deg(q2, q1)), 2),
                 rest_tilt_abs_deg=round(float(tilt_deg(q2)), 2), touching=sorted(set(t0) | set(t1)))
    why = []
    if stats["settle_tilt_deg"] > float(sc["max_settle_tilt_deg"]):
        why.append("tips while settling")
    if stats["drift_m"] > float(sc["max_drift_m"]) or stats["drift_tilt_deg"] > float(sc["max_drift_tilt_deg"]):
        why.append("still moving after the settle")
    if stats["touching"]:
        why.append("touches " + ",".join(stats["touching"][:3]))
    stats["reject"] = "; ".join(why)
    return not why, stats


def _checked(task: dict, cfg: dict, log: list, tk) -> bool:
    """settle_check when tasks.settle_check is configured (records its stats in the task); always True otherwise."""
    sc = cfg["tasks"].get("settle_check")
    if not sc:
        return True
    try:
        ok, st = settle_check(task, sc, tk)
    except Exception as ex:  # noqa: BLE001  (an object that does not compile or load is not a task)
        ok, st = False, dict(reject=f"compose failed: {str(ex)[:80]}")
    task["settle_check"] = st
    if not ok:
        log.append(f"{task['type']} ({task['source']}, {task.get('placement', '-')}): {st['reject']}")
    return ok


# ------------------------------------------------------------------------------------------------ GRScenes furniture
def _yaw_only(q, tol=0.02):
    return abs(q[1]) < tol and abs(q[2]) < tol


def body_aabb(m: mujoco.MjModel, d: mujoco.MjData, bid: int, frame: str = "world"):
    """AABB (lo, hi) of a body's colliding geoms (incl. descendants) in the world or the body's own frame, or None."""
    sub = {bid} | {i for i in range(m.nbody) if _is_desc(m, i, bid)}
    Rb, pb = d.xmat[bid].reshape(3, 3), d.xpos[bid]
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in sub or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c, R, a = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_aabb[g]
        for s in np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]):
            w = c + R @ (a[:3] + a[3:] * s)
            pts.append(w if frame == "world" else Rb.T @ (w - pb))
    if not pts:
        return None
    pts = np.array(pts)
    return pts.min(0), pts.max(0)


def _is_desc(m, i, b):
    while i > 0:
        i = m.body_parentid[i]
        if i == b:
            return True
    return False


def furniture_pieces(meta: dict, m: mujoco.MjModel, d: mujoco.MjData, ranges: dict) -> list[dict]:
    """The home's floor-standing pieces of a test type: id, body, type, world AABB, body-frame size."""
    out = []
    for oid, o in sorted(meta["objects"].items()):
        if o.get("dynamic") or o["category"] not in GRSCENES_CATEGORIES or len(o.get("bodies") or []) != 1:
            continue
        try:
            bid = m.body(o["root_body"]).id
        except KeyError:
            continue
        ab = body_aabb(m, d, bid)
        if ab is None or not _yaw_only(m.body_quat[bid]):
            continue
        lo, hi = ab
        D, W, H = (hi - lo).tolist()
        if lo[2] > 0.06 or not (LIMITS["min_foot"] <= max(D, W) <= LIMITS["max_foot"]) or not (LIMITS["min_h"] <= H <= LIMITS["max_h"]):
            continue
        typ = _gr_type(o["category"], D, W, H)
        if typ is None or typ not in ranges:
            continue
        blo, bhi = body_aabb(m, d, bid, frame="body")
        out.append(dict(id=oid, body=o["root_body"], bid=bid, category=o["category"], type=typ, lo=lo, hi=hi,
                        size=np.round(bhi - blo, 3).tolist()))
    return out


def furniture_tasks(scene: str, room: str, meta: dict, m: mujoco.MjModel, d: mujoco.MjData, rng, cfg: dict, tk,
                    ranges: dict, log: list | None = None) -> list[dict]:
    """GRScenes pieces, pushed in place when the home leaves room for it (rare: real furniture stands against
    walls and at tables), otherwise relocated by the independent sampler to free floor of the same home
    (placement "relocated"; the mesh, not the spot, is what makes the object independent of every arm)."""
    from humanoid_datagen.datagen_pipelines.shared.sampler import _floor_ok
    from humanoid_datagen.tasks.push.spots import corridor_ok, sample_tasks
    tc = cfg["tasks"]
    log = [] if log is None else log
    tries = int((tc.get("settle_check") or {}).get("tries", 1))
    pieces = furniture_pieces(meta, m, d, ranges)
    rng.shuffle(pieces)
    out, n_type = [], {}
    for p in pieces:
        if len(out) >= int(tc["furniture_per_scene"]):
            break
        typ, bid = p["type"], p["bid"]
        if n_type.get(typ, 0) >= int(tc.get("furniture_max_per_type", 2)):   # homes hold a dozen dining chairs
            continue
        c = (p["lo"] + p["hi"]) / 2
        hx, hy = (p["hi"] - p["lo"])[:2] / 2
        q = m.body_quat[bid]
        yaw0 = 2 * np.arctan2(q[3], q[0])
        base = dict(scene=scene, room=room, source="grscenes", type=typ, in_training_types=typ in TRAINING_TYPES,
                    object_body=p["body"], instruction=tk.instruction(typ))
        obj = dict(name=typ, grscenes_id=p["id"], category=p["category"], size=p["size"])
        placed = False
        yaws = [yaw0 + k * np.pi / 2 for k in range(4)] + [yaw0 + np.pi / 4 + k * np.pi / 6 for k in range(12)]
        rng.shuffle(yaws)
        for yaw in yaws:                                        # in place: a free run behind and beyond the piece
            u = np.array([np.cos(yaw), np.sin(yaw)])
            half_u = hx * abs(u[0]) + hy * abs(u[1])
            half_n = hx * abs(u[1]) + hy * abs(u[0])
            gap = float(rng.uniform(*tc["gap"]))
            start = c[:2] - u * (half_u + gap)
            ahead = gap + 2 * half_u + float(tc["furniture_push_len"]) + 0.3
            if _floor_ok(m, d, start, 0.04) and corridor_ok(m, d, start, yaw, ahead, 0.2, max(half_n + 0.05, 0.3),
                                                           exclude_body=bid):
                t = dict(base, placement="in_place", object=dict(obj, **draw_physics(ranges, typ, rng)),
                         object_pose=[float(m.body_pos[bid][0]), float(m.body_pos[bid][1]), float(np.degrees(yaw0))],
                         start=[float(start[0]), float(start[1])], yaw_deg=float(np.degrees(yaw)),
                         object_distance=gap)
                if _checked(t, cfg, log, tk):   # in place it may touch the table it stands at: relocate instead
                    out.append(t)
                    placed = True
                break
        if not placed and tc.get("furniture_relocate", True):
            for _ in range(tries):
                ts = sample_tasks(m, 1, rng, p["size"], float(tc["corridor_push_len"]), gap=tuple(tc["gap"]))
                if not ts:
                    break
                t = dict(ts[0], **base, placement="relocated", object=dict(obj, **draw_physics(ranges, typ, rng)),
                         object_pose="auto")
                if _checked(t, cfg, log, tk):
                    out.append(t)
                    placed = True
                    break
                if t["settle_check"].get("reject", "").startswith(("tips", "compose")):
                    break                    # the piece itself is unstable: another spot will not help
        if placed:
            n_type[typ] = n_type.get(typ, 0) + 1
    return out


# ------------------------------------------------------------------------------------------------ held-out library instances
def _split(C: dict) -> dict:
    p = repo_path(C["held_out"]["split_file"])
    return json.load(open(p)) if os.path.exists(p) else {}


def _export_asset(uid: str, work: str, C: dict, quat=None, asset_xml: str | None = None) -> dict:
    """Library asset `uid` exported to work/assets (or a built asset's own `asset_xml`, push_box/test_suitcases):
    {asset (repo-relative xml), size (D, W, H of its colliders, after `quat`)}."""
    if asset_xml is not None:
        m = mujoco.MjModel.from_xml_path(repo_path(asset_xml))
        dd = mujoco.MjData(m)
        mujoco.mj_forward(m, dd)
        lo, hi = body_aabb(m, dd, m.body(uid).id)
        return dict(asset=asset_xml, size=np.round(hi - lo, 3).tolist())
    library = _library(C)
    d = os.path.join(work, "assets")
    os.makedirs(d, exist_ok=True)
    if not os.path.isfile(os.path.join(d, uid, "asset.xml")):
        try:
            library.export(uid, uid, root=d)
        except Exception as ex:  # noqa: BLE001  (degenerate collider meshes etc.)
            print(f"[tasks] held-out asset {uid} cannot be exported ({str(ex)[:80]}): remove it from the split")
            return None
    m = mujoco.MjSpec.from_file(os.path.join(d, uid, "asset.xml")).compile()
    dd = mujoco.MjData(m)
    mujoco.mj_forward(m, dd)
    lo, hi = body_aabb(m, dd, m.body(uid).id)
    if quat is not None:                     # size of the rotated asset (90-degree turns: exact)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat, float))
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ R.reshape(3, 3).T
        lo, hi = c.min(0), c.max(0)
    return dict(asset=os.path.relpath(os.path.join(d, uid, "asset.xml"), repo_path(".")),
                size=np.round(hi - lo, 3).tolist())


def library_tasks(scene: str, room: str, m: mujoco.MjModel, rng, cfg: dict, tk, ranges: dict, start_k: int,
                  work: str, log: list | None = None) -> list[dict]:
    """Per slot (types cycled) up to settle_check.tries draws of an asset of that type + a spot, until one passes."""
    from humanoid_datagen.tasks.push.spots import sample_tasks
    tc = cfg["tasks"]
    log = [] if log is None else log
    tries = int((tc.get("settle_check") or {}).get("tries", 1))
    held = _split(cfg).get("held_out", {})
    types = [t for t in sorted(held) if held[t] and t in ranges]
    out = []
    if not types:
        print(f"[tasks] no held-out library split ({cfg['held_out']['split_file']})")
        return out
    for i in range(int(tc["library_per_scene"])):
        typ = types[(start_k + i) % len(types)]
        for _ in range(tries):
            a = held[typ][int(rng.integers(len(held[typ])))]
            ex = _export_asset(a["uid"], work, cfg, a.get("quat"), a.get("asset_xml"))
            if ex is None:
                continue
            ts = sample_tasks(m, 1, rng, ex["size"], float(tc["corridor_push_len"]), gap=tuple(tc["gap"]))
            if not ts:
                continue
            phys = draw_physics(ranges, typ, rng)
            t = dict(ts[0], scene=scene, room=room, source="library", type=typ, in_training_types=True,
                     object_body="push_object", object_pose="auto",
                     object=dict(name=typ, uid=a["uid"], category=a.get("category"), asset=ex["asset"],
                                 size=ex["size"], **({"quat": a["quat"]} if a.get("quat") else {}), **phys),
                     instruction=tk.instruction(typ))
            if _checked(t, cfg, log, tk):
                out.append(t)
                break
    return out


# ------------------------------------------------------------------------------------------------ task list
def _home_tasks(e: dict, C: dict, tk, comp: dict, seed: int, start_k: int, work: str) -> tuple[str, list[dict]]:
    """Compose one GRScenes home (cached in work/<name>/room.xml) and draw its furniture + held-out library tasks.
    Returns (log line, tasks); tasks is empty when the home is skipped."""
    from humanoid_datagen.utils.scenes import composer, spec as S
    tc = C["tasks"]
    ranges = physics_ranges(C)
    rng = np.random.default_rng(seed)
    room = os.path.join(work, e["name"], "room.xml")
    meta_path = os.path.join(os.path.dirname(room), "room_meta.json")
    failed = os.path.join(os.path.dirname(room), "room_failed.txt")
    if os.path.exists(failed):
        return f"[tasks] {e['name']}: room does not compile ({open(failed).read()[:60]}): home skipped", []
    if not os.path.exists(meta_path):   # no room yet, or a room.xml left behind by a compose that was killed
        os.makedirs(os.path.dirname(room), exist_ok=True)
        try:
            composer.compose(S.load_spec(repo_path(e["spec"])), comp, room, robot=False)
        except ValueError as ex:   # degenerate mesh in the converted home: remember, so the next draw skips it at once
            open(failed, "w").write(str(ex))
            return f"[tasks] {e['name']}: room does not compile ({str(ex)[:60]}): home skipped", []
    meta = json.load(open(meta_path))
    try:
        m = mujoco.MjModel.from_xml_path(room)
    except ValueError as ex:   # e.g. a degenerate mesh in the converted home ("at least 4 vertices required")
        return f"[tasks] {e['name']}: room does not load ({str(ex)[:60]}): home skipped", []
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rel = os.path.relpath(room, repo_path("."))
    log = []
    ts = furniture_tasks(e["name"], rel, meta, m, d, rng, C, tk, ranges, log)[: int(tc["furniture_per_scene"])]
    ts += library_tasks(e["name"], rel, m, rng, C, tk, ranges, start_k, work, log)
    for k, t in enumerate(ts):
        t["task_id"] = f"{e['name']}__{k:03d}"
    msg = (f"[tasks] {e['name']}: {sum(t['source'] == 'grscenes' for t in ts)} furniture "
           f"({', '.join(sorted({t['type'] for t in ts if t['source'] == 'grscenes'}))}), "
           f"{sum(t['source'] == 'library' for t in ts)} held-out library"
           + (f"; settle check rejected {len(log)}: " + " | ".join(log) if log else ""))
    return msg, ts


def _home_tasks_child(q, *args):
    q.put(_home_tasks(*args))


def build_tasks(C: dict, tk) -> list[dict]:
    """One child process per home: a converted GRScenes home peaks at ~10 GB while composing and about 1 GB per home
    stays resident afterwards, so a single process drawing 69 homes gets killed on the login node.
    tasks.workers homes run at once (each child ~8-10 GB: 1 on the login node, ~6 on a 64 GB node). Homes are
    independent: home i's seed is seed * 1000 + i and its library types start at i * library_per_scene, so the draw
    does not depend on the order homes finish in (v1 / v2 started at the running task count instead)."""
    import multiprocessing as mp
    import yaml
    tc = C["tasks"]
    comp = C["composer"]
    entries = yaml.safe_load(open(repo_path(C["scene_index"])))[: C.get("max_scenes") or None]
    work = repo_path(C["work_dir"])
    ctx = mp.get_context("spawn")
    workers = max(1, int(tc.get("workers", 1)))
    per = int(tc["library_per_scene"])
    results: dict[int, tuple[str, list[dict]]] = {}
    pending = list(range(len(entries)))      # home indices still to start
    attempts = {i: 0 for i in pending}
    running: dict[int, tuple] = {}           # index -> (process, queue)

    def start(i):
        q = ctx.Queue()
        p = ctx.Process(target=_home_tasks_child, args=(q, entries[i], C, tk, comp, int(tc["seed"]) * 1000 + i, i * per, work))
        p.start()
        attempts[i] += 1
        running[i] = (p, q)

    while pending or running:
        while pending and len(running) < workers:
            start(pending.pop(0))
        time.sleep(1)
        for i, (p, q) in list(running.items()):
            try:
                results[i] = q.get_nowait()
            except queue.Empty:
                if p.is_alive():
                    continue
                try:                         # finished between the two checks
                    results[i] = q.get(timeout=2)
                except queue.Empty:
                    pass
            p.join(timeout=30)
            del running[i]
            name = entries[i]["name"]
            if i not in results:
                if p.exitcode == -9 and attempts[i] < 3:   # the login node SIGKILLs a child now and then
                    print(f"[tasks] {name}: child killed (SIGKILL), retry {attempts[i]}", flush=True)
                    pending.append(i)
                    continue
                results[i] = (f"[tasks] {name}: child process died (exit {p.exitcode}): home skipped", [])
            print(results[i][0], flush=True)
    out = []
    for i in range(len(entries)):
        out += results[i][1]
    return out

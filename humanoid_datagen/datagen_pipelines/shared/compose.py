"""Agent output contract -> simulation: room MJCF (any content) + task.json -> G1 + room + staged push object.

Contract (written by the push-scene agent in its run dir):
  room.xml   any MuJoCo MJCF without a robot: floor, walls, furniture, the agent's own meshes / textures /
             cameras / lights. Exactly one body named by task.object_body (default "push_object") with a
             <freejoint/>; its geoms carry the object's mass (or density) and friction.
  task.json  {"motion": "motions/m0.csv",           Kimodo CSV (30 fps, root xyz, root wxyz, 29 joints)
              "start": [x, y],                      frame-0 root position in the room  (older files: robot_start)
              "yaw_deg": 0,                         task direction in the room; the motion is rotated to it
                                                    (older files: push_yaw_deg)
              "object_body": "push_object",
              "object_pose": "auto" | [x, y, yaw_deg],   auto = stage the object under the palms
              "instruction": "Push the cardboard box to the window"}

compose() attaches the G1 (humanoid_datagen/assets/robots/g1/mjcf/g1_29dof_v2_thor.xml, OAK-1 head camera "oak1") at the world origin,
rotates/translates the motion, stages the object, and applies the physics conventions the tracker needs:
200 Hz physics, elliptic cones; robot geoms collide with the world and the object but not with themselves;
object geoms get priority 1 so the object's own friction is used against the floor.
"""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np

from humanoid_datagen.utils.paths import repo_path

# the robot: official G1 MJCF with the calibrated OAK-1 head-camera mount
G1_XML = repo_path("humanoid_datagen/assets/robots/g1/g1_29dof_official_oak1_mount.xml")
from humanoid_datagen.utils.sonic import Reference, set_robot_pose

ROBOT_CONTYPE, ROBOT_CONAFF = 2, 1


def _robot_spec():
    """G1 thor spec with only what the robot itself uses: its textures (sky, ground) and the ground material are
    dropped, its materials and meshes are renamed g1_* so they never clash with the agent's room assets."""
    s = mujoco.MjSpec.from_file(G1_XML)
    s.meshdir = os.path.normpath(os.path.join(os.path.dirname(G1_XML), s.meshdir))
    for t in list(s.textures):
        s.delete(t)
    for m in list(s.materials):
        if m.name == "groundplane":
            s.delete(m)
    ren = {}
    for m in s.materials:
        ren[m.name] = "g1_" + m.name
        m.name = ren[m.name]
    mren = {}
    for m in s.meshes:
        mren[m.name] = "g1_" + m.name
        m.name = mren[m.name]
    for g in s.geoms:
        if g.material in ren:
            g.material = ren[g.material]
        if g.meshname in mren:
            g.meshname = mren[g.meshname]
    return s


_ROBOT_ONLY = None


def robot_only_model() -> mujoco.MjModel:
    """The same G1 as attach_robot (official + OAK-1 mount) alone on a floor plane: staging FK / motion analysis."""
    global _ROBOT_ONLY
    if _ROBOT_ONLY is None:
        sp = mujoco.MjSpec()
        sp.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1])
        sp.option.timestep = 0.005
        rs = _robot_spec()
        for g in rs.geoms:
            if g.contype or g.conaffinity:
                g.contype, g.conaffinity = ROBOT_CONTYPE, ROBOT_CONAFF
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.worldbody.add_frame().attach_body(rs.body("pelvis"), "", "")
            _ROBOT_ONLY = sp.compile()
    return _ROBOT_ONLY


def robot_floor_model(object_body: str | None = None) -> mujoco.MjModel:
    """The same G1 on an open floor: tracking a motion without any object or room in the way (the motions tool,
    motion selection). object_body: also add a small free box of that name far away."""
    sp = mujoco.MjSpec()
    sp.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=[0.55, 0.55, 0.58], rgb2=[0.42, 0.42, 0.45], width=512, height=512)
    sp.add_material(name="grid", textures=["", "grid"], texrepeat=[100, 100])
    sp.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1], material="grid")
    sp.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.8, 0.8, 0.8], castshadow=False)
    sp.visual.headlight.ambient = [0.4, 0.4, 0.4]
    sp.option.timestep = 0.005
    if object_body is not None:
        b = sp.worldbody.add_body(name=object_body, pos=[40, 40, 0.05])
        b.add_freejoint()
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05], mass=1.0)
    rs = _robot_spec()
    for g in rs.geoms:
        if g.contype or g.conaffinity:
            g.contype, g.conaffinity = ROBOT_CONTYPE, ROBOT_CONAFF
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp.worldbody.add_frame().attach_body(rs.body("pelvis"), "", "")
        return sp.compile()


def _fib_sphere(n: int = 256) -> np.ndarray:
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r, a = np.sqrt(1 - z * z), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(a), r * np.sin(a), z], 1)


def geom_points(m: mujoco.MjModel, g: int) -> np.ndarray | None:
    """Points on geom g's collision surface in the geom frame: mesh vertices (MuJoCo stores them in the geom frame),
    box corners, dense samples of round primitives. None for planes / height fields / SDFs."""
    T, t, s = mujoco.mjtGeom, m.geom_type[g], m.geom_size[g]
    if t == T.mjGEOM_MESH:
        k = m.geom_dataid[g]
        return m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]].astype(float)
    if t == T.mjGEOM_BOX:
        return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * s
    if t == T.mjGEOM_CYLINDER:
        a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        rim = np.stack([np.cos(a) * s[0], np.sin(a) * s[0]], 1)
        return np.concatenate([np.c_[rim, np.full(64, z)] for z in (-s[1], s[1])])
    u = _fib_sphere()
    if t == T.mjGEOM_SPHERE:
        return u * s[0]
    if t == T.mjGEOM_ELLIPSOID:
        return u * s[:3]
    if t == T.mjGEOM_CAPSULE:
        p = u * s[0]
        p[:, 2] += np.where(p[:, 2] >= 0, s[1], -s[1])
        return p
    return None


def model_extent(m: mujoco.MjModel, body: str, exact: bool = False) -> dict:
    """Body-frame AABB of the object's colliding geoms (half extents, centre offset, lowest point), its mass and its
    centre of mass (body frame). Default: from the corners of each geom's own bounding box (the training datagen's
    convention, kept so its scenes compose as generated). exact=True: from the colliders' true surface points
    (geom_points), so rotated / round colliders do not inflate the box or lift the object off the floor (eval test
    objects, docs/eval_push_issues.md section 2)."""
    d = mujoco.MjData(m)
    b = m.body(body).id
    jnt = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not jnt:
        raise ValueError(f"body {body!r} needs a <freejoint/>")
    mujoco.mj_forward(m, d)
    R, p = d.xmat[b].reshape(3, 3), d.xpos[b]
    sub = {b} | {i for i in range(m.nbody) if _is_desc(m, i, b)}
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in sub or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c = d.geom_xpos[g]
        Rg = d.geom_xmat[g].reshape(3, 3)
        loc = geom_points(m, g) if exact else None
        if loc is None:
            a = m.geom_aabb[g]  # local centre (3) + half sizes (3) in geom frame
            loc = a[:3] + a[3:] * corners
        pts.append((c + loc @ Rg.T - p) @ R)
    if not pts:
        raise ValueError(f"body {body!r} has no colliding geoms")
    pts = np.concatenate(pts)
    lo, hi = pts.min(0), pts.max(0)
    return dict(half=((hi - lo) / 2).tolist(), center=((hi + lo) / 2).tolist(), z_low=float(lo[2]),
                mass=float(m.body_subtreemass[b]), com=(R.T @ (d.subtree_com[b] - p)).tolist(), exact=exact)


def _is_desc(m, i, b):
    while i > 0:
        if i == b:
            return True
        i = m.body_parentid[i]
    return False


def object_extent(room_xml: str, body: str) -> dict:
    return model_extent(mujoco.MjModel.from_xml_path(room_xml), body)


def room_spec(room_xml: str) -> mujoco.MjSpec:
    """Load a room MJCF."""
    return mujoco.MjSpec.from_file(room_xml)


def attach_robot(spec: mujoco.MjSpec, obj: str, pose, ext: dict, timestep: float = 0.005) -> mujoco.MjModel:
    """Place the object, apply the physics conventions, attach the G1, compile."""
    spec.option.timestep = timestep
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 1280)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 800)
    ob = spec.body(obj)
    ob.pos = [pose[0], pose[1], -ext["z_low"] + 0.002]
    ob.quat = [np.cos(np.radians(pose[2]) / 2), 0, 0, np.sin(np.radians(pose[2]) / 2)]
    for g in spec.geoms:
        if g.contype or g.conaffinity:
            g.conaffinity = g.conaffinity | ROBOT_CONTYPE  # everything solid touches the robot
            g.contype = g.contype | 1

    def prio(b):  # the object's own friction wins against the floor (MuJoCo otherwise takes the max)
        for g in b.geoms:
            if (g.contype or g.conaffinity) and g.priority == 0:
                g.priority = 1
        for ch in b.bodies:
            prio(ch)
    prio(ob)

    rs = _robot_spec()
    for g in rs.geoms:
        if g.contype or g.conaffinity:
            g.contype, g.conaffinity = ROBOT_CONTYPE, ROBOT_CONAFF
    frame = spec.worldbody.add_frame()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # empty attach prefix: joint names must stay the G1's own
        frame.attach_body(rs.body("pelvis"), "", "")
        return spec.compile()


def episode_frame(ep: dict) -> tuple[list[float], float]:
    """An episode's robot start [x, y] and direction (deg): `start` / `yaw_deg`, or the older names robot_start /
    push_yaw_deg (agent task.json files, the v3 eval task list, earlier spots.json)."""
    start = ep["start"] if "start" in ep else ep["robot_start"]
    return [float(v) for v in start], float(ep.get("yaw_deg", ep.get("push_yaw_deg", 0.0)))


def compose_room(room_xml: str, episode: dict, task, base_dir: str = ".", ref0: Reference | None = None,
                 timestep: float = 0.005, prepare=None, exact_object: bool = False):
    """Room MJCF + episode (start, yaw_deg, object_pose, ...) -> (model, spec, ref, info); info has start, yaw_deg,
    direction (unit 2-d), the object's body, extent and pose, and the task's staging info and checks. `task` (a
    tasks.base.Task) places the object against the staged motion (task.place_object), or without a motion
    (task.object_pose), and checks the scene (task.scene_checks). `prepare(spec)` may edit the room spec before the
    object is measured (test set: make a piece of the room's own furniture free-moving). exact_object (eval test
    objects): measure the object from its true collider surface and place it by its centre of mass."""
    obj = episode.get("object_body", "push_object")
    spec = room_spec(room_xml)
    if prepare is not None:
        prepare(spec)
    ext = model_extent(spec.compile(), obj, exact=exact_object)
    if ref0 is None and episode.get("motion"):
        q = np.loadtxt(os.path.join(base_dir, episode["motion"]), delimiter=",")
        ref0 = Reference.from_qpos(q, episode.get("motion_fps", 30.0))
    start, yaw = episode_frame(episode)
    staged, ref = None, None
    if ref0 is not None:
        ref, pose, staged = task.place_object(ref0, ext, start, yaw)
    op = episode.get("object_pose", "auto")
    if op == "auto" and ref0 is None:
        op = task.object_pose(ext, start, yaw, float(episode.get("object_distance", 0.6)),
                              float(episode.get("object_lateral", 0.0)), exact=exact_object)
    if op != "auto":
        pose = [float(v) for v in op]
    model = attach_robot(spec, obj, pose, ext, timestep)
    d = [float(np.cos(np.radians(yaw))), float(np.sin(np.radians(yaw)))]
    info = dict(object_body=obj, object_extent=ext, object_pose=pose, direction=d, start=start, yaw_deg=yaw)
    if staged is not None:
        info.update(staged)
        info["checks"] = task.scene_checks(model, ref, info)
    return model, spec, ref, info


def compose(run_dir: str, episode: dict | str = "task.json", room: str = "room.xml", timestep: float = 0.005, task=None):
    """Agent contract: <run_dir>/room.xml + task.json. Returns (model, spec, ref, info); ref = 50 Hz reference in
    room coordinates. `task`: default, the task in $HDG_TASK (tasks/load.py; the agent batch driver sets it)."""
    if isinstance(episode, str):
        episode = json.load(open(os.path.join(run_dir, episode)))
    if task is None:
        from humanoid_datagen.tasks.load import task_from_env
        task = task_from_env()
    return compose_room(os.path.join(run_dir, room), episode, task, run_dir, timestep=timestep)


def start_pose_check(model, ref: Reference, info: dict, allow_object_m: float = 0.03) -> dict:
    """The robot's start pose (frame 0 of `ref`) against the room: penetrations deeper than 1 cm, ignoring the floor
    and up to `allow_object_m` into the object (motions that start with the hands on it)."""
    d = mujoco.MjData(model)
    set_robot_pose(model, d, ref, 0)
    rob = model.geom_contype == ROBOT_CONTYPE
    pen = []
    obj_b = model.body(info["object_body"]).id
    for i in range(d.ncon):
        c = d.contact[i]
        if (rob[c.geom1] ^ rob[c.geom2]) and c.dist < -0.01:
            other = c.geom2 if rob[c.geom1] else c.geom1
            if _is_desc(model, int(model.geom_bodyid[other]), obj_b) and c.dist > -allow_object_m:
                continue
            pen.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom{other}")
    floor_like = [g for g in pen if "floor" in g.lower()]
    out = dict(start_pose_penetrations=sorted(set(pen) - set(floor_like)))
    out["start_pose_free"] = not out["start_pose_penetrations"]
    return out

"""Agent output contract -> simulation: room MJCF (any content) + task.json -> G1 + room + staged push object.

Contract (written by the push-scene agent in its run dir):
  room.xml   any MuJoCo MJCF without a robot: floor, walls, furniture, the agent's own meshes / textures /
             cameras / lights. Exactly one body named by task.object_body (default "push_object") with a
             <freejoint/>; its geoms carry the object's mass (or density) and friction.
  task.json  {"motion": "motions/m0.csv",           Kimodo CSV (30 fps, root xyz, root wxyz, 29 joints)
              "robot_start": [x, y],                frame-0 root position in the room
              "push_yaw_deg": 0,                    push direction in the room (the motion is rotated to it)
              "object_body": "push_object",
              "object_pose": "auto" | [x, y, yaw_deg],   auto = stage the object under the palms
              "instruction": "Push the cardboard box to the window"}

compose() attaches the G1 (robots/g1/mjcf/g1_29dof_v2_thor.xml, OAK-1 head camera "oak1") at the world origin,
rotates/translates the motion, stages the object, and applies the physics conventions the tracker needs:
200 Hz physics, elliptic cones; robot geoms collide with the world and the object but not with themselves;
object geoms get priority 1 so the object's own friction is used against the floor.
"""
from __future__ import annotations

import json
import os

import mujoco
import numpy as np


# the robot: official G1 MJCF with the calibrated OAK-1 head-camera mount
G1_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs", "assets", "g1",
                      "g1_29dof_official_oak1_mount.xml")
from .sonic import Reference, set_robot_pose
from .stage import stage

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


def robot_floor_model(object_body: str = "box") -> mujoco.MjModel:
    """The same G1 on an open floor with a small free box far away (named `object_body`): tracking a motion without
    any object or room in the way (the motions tool, motion selection)."""
    sp = mujoco.MjSpec()
    sp.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                   rgb1=[0.55, 0.55, 0.58], rgb2=[0.42, 0.42, 0.45], width=512, height=512)
    sp.add_material(name="grid", textures=["", "grid"], texrepeat=[100, 100])
    sp.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[50, 50, 0.1], material="grid")
    sp.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.8, 0.8, 0.8], castshadow=False)
    sp.visual.headlight.ambient = [0.4, 0.4, 0.4]
    sp.option.timestep = 0.005
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


def model_extent(m: mujoco.MjModel, body: str) -> dict:
    """Body-frame AABB of the object's colliding geoms (half extents, centre offset, lowest point) and its mass."""
    d = mujoco.MjData(m)
    b = m.body(body).id
    jnt = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not jnt:
        raise ValueError(f"body {body!r} needs a <freejoint/>")
    mujoco.mj_forward(m, d)
    R, p = d.xmat[b].reshape(3, 3), d.xpos[b]
    sub = {b} | {i for i in range(m.nbody) if _is_desc(m, i, b)}
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in sub or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c = d.geom_xpos[g]
        Rg = d.geom_xmat[g].reshape(3, 3)
        a = m.geom_aabb[g]  # local centre (3) + half sizes (3) in geom frame
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    w = c + Rg @ (a[:3] + a[3:] * np.array([sx, sy, sz]))
                    pts.append(R.T @ (w - p))
    if not pts:
        raise ValueError(f"body {body!r} has no colliding geoms")
    pts = np.array(pts)
    lo, hi = pts.min(0), pts.max(0)
    return dict(half=((hi - lo) / 2).tolist(), center=((hi + lo) / 2).tolist(), z_low=float(lo[2]),
                mass=float(m.body_subtreemass[b]))


def _is_desc(m, i, b):
    while i > 0:
        if i == b:
            return True
        i = m.body_parentid[i]
    return False


def object_extent(room_xml: str, body: str) -> dict:
    return model_extent(mujoco.MjModel.from_xml_path(room_xml), body)


def room_spec(room_xml: str, add_object: dict | None = None, rng=None) -> mujoco.MjSpec:
    """Load a room MJCF; optionally add a procedural push object (objects.add_object card instance)."""
    spec = mujoco.MjSpec.from_file(room_xml)
    if add_object is not None:
        from .objects import add_object as _add
        _add(spec, add_object)
    return spec


def stage_motion(ref0: Reference, ext: dict, start, push_yaw_deg: float):
    """Rotate / translate the motion so its push runs along push_yaw from `start`; object pose with its AABB's near
    face on the palms at the contact frame, centred on the hands. Returns (ref, pose [x, y, yaw_deg], st)."""
    st = stage(robot_only_model(), ref0, ext["half"], friction=0.0, tip_safety=0.0)  # pure kinematics; sizes untouched
    yaw_push = np.radians(push_yaw_deg)
    rot = yaw_push - np.arctan2(st["push_dir"][1], st["push_dir"][0])
    c, s = np.cos(rot), np.sin(rot)
    R2 = np.array([[c, -s], [s, c]])
    start = np.asarray(start, float)
    ref = ref0.transformed(start, rot)
    d = np.array([np.cos(yaw_push), np.sin(yaw_push)])
    hand_xy = R2 @ (np.array(st["box_center"]) - ref0.root_pos[0, :2] - np.array(st["push_dir"]) * st["box_half"][0]) + start
    cxy = hand_xy + d * ext["half"][0]
    Ro = np.array([[c_ := np.cos(yaw_push), -np.sin(yaw_push)], [np.sin(yaw_push), c_]])
    oxy = cxy - Ro @ np.array(ext["center"][:2])
    return ref, [float(oxy[0]), float(oxy[1]), float(push_yaw_deg)], st


def pose_at_distance(ext: dict, start, push_yaw_deg: float, dist: float, lateral: float = 0.0):
    """Object pose with its near face `dist` m ahead of the start pelvis along push_yaw (no motion needed)."""
    y = np.radians(push_yaw_deg)
    d, lat = np.array([np.cos(y), np.sin(y)]), np.array([-np.sin(y), np.cos(y)])
    cxy = np.asarray(start, float) + d * (dist + ext["half"][0]) + lat * lateral
    Ro = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
    oxy = cxy - Ro @ np.array(ext["center"][:2])
    return [float(oxy[0]), float(oxy[1]), float(push_yaw_deg)]


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


def compose_room(room_xml: str, task: dict, base_dir: str = ".", add_object: dict | None = None,
                 ref0: Reference | None = None, timestep: float = 0.005, prepare=None):
    """General form of compose(): returns (model, spec, ref, info). `prepare(spec)` may edit the room spec before
    the object is measured (test set: make a piece of the room's own furniture free-moving, test_objects)."""
    obj = task.get("object_body", "push_object")
    spec = room_spec(room_xml, add_object)
    if prepare is not None:
        prepare(spec)
    ext = model_extent(spec.compile(), obj)
    if ref0 is None and task.get("motion"):
        q = np.loadtxt(os.path.join(base_dir, task["motion"]), delimiter=",")
        ref0 = Reference.from_qpos(q, task.get("motion_fps", 30.0))
    yaw = float(task.get("push_yaw_deg", 0.0))
    st, ref = None, None
    if ref0 is not None:
        ref, pose, st = stage_motion(ref0, ext, task["robot_start"], yaw)
    op = task.get("object_pose", "auto")
    if op == "auto" and ref0 is None:
        op = pose_at_distance(ext, task["robot_start"], yaw, float(task.get("object_distance", 0.6)),
                              float(task.get("object_lateral", 0.0)))
    if op != "auto":
        pose = [float(v) for v in op]
    model = attach_robot(spec, obj, pose, ext, timestep)
    d = [float(np.cos(np.radians(yaw))), float(np.sin(np.radians(yaw)))]
    info = dict(object_body=obj, object_extent=ext, object_pose=pose, push_dir=d,
                robot_start=[float(v) for v in task["robot_start"]], push_yaw_deg=yaw)
    if st is not None:
        info.update(contact_frame=st["frame_contact"], t_contact=st["t_contact"], contact_height=st["contact_height"],
                    kinematic_push_dist=st["kinematic_push_dist"], body_intrusion=st["body_intrusion"])
        info["checks"] = checks(model, ref, info)
    return model, spec, ref, info


def compose(run_dir: str, task: dict | str = "task.json", room: str = "room.xml", timestep: float = 0.005):
    """Agent contract: <run_dir>/room.xml + task.json. Returns (model, spec, ref, info); ref = 50 Hz reference in
    room coordinates."""
    if isinstance(task, str):
        task = json.load(open(os.path.join(run_dir, task)))
    return compose_room(os.path.join(run_dir, room), task, run_dir, timestep=timestep)


def checks(model, ref: Reference, info: dict) -> dict:
    """Static checks: palms below the object's top and above its bottom, robot start pose collision-free,
    robot path inside the floor area, object resting on something."""
    ext = info["object_extent"]
    top = ext["center"][2] + ext["half"][2] - ext["z_low"]
    out = {}
    hz = info["contact_height"]
    out["palms_on_object_face"] = bool(0.05 < hz < top - 0.03)
    out["palm_height"] = round(hz, 3)
    out["object_top"] = round(top, 3)
    d = mujoco.MjData(model)
    set_robot_pose(model, d, ref, 0)
    rob = model.geom_contype == ROBOT_CONTYPE
    pen = []
    obj_b = model.body(info["object_body"]).id
    for i in range(d.ncon):
        c = d.contact[i]
        if (rob[c.geom1] ^ rob[c.geom2]) and c.dist < -0.01:
            other = c.geom2 if rob[c.geom1] else c.geom1
            # palms already on the object (motions whose contact frame is t = 0): fine up to 3 cm
            if _is_desc(model, int(model.geom_bodyid[other]), obj_b) and c.dist > -0.03:
                continue
            pen.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom{other}")
    floor_like = [g for g in pen if "floor" in g.lower()]
    out["start_pose_penetrations"] = sorted(set(pen) - set(floor_like))
    out["start_pose_free"] = not out["start_pose_penetrations"]
    out["body_intrusion_m"] = round(info["body_intrusion"], 3)
    out["ok"] = bool(out["palms_on_object_face"] and out["start_pose_free"])
    return out

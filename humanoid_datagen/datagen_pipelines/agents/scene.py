"""Push-box MuJoCo scene: room + static furniture + one free-body box + the G1 (thor MJCF, oak1 head camera).

The scene is described by a `layout.json` (written by the scene agent, see docs/push_box.md):

    {
      "room":  {"size": [Lx, Ly], "height": 2.7, "floor_rgb": [...], "wall_rgb": [...], "floor_friction": 1.0},
      "furniture": [{"name": "sofa", "type": "box", "size": [hx, hy, hz], "pos": [x, y], "yaw": 0.0,
                     "rgb": [...]}, ...],          # static; z is set so the piece stands on the floor
      "box":   {"size": [hx, hy, hz], "mass": 3.0, "friction": 0.35, "rgb": [...]},
      "third_camera": {"pos": [x, y, z], "lookat": [x, y, z], "fovy": 60}
    }

The robot and box poses come from staging (`stage.py`) and are passed to `build`.
The room spans [-Lx/2, Lx/2] x [-Ly/2, Ly/2]. Sizes are half-extents (MuJoCo convention).

Collision filtering: robot geoms contype=2/conaffinity=1, world + box contype=1/conaffinity=3,
so the robot touches the world and the box but not itself (as in SONIC's training sim).
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # humanoid_datagen/
G1_XML = os.path.join(ROOT, "assets", "robots", "g1", "mjcf", "g1_29dof_v2_thor.xml")


def _quat_z(yaw):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def robot_model(timestep: float = 0.005) -> mujoco.MjModel:
    """The G1 alone on a plane, with the same collision filtering as `build` (used for staging/FK)."""
    layout = {"room": {"size": [20.0, 20.0]}, "box": {"size": [0.1, 0.1, 0.1], "mass": 1.0}}
    return build(layout, (9.0, 9.0, 0.0), timestep)


def build(layout: dict, box_pose: tuple[float, float, float], timestep: float = 0.005, return_spec: bool = False):
    """box_pose = (x, y, yaw) of the box centre on the floor. Returns the compiled model.

    The robot base pose is set at run time from the staged reference (qpos), not here.
    """
    spec = mujoco.MjSpec.from_file(G1_XML)
    spec.meshdir = os.path.join(os.path.dirname(G1_XML), "..", "meshes")  # absolute, so saved XMLs load anywhere
    spec.option.timestep = timestep
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    pelvis = spec.body("pelvis")
    robot_geoms = set()

    def collect(b):
        for g in b.geoms:
            robot_geoms.add(g.name or id(g))
        for c in b.bodies:
            collect(c)
    collect(pelvis)
    for g in spec.geoms:
        key = g.name or id(g)
        if key in robot_geoms:
            if g.contype or g.conaffinity:
                g.contype, g.conaffinity = 2, 1
        elif g.contype or g.conaffinity:
            g.contype, g.conaffinity = 1, 3

    room = layout["room"]
    Lx, Ly = room["size"]
    H = room.get("height", 2.7)
    wb = spec.worldbody
    floor = spec.geom("floor")
    floor.contype, floor.conaffinity = 1, 3
    floor.friction = [room.get("floor_friction", 1.0), 0.005, 0.0001]
    mat = spec.add_material(name="floor_mat", rgba=list(room.get("floor_rgb", [0.72, 0.55, 0.36])) + [1])
    floor.material = "floor_mat"
    floor.size = [Lx / 2, Ly / 2, 0.05]
    wall_rgba = list(room.get("wall_rgb", [0.93, 0.93, 0.91])) + [1]
    t = 0.05
    for name, pos, size in [("wall_n", [0, Ly / 2 + t, H / 2], [Lx / 2 + 2 * t, t, H / 2]),
                            ("wall_s", [0, -Ly / 2 - t, H / 2], [Lx / 2 + 2 * t, t, H / 2]),
                            ("wall_e", [Lx / 2 + t, 0, H / 2], [t, Ly / 2, H / 2]),
                            ("wall_w", [-Lx / 2 - t, 0, H / 2], [t, Ly / 2, H / 2])]:
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=wall_rgba,
                    contype=1, conaffinity=3)

    # ceiling (visual + collision-free) so egocentric views never see the skybox
    wb.add_geom(name="ceiling", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, H + t], size=[Lx / 2 + 2 * t, Ly / 2 + 2 * t, t],
                rgba=list(room.get("ceiling_rgb", [0.97, 0.97, 0.96])) + [1], contype=0, conaffinity=0)

    # furniture: type box | cylinder | ellipsoid | sphere; size = half extents (cylinder: [r, r, half height]);
    # pos = [x, y] (stands on the floor) or [x, y, z] (centre); collide=false for wall art / foliage / windows
    gtypes = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
              "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}
    for f in layout.get("furniture", []):
        hx, hy, hz = f["size"]
        x, y = f["pos"][:2]
        z = f["pos"][2] if len(f["pos"]) > 2 else hz
        rgba = list(f.get("rgb", [0.6, 0.6, 0.6])) + [1]
        ftype = f.get("type", "box")
        size = [hx, hz, 0] if ftype == "cylinder" else [hx, hy, hz]
        c = 1 if f.get("collide", True) else 0
        wb.add_geom(name=f["name"], type=gtypes[ftype], pos=[x, y, z], quat=_quat_z(f.get("yaw", 0.0)),
                    size=size, rgba=rgba, contype=c, conaffinity=3 * c)

    bx = layout["box"]
    hx, hy, hz = bx["size"]
    body = wb.add_body(name="box", pos=[box_pose[0], box_pose[1], hz + 0.001], quat=_quat_z(box_pose[2]))
    body.add_freejoint(name="box_free")
    body.add_geom(name="box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], mass=bx["mass"],
                  rgba=list(bx.get("rgb", [0.62, 0.45, 0.28])) + [1],
                  friction=[bx.get("friction", 0.35), 0.005, 0.0001], contype=1, conaffinity=3,
                  priority=1)  # box friction wins over the floor's (MuJoCo otherwise takes the max)
    # packing-tape strip on top, visual only
    body.add_geom(name="box_tape", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx * 1.001, 0.03, 0.001],
                  pos=[0, 0, hz], rgba=[0.80, 0.70, 0.52, 1], contype=0, conaffinity=0, mass=0)
    body.add_site(name="box_center", size=[0.01, 0.01, 0.01])

    cam = layout.get("third_camera")
    if cam:
        p = np.array(cam["pos"], float)
        fwd = np.array(cam["lookat"], float) - p
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        wb.add_camera(name="third", pos=p.tolist(), fovy=cam.get("fovy", 60),
                      mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
                      xyaxes=np.concatenate([right, up]).tolist())
    # soft indoor lighting: dim headlight + two shadowless directional lights (the ceiling would shadow a sun)
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.ambient = [0.28, 0.28, 0.28]
    for li in spec.lights:
        li.castshadow = False
        li.diffuse = [0.25, 0.25, 0.25]
    wb.add_light(name="window", pos=[0, 0, H - 0.1], dir=[0.2, -0.5, -1],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.35, 0.33, 0.30], castshadow=False)
    model = spec.compile()
    return (model, spec) if return_spec else model


def room_xml_from_layout(layout: dict, out_path: str, object_body: str = "push_object") -> str:
    """Optional starting point for the agent: layout.json (format above, no robot) -> room.xml in the compose
    contract (floor at z=0, one free body `object_body`). The object is put at the origin; compose stages it."""
    spec = build(layout, (0.0, 0.0, 0.0), return_spec=True)[1]
    pel = spec.body("pelvis")
    spec.delete(pel)
    for s in list(spec.sensors):
        spec.delete(s)
    for msh in list(spec.meshes):  # robot meshes; compose attaches the robot with its own assets
        spec.delete(msh)
    spec.body("box").name = object_body
    xml = spec.to_xml()
    with open(out_path, "w") as f:
        f.write(xml)
    return out_path

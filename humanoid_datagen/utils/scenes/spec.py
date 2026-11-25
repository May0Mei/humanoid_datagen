"""Common scene spec: the one interchange format every scene source exports to.

A spec is a plain dict, stored as YAML or JSON (``load_spec`` / ``save_spec``). ``validate_spec`` checks it
and ``normalize_spec`` fills defaults. ``composer.compose`` turns it into a MuJoCo MJCF with the G1.

Top-level keys (``?`` = optional)::

    schema_version: 1
    scene_id: str                       # unique, used for file names
    asset_root?: str                    # relative file paths in the spec resolve against this dir
                                        # (itself relative to the spec file); default: the spec's dir
    metadata:
      source: str                       # ours | agent_no_video | procedural | hand_designed | curated | ...
      source_detail?: str
      split?: str                       # train | test | t_self | t_twin | ...
      generation_cost?: {gpu_hours, cpu_hours, human_hours, notes}   # numbers or null (= not logged)
      tags?: {layout: [str], objects: [str], appearance: [str], physics: [str]}  # evaluation axes
      notes?: str
    room:
      extents: {x: [min, max], y: [min, max]}   # interior floor rectangle (m), z = 0 is the floor
      height: float
      floor: {material?, friction?}
      walls?: {thickness, material?, collide?, sides?: [x_min, x_max, y_min, y_max],
               openings?: [{wall: x_min|x_max|y_min|y_max, span: [a, b], z: [z0, z1]}]}
      ceiling?: {material?, collide?}            # omit (or null) for no ceiling
      baseboard?: {height, thickness, material?}
    materials?: {name: {rgba?, texture?: file, texrepeat?, texuniform?, specular?, shininess?,
                        reflectance?, emission?}}
    meshes?: {name: {file, scale?: [sx, sy, sz], inertia?: convex|exact|legacy|shell}}
    lighting?:
      headlight?: {ambient, diffuse, specular}
      lights?: [{name?, type: directional|spot|point, pos, dir?, diffuse?, specular?, castshadow?,
                 cutoff?, exponent?, attenuation?}]
    cameras?: [{name, pos, lookat | xyaxes | quat, fovy?, resolution?}]   # static world cameras
    objects: [Object]
    robot?:
      start_pose?: {xy: [x, y], yaw: rad, z?: float}   # pelvis pose; z defaults to the robot MJCF's
      staged?: bool                                    # true if the agent / a human chose the pose
      staged_by?: str

Object::

    id: str                    # unique; body name defaults to the id
    category: str              # semantic label (layout stats, co-occurrence, object-novelty tags)
    dynamic: bool              # true: free joint; false: welded to the world
    task_relevant?: bool
    body_name?: str
    pose: {pos: [x, y, z], quat?: [w, x, y, z] | yaw?: rad}
    scale?: float | [sx, sy, sz]          # applied to geom/part positions, sizes and meshes
    mass?: float                          # kg, root-body mass split over its collision geoms by volume
    friction?: [slide, spin, roll]        # default for this object's collision geoms
    contact?: {solref?, solimp?, solmix?, condim?, margin?, gap?}
    collision?: {method: convex_hull|coacd|none, coacd?: {...}}   # colliders for mesh geoms
    geoms: [Geom]                         # root-body geoms, in the object frame
    parts?: [{name, parent?: part name (null = root), pos, quat?, mass?,
              joint?: {type: ball|hinge|slide, axis?, range?, stiffness?, damping?, armature?},
              geoms: [Geom]}]             # body <id>_<name>; articulated / multi-body objects (e.g. a plush ragdoll)
    semantic?: dict                       # free-form extras (asset ids, retrieval scores, ...)

Geom::

    name?: str                 # MJCF name becomes <object id>_<name> (unique within the object)
    type: box|sphere|capsule|cylinder|ellipsoid|mesh|plane
    size?: [...]               # MuJoCo conventions (half sizes / radii)
    fromto?: [x0 y0 z0 x1 y1 z1]   # capsule / cylinder
    pos?: [x, y, z]
    quat?: [w, x, y, z]
    mesh?: str                 # key into `meshes`
    material?: str             # key into `materials`
    rgba?: [r, g, b, a]
    role?: both|visual|collision   # default both. Mesh + both: visual mesh plus generated colliders.
    mass?: float
    friction?, solref?, solimp?, solmix?, condim?, margin?, gap?   # per-geom overrides
    mujoco_only?: bool         # MuJoCo stand-in, skipped by the Blender exporter
"""
from __future__ import annotations

import copy
import json
import math
import os
from typing import Any

import yaml

SCHEMA_VERSION = 1
KNOWN_SOURCES = ("ours", "agent_no_video", "procedural", "hand_designed", "curated", "digital_cousin", "t_twin",
                 "other")
TAG_AXES = ("layout", "objects", "appearance", "physics")
GEOM_TYPES = ("box", "sphere", "capsule", "cylinder", "ellipsoid", "mesh", "plane")
GEOM_ROLES = ("both", "visual", "collision")
COLLISION_METHODS = ("convex_hull", "coacd", "none")
JOINT_TYPES = ("ball", "hinge", "slide")
LIGHT_TYPES = ("directional", "spot", "point")
WALL_SIDES = ("x_min", "x_max", "y_min", "y_max")
# Number of size values per type (fromto variants need one fewer).
SIZE_LEN = {"box": (3,), "sphere": (1, 3), "capsule": (1, 2, 3), "cylinder": (1, 2, 3), "ellipsoid": (3,),
            "plane": (2, 3), "mesh": (0, 3)}


class SpecError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------------------------


def load_spec(path: str) -> dict:
    """Read a YAML/JSON spec and remember where it came from (for relative asset paths)."""
    with open(path) as f:
        spec = json.load(f) if path.endswith(".json") else yaml.safe_load(f)
    spec["_spec_path"] = os.path.abspath(path)
    return spec


def save_spec(spec: dict, path: str) -> str:
    out = {k: v for k, v in spec.items() if not k.startswith("_")}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        if path.endswith(".json"):
            json.dump(out, f, indent=1)
        else:
            yaml.safe_dump(out, f, sort_keys=False, default_flow_style=None, width=120)
    return path


def asset_root(spec: dict) -> str:
    base = os.path.dirname(spec.get("_spec_path", os.path.join(os.getcwd(), "x")))
    root = spec.get("asset_root")
    return os.path.normpath(os.path.join(base, root)) if root else base


def resolve(spec: dict, path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(asset_root(spec), path))


# ----------------------------------------------------------------------------------------------
# Small helpers shared by composer / sampler
# ----------------------------------------------------------------------------------------------


def yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def object_quat(obj: dict) -> list[float]:
    pose = obj.get("pose", {})
    if pose.get("quat") is not None:
        return [float(q) for q in pose["quat"]]
    return yaw_quat(float(pose.get("yaw", 0.0)))


def object_scale(obj: dict) -> list[float]:
    s = obj.get("scale", 1.0)
    return [float(s)] * 3 if isinstance(s, (int, float)) else [float(x) for x in s]


def room_rect(spec: dict) -> tuple[float, float, float, float]:
    ex = spec["room"]["extents"]
    return float(ex["x"][0]), float(ex["y"][0]), float(ex["x"][1]), float(ex["y"][1])


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------


def _num_list(v, n=None) -> bool:
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v):
        return False
    return n is None or len(v) in ((n,) if isinstance(n, int) else n)


def _check_geom(g: Any, where: str, spec: dict, errs: list[str], check_files: bool):
    if not isinstance(g, dict):
        errs.append(f"{where}: geom must be a mapping")
        return
    t = g.get("type")
    if t not in GEOM_TYPES:
        errs.append(f"{where}: type {t!r} not in {GEOM_TYPES}")
        return
    if g.get("role", "both") not in GEOM_ROLES:
        errs.append(f"{where}: role {g.get('role')!r} not in {GEOM_ROLES}")
    has_ft = g.get("fromto") is not None
    if has_ft:
        if t not in ("capsule", "cylinder", "box", "ellipsoid"):
            errs.append(f"{where}: fromto only for capsule/cylinder/box/ellipsoid")
        if not _num_list(g["fromto"], 6):
            errs.append(f"{where}: fromto must be 6 numbers")
    size = g.get("size", [])
    if t != "mesh" or size:
        if not _num_list(size) or len(size) not in SIZE_LEN[t]:
            errs.append(f"{where}: size {size!r} has wrong length for {t} (allowed {SIZE_LEN[t]})")
        elif t != "plane" and any(s <= 0 for s in size):
            errs.append(f"{where}: size must be positive")
        elif t in ("capsule", "cylinder") and not has_ft and len(size) < 2:
            errs.append(f"{where}: {t} needs [radius, half_length] or fromto")
    for k, n in (("pos", 3), ("quat", 4), ("rgba", 4)):
        if g.get(k) is not None and not _num_list(g[k], n):
            errs.append(f"{where}: {k} must be {n} numbers")
    if g.get("quat") is not None and _num_list(g["quat"], 4) and sum(q * q for q in g["quat"]) < 1e-12:
        errs.append(f"{where}: zero quaternion")
    if t == "mesh":
        m = g.get("mesh")
        if m not in (spec.get("meshes") or {}):
            errs.append(f"{where}: mesh {m!r} not declared in `meshes`")
    mat = g.get("material")
    if mat is not None and mat not in (spec.get("materials") or {}):
        errs.append(f"{where}: material {mat!r} not declared in `materials`")
    if g.get("mass") is not None and not (isinstance(g["mass"], (int, float)) and g["mass"] >= 0):
        errs.append(f"{where}: mass must be >= 0")
    if g.get("friction") is not None and not _num_list(g["friction"], (1, 2, 3)):
        errs.append(f"{where}: friction must be 1-3 numbers")


def validate_spec(spec: dict, check_files: bool = True, raise_on_error: bool = True) -> list[str]:
    """Return a list of problems (empty = valid). Raises SpecError if raise_on_error and any problem exists."""
    errs: list[str] = []
    if not isinstance(spec, dict):
        raise SpecError("spec must be a mapping")
    if spec.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(spec.get("scene_id"), str) or not spec.get("scene_id"):
        errs.append("scene_id must be a non-empty string")

    md = spec.get("metadata")
    if not isinstance(md, dict) or not isinstance(md.get("source"), str):
        errs.append("metadata.source is required")
    else:
        for k, v in (md.get("generation_cost") or {}).items():
            if k != "notes" and v is not None and not (isinstance(v, (int, float)) and v >= 0):
                errs.append(f"metadata.generation_cost.{k} must be a number >= 0 or null")
        for k, v in (md.get("tags") or {}).items():
            if k not in TAG_AXES:
                errs.append(f"metadata.tags.{k}: unknown axis (allowed {TAG_AXES})")
            elif not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                errs.append(f"metadata.tags.{k} must be a list of strings")

    room = spec.get("room")
    if not isinstance(room, dict):
        errs.append("room is required")
    else:
        ex = room.get("extents", {})
        for ax in ("x", "y"):
            if not _num_list(ex.get(ax), 2) or ex[ax][0] >= ex[ax][1]:
                errs.append(f"room.extents.{ax} must be [min, max] with min < max")
        if not (isinstance(room.get("height"), (int, float)) and room["height"] > 0):
            errs.append("room.height must be > 0")
        walls = room.get("walls")
        if walls:
            if not (isinstance(walls.get("thickness"), (int, float)) and walls["thickness"] > 0):
                errs.append("room.walls.thickness must be > 0")
            for s in walls.get("sides", WALL_SIDES):
                if s not in WALL_SIDES:
                    errs.append(f"room.walls.sides: {s!r} not in {WALL_SIDES}")
            for i, o in enumerate(walls.get("openings") or []):
                w = f"room.walls.openings[{i}]"
                if o.get("wall") not in WALL_SIDES:
                    errs.append(f"{w}.wall must be one of {WALL_SIDES}")
                    continue
                if not _num_list(o.get("span"), 2) or not _num_list(o.get("z"), 2):
                    errs.append(f"{w}: span and z must be [a, b]")
                    continue
                lo, hi = ex.get("y" if o["wall"].startswith("x") else "x", [0, 0])
                if not (lo <= o["span"][0] < o["span"][1] <= hi):
                    errs.append(f"{w}: span outside the wall")
                if not (0 <= o["z"][0] < o["z"][1] <= room.get("height", 0)):
                    errs.append(f"{w}: z outside the wall")
        for key in ("floor", "walls", "ceiling", "baseboard"):
            sub = room.get(key) or {}
            if sub.get("material") is not None and sub["material"] not in (spec.get("materials") or {}):
                errs.append(f"room.{key}.material {sub['material']!r} not declared")

    for name, m in (spec.get("materials") or {}).items():
        if m.get("rgba") is not None and not _num_list(m["rgba"], 4):
            errs.append(f"materials.{name}.rgba must be 4 numbers")
        if check_files and m.get("texture") and not os.path.isfile(resolve(spec, m["texture"])):
            errs.append(f"materials.{name}.texture missing: {resolve(spec, m['texture'])}")
    for name, m in (spec.get("meshes") or {}).items():
        if not isinstance(m, dict) or not m.get("file"):
            errs.append(f"meshes.{name}.file is required")
        elif check_files and not os.path.isfile(resolve(spec, m["file"])):
            errs.append(f"meshes.{name}.file missing: {resolve(spec, m['file'])}")

    for i, li in enumerate((spec.get("lighting") or {}).get("lights") or []):
        if li.get("type", "spot") not in LIGHT_TYPES:
            errs.append(f"lighting.lights[{i}].type not in {LIGHT_TYPES}")
        if not _num_list(li.get("pos"), 3):
            errs.append(f"lighting.lights[{i}].pos must be 3 numbers")
    for i, c in enumerate(spec.get("cameras") or []):
        if not c.get("name") or not _num_list(c.get("pos"), 3):
            errs.append(f"cameras[{i}] needs name and pos")
        if sum(k in c for k in ("lookat", "xyaxes", "quat")) != 1:
            errs.append(f"cameras[{i}] needs exactly one of lookat / xyaxes / quat")

    objs = spec.get("objects")
    if not isinstance(objs, list):
        errs.append("objects must be a list")
        objs = []
    ids, bodies = set(), set()
    for i, o in enumerate(objs):
        w = f"objects[{i}]"
        if not isinstance(o, dict):
            errs.append(f"{w} must be a mapping")
            continue
        oid = o.get("id")
        w = f"objects[{i}] ({oid})"
        if not isinstance(oid, str) or not oid:
            errs.append(f"{w}: id required")
        elif oid in ids:
            errs.append(f"{w}: duplicate id")
        ids.add(oid)
        if not isinstance(o.get("category"), str) or not o.get("category"):
            errs.append(f"{w}: category required")
        if not isinstance(o.get("dynamic"), bool):
            errs.append(f"{w}: dynamic must be true/false")
        pose = o.get("pose") or {}
        if not _num_list(pose.get("pos"), 3):
            errs.append(f"{w}: pose.pos must be 3 numbers")
        if pose.get("quat") is not None and not _num_list(pose["quat"], 4):
            errs.append(f"{w}: pose.quat must be 4 numbers")
        if pose.get("quat") is not None and pose.get("yaw") is not None:
            errs.append(f"{w}: give pose.quat or pose.yaw, not both")
        s = o.get("scale", 1.0)
        if not ((isinstance(s, (int, float)) and s > 0) or (_num_list(s, 3) and min(s) > 0)):
            errs.append(f"{w}: scale must be > 0 (scalar or 3-vector)")
        if o.get("mass") is not None and not (isinstance(o["mass"], (int, float)) and o["mass"] > 0):
            errs.append(f"{w}: mass must be > 0")
        if o.get("friction") is not None and not _num_list(o["friction"], (1, 2, 3)):
            errs.append(f"{w}: friction must be 1-3 numbers")
        meth = (o.get("collision") or {}).get("method", "convex_hull")
        if meth not in COLLISION_METHODS:
            errs.append(f"{w}: collision.method {meth!r} not in {COLLISION_METHODS}")
        bname = o.get("body_name", oid)
        if bname in bodies:
            errs.append(f"{w}: duplicate body name {bname!r}")
        bodies.add(bname)
        geoms = o.get("geoms") or []
        gnames = [g.get("name") for g in geoms + [g for p in o.get("parts") or [] for g in p.get("geoms") or []]
                  if isinstance(g, dict) and g.get("name")]
        if len(gnames) != len(set(gnames)):
            errs.append(f"{w}: duplicate geom names within the object")
        for j, g in enumerate(geoms):
            _check_geom(g, f"{w}.geoms[{j}]", spec, errs, check_files)
        part_names = set()
        n_col = sum(1 for g in geoms if isinstance(g, dict) and g.get("role", "both") != "visual")
        for k, p in enumerate(o.get("parts") or []):
            pw = f"{w}.parts[{k}]"
            pn = p.get("name")
            if not pn or pn in part_names:
                errs.append(f"{pw}: missing or duplicate part name")
            if p.get("parent") is not None and p["parent"] not in part_names:
                errs.append(f"{pw}: parent {p['parent']!r} must be listed before this part (or null = root)")
            part_names.add(pn)
            bodies.add(f"{oid}_{pn}")
            if not _num_list(p.get("pos", [0, 0, 0]), 3):
                errs.append(f"{pw}: pos must be 3 numbers")
            j = p.get("joint")
            if j is not None:
                if j.get("type") not in JOINT_TYPES:
                    errs.append(f"{pw}: joint.type not in {JOINT_TYPES}")
                if not o.get("dynamic"):
                    errs.append(f"{pw}: joints on a static object")
            for jj, g in enumerate(p.get("geoms") or []):
                _check_geom(g, f"{pw}.geoms[{jj}]", spec, errs, check_files)
            n_col += sum(1 for g in p.get("geoms") or [] if g.get("role", "both") != "visual")
        if o.get("dynamic") and n_col == 0:
            errs.append(f"{w}: dynamic object without any collision geom")
        if not geoms and not o.get("parts"):
            errs.append(f"{w}: no geoms")

    rob = spec.get("robot") or {}
    sp = rob.get("start_pose")
    if sp is not None:
        if not _num_list(sp.get("xy"), 2) or not isinstance(sp.get("yaw", 0.0), (int, float)):
            errs.append("robot.start_pose needs xy: [x, y] and yaw")
        elif isinstance(room, dict) and _num_list(room.get("extents", {}).get("x"), 2):
            x0, y0, x1, y1 = room_rect(spec)
            if not (x0 < sp["xy"][0] < x1 and y0 < sp["xy"][1] < y1):
                errs.append("robot.start_pose.xy outside the room")

    if errs and raise_on_error:
        raise SpecError("invalid scene spec:\n  " + "\n  ".join(errs))
    return errs


def normalize_spec(spec: dict) -> dict:
    """Deep copy with defaults filled in (does not validate)."""
    s = copy.deepcopy(spec)
    s.setdefault("materials", {})
    s.setdefault("meshes", {})
    s.setdefault("lighting", {})
    s["lighting"].setdefault("lights", [])
    s.setdefault("cameras", [])
    s.setdefault("robot", {})
    md = s.setdefault("metadata", {})
    md.setdefault("tags", {})
    md.setdefault("generation_cost", {"gpu_hours": None, "cpu_hours": None, "human_hours": None})
    for o in s.get("objects", []):
        o.setdefault("task_relevant", False)
        o.setdefault("body_name", o["id"])
        o.setdefault("geoms", [])
        o.setdefault("parts", [])
        o.setdefault("collision", {})
        o["collision"].setdefault("method", "convex_hull")
        for g in o["geoms"] + [g for p in o["parts"] for g in p.get("geoms", [])]:
            g.setdefault("role", "both")
    return s

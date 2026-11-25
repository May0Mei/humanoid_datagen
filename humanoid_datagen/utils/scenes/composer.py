"""Scene spec -> MuJoCo MJCF with the G1 (or any robot MJCF given in the config).

The robot MJCF is used as the base document and left untouched apart from:
  * its asset file paths, rewritten relative to the output XML (meshdir/texturedir/assetdir resolved);
  * optionally its own world-level floor / lights (``robot.strip_world_tags``); the scene floor reuses
    the name in ``floor_geom_name`` so robot contact pairs that reference it (thor: foot-floor) still resolve;
  * the pose of its root body (spec ``robot.start_pose``, else ``robot.default_start_pose``).
Bodies, joints, sensors, actuators and the head camera (d435 / oak1) are copied verbatim, so the egocentric camera is
exactly the one in the robot MJCF.

Every scene object becomes one body named ``body_name`` (default: its id) welded to the world (static) or
with a free joint (dynamic); parts become child bodies ``<id>_<part>``; named geoms become ``<id>_<geom name>``. The room shell lives in the
world body. ``compose`` returns (and writes next to the XML) a meta dict mapping objects to bodies.
"""
from __future__ import annotations

import copy
import json
import math
import os
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from humanoid_datagen.utils.scenes import geometry, spec as S
from humanoid_datagen.utils.paths import repo_path

CUBE_FILE_ATTRS = ("file", "fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")


def f2s(v) -> str:
    if isinstance(v, (list, tuple, np.ndarray)):
        return " ".join("%.9g" % float(x) for x in v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "%.9g" % v
    return str(v)


def _attrs(d: dict | None) -> dict[str, str]:
    return {k: f2s(v) for k, v in (d or {}).items() if v is not None}


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def lookat_xyaxes(pos, target):
    f = np.subtract(target, pos)
    f = f / np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.concatenate([r, u])


# ----------------------------------------------------------------------------------------------
# Robot document
# ----------------------------------------------------------------------------------------------


def _expand_includes(root: ET.Element, xml_dir: str):
    for parent in list(root.iter()):
        for i, child in enumerate(list(parent)):
            if child.tag == "include":
                sub = ET.parse(os.path.join(xml_dir, child.get("file"))).getroot()
                _expand_includes(sub, xml_dir)
                parent.remove(child)
                for k, el in enumerate(list(sub)):
                    parent.insert(i + k, el)


def load_robot_root(robot_xml: str, out_dir: str) -> tuple[ET.Element, str]:
    """Parse the robot MJCF, inline includes and make asset paths relative to out_dir. Returns (root, angle)."""
    xml_dir = os.path.dirname(os.path.abspath(robot_xml))
    root = ET.parse(robot_xml).getroot()
    _expand_includes(root, xml_dir)
    comp = root.find("compiler")
    attrs = comp.attrib if comp is not None else {}
    assetdir = attrs.get("assetdir", "")
    meshdir = attrs.get("meshdir", assetdir)
    texdir = attrs.get("texturedir", assetdir)
    for asset in root.iter("asset"):
        for el in asset:
            base = texdir if el.tag == "texture" else meshdir
            for a in CUBE_FILE_ATTRS:
                if el.get(a):
                    p = os.path.normpath(os.path.join(xml_dir, base, el.get(a)))
                    el.set(a, os.path.relpath(p, out_dir))
    angle = attrs.get("angle", "degree")
    if comp is not None:
        for a in ("assetdir", "meshdir", "texturedir"):
            comp.attrib.pop(a, None)
    return root, angle


def _strip_world(root: ET.Element, tags) -> list[str]:
    removed = []
    for wb in root.findall("worldbody"):
        for el in list(wb):
            if el.tag in tags:
                removed.append(f"{el.tag}:{el.get('name', '')}")
                wb.remove(el)
    return removed


def _set_robot_pose(root: ET.Element, body_name: str, xy, yaw, z):
    body = None
    for wb in root.findall("worldbody"):
        for b in wb.findall("body"):
            if b.get("name") == body_name:
                body = b
    if body is None:
        raise ValueError(f"robot root body {body_name!r} not found at the top of the robot worldbody")
    for a in ("euler", "axisangle", "xyaxes", "zaxis"):
        if body.get(a) is not None:
            raise NotImplementedError(f"robot root body orientation given as {a}; use quat")
    q0 = [float(x) for x in body.get("quat", "1 0 0 0").split()]
    p0 = [float(x) for x in body.get("pos", "0 0 0").split()]
    body.set("pos", f2s([xy[0], xy[1], p0[2] if z is None else z]))
    body.set("quat", f2s(quat_mul(S.yaw_quat(yaw), q0)))
    return body


# ----------------------------------------------------------------------------------------------
# Composer
# ----------------------------------------------------------------------------------------------


class _Ctx:
    def __init__(self, spec: dict, cfg: dict, out_dir: str, angle: str):
        self.spec, self.cfg, self.out_dir = spec, cfg, out_dir
        self.deg = angle == "degree"
        self.asset = ET.Element("asset")
        self.mesh_assets: dict[tuple, str] = {}
        self.mesh_files: dict[str, str] = {}
        self.mujoco_only: list[str] = []
        self.coacd_cache = repo_path(cfg["collision"]["cache_dir"])

    def rel(self, path: str) -> str:
        return os.path.relpath(path, self.out_dir)

    def mesh_asset(self, name: str, scale) -> str:
        key = (name, tuple(round(float(s), 9) for s in scale))
        if key in self.mesh_assets:
            return self.mesh_assets[key]
        m = self.spec["meshes"][name]
        sc = np.multiply(m.get("scale", [1, 1, 1]), scale)
        aname = name if np.allclose(scale, 1) else f"{name}__s{len(self.mesh_assets)}"
        attrs = {"name": aname, "file": self.rel(S.resolve(self.spec, m["file"])), "scale": f2s(sc)}
        if m.get("inertia"):
            attrs["inertia"] = m["inertia"]
        ET.SubElement(self.asset, "mesh", attrs)
        self.mesh_assets[key] = aname
        self.mesh_files[aname] = S.resolve(self.spec, m["file"])
        return aname

    def angle(self, v: float) -> float:
        return math.degrees(v) if self.deg else v


def _scaled_geom(g: dict, sc) -> dict:
    g = copy.deepcopy(g)
    sc = np.asarray(sc, float)
    if g.get("pos") is not None:
        g["pos"] = list(np.multiply(g["pos"], sc))
    if g.get("fromto") is not None:
        g["fromto"] = list(np.multiply(g["fromto"], np.tile(sc, 2)))
    size = g.get("size")
    if size:
        t = g["type"]
        if t in ("box", "ellipsoid"):
            g["size"] = list(np.multiply(size, sc))
        elif t == "plane":
            g["size"] = [size[0] * sc[0], size[1] * sc[1]] + list(size[2:])
        else:  # radius-like: use the geometric mean of the scale
            m = float(np.prod(sc) ** (1 / 3))
            g["size"] = [s * m for s in size]
    return g


def _contact_attrs(obj: dict, g: dict) -> dict:
    out = dict(obj.get("contact") or {})
    if obj.get("friction") is not None:
        out["friction"] = obj["friction"]
    for k in ("friction", "solref", "solimp", "solmix", "condim", "margin", "gap", "priority"):
        if g.get(k) is not None:
            out[k] = g[k]
    return out


def _body_geoms(ctx: _Ctx, obj: dict, body: ET.Element, geoms: list[dict], target_mass: float | None, sc):
    """Append geoms of one body; returns nothing. Masses split over collision geoms by volume."""
    cfg = ctx.cfg
    dyn = obj["dynamic"]
    col_cls = "scene_dynamic" if dyn else "scene_static"
    vis_cls = "scene_visual"
    colgroup = cfg["collision"]["collision_group"]
    method = obj.get("collision", {}).get("method") or cfg["collision"]["default_method"]
    pending = []  # (attrs, volume, explicit_mass)
    for g in geoms:
        g = _scaled_geom(g, sc)
        role = g.get("role", "both")
        name = f"{obj['id']}_{g['name']}" if g.get("name") else None
        base = {}
        if g["type"] == "mesh":
            base["mesh"] = ctx.mesh_asset(g["mesh"], sc)
        if g.get("size"):
            base["size"] = g["size"]
        for k in ("fromto", "pos", "quat"):
            if g.get(k) is not None and not (k == "fromto" and g["type"] == "mesh"):
                base[k] = g[k]
        if g.get("fromto") is not None:
            base.pop("pos", None)
            base.pop("quat", None)
        look = {}
        if g.get("material"):
            look["material"] = g["material"]
        if g.get("rgba") is not None:
            look["rgba"] = g["rgba"]
        if g.get("mujoco_only") and name:
            ctx.mujoco_only.append(name)
        is_mesh = g["type"] == "mesh"
        # Visual geom: a separate non-colliding geom for meshes (role both/visual) and visual-only prims.
        if role == "visual" or (is_mesh and role == "both"):
            body.append(ET.Element("geom", _attrs({"name": name, "type": g["type"], "class": vis_cls, **base, **look})))
        if role == "visual":
            continue
        contact = _contact_attrs(obj, g)
        if not is_mesh:
            attrs = {"name": name, "type": g["type"], "class": col_cls, **base, **contact}
            if role == "collision":
                attrs["group"] = colgroup
                attrs["rgba"] = g.get("rgba", cfg["collision"]["collision_rgba"])
            else:
                attrs.update(look)
            pending.append((attrs, geometry.primitive_volume(g), g.get("mass")))
            continue
        if method == "none":
            continue
        mfile = ctx.mesh_files[base["mesh"]]
        mscale = np.multiply(ctx.spec["meshes"][g["mesh"]].get("scale", [1, 1, 1]), sc)
        pose = {k: base[k] for k in ("pos", "quat") if k in base}
        if method == "convex_hull":
            hulls = [(base["mesh"], geometry.mesh_volume(mfile, mscale))]
        else:
            params = dict(cfg["collision"]["coacd"])
            params.update((obj.get("collision") or {}).get("coacd") or {})
            hulls = []
            for k, part in enumerate(geometry.coacd_parts(mfile, mscale, params, ctx.coacd_cache)):
                an = f"{base['mesh']}__cd{k:03d}"
                if an not in ctx.mesh_files:
                    ET.SubElement(ctx.asset, "mesh", {"name": an, "file": ctx.rel(part)})
                    ctx.mesh_files[an] = part
                hulls.append((an, geometry.mesh_volume(part)))
        for k, (an, vol) in enumerate(hulls):
            cname = None if name is None else (f"{name}_col" if len(hulls) == 1 else f"{name}_col{k:03d}")
            attrs = {"name": cname, "type": "mesh", "mesh": an, "class": col_cls, "group": colgroup,
                     "rgba": cfg["collision"]["collision_rgba"], **pose, **contact}
            pending.append((attrs, vol, None if g.get("mass") is None else g["mass"] / len(hulls)))
    # Masses: explicit per-geom masses first, the rest of target_mass by volume.
    if dyn and pending:
        explicit = sum(m for _, _, m in pending if m is not None)
        free = [(a, v) for a, v, m in pending if m is None]
        vol = sum(v for _, v in free)
        for a, v, m in pending:
            if m is not None:
                a["mass"] = m
            elif target_mass is not None:
                rest = max(target_mass - explicit, 0.0)
                a["mass"] = rest * (v / vol if vol > 0 else 1.0 / len(free))
    elif dyn and target_mass:
        ET.SubElement(body, "inertial", _attrs({"pos": [0, 0, 0], "mass": target_mass,
                                                "diaginertia": [1e-4 * target_mass] * 3}))
    for a, _, _ in pending:
        body.append(ET.Element("geom", _attrs(a)))


def object_body(ctx: _Ctx, obj: dict) -> ET.Element:
    sc = S.object_scale(obj)
    bname = obj["body_name"]
    body = ET.Element("body", _attrs({"name": bname, "pos": obj["pose"]["pos"], "quat": S.object_quat(obj)}))
    if obj["dynamic"]:
        body.append(ET.Element("freejoint", {"name": f"{obj['id']}_free"}))
    _body_geoms(ctx, obj, body, obj.get("geoms", []), obj.get("mass"), sc)
    elems = {None: body}
    for p in obj.get("parts", []):
        pb = ET.Element("body", _attrs({"name": f"{obj['id']}_{p['name']}", "pos": np.multiply(p.get("pos", [0, 0, 0]), sc),
                                        "quat": p.get("quat")}))
        j = p.get("joint")
        if j is not None:
            ja = {"name": f"{obj['id']}_{p['name']}_joint", "type": j["type"]}
            for k in ("axis", "stiffness", "damping", "armature", "frictionloss", "springref"):
                if j.get(k) is not None:
                    ja[k] = j[k]
            if j.get("range") is not None:
                ja["range"] = [ctx.angle(r) if j["type"] != "slide" else r for r in j["range"]]
            pb.append(ET.Element("joint", _attrs(ja)))
        _body_geoms(ctx, obj, pb, p.get("geoms", []), p.get("mass"), sc)
        elems[p.get("parent")].append(pb)
        elems[p["name"]] = pb
    return body


def room_geoms(spec: dict, cfg: dict) -> list[ET.Element]:
    room = spec["room"]
    x0, y0, x1, y1 = S.room_rect(spec)
    H = float(room["height"])
    cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    out = []
    floor = room.get("floor") or {}
    fa = {"name": cfg["floor_geom_name"], "type": "plane", "class": "scene_static", "pos": [cx, cy, 0],
          "size": [hx, hy, cfg["floor_plane_spacing"]], "material": floor.get("material"),
          "friction": floor.get("friction")}
    out.append(ET.Element("geom", _attrs(fa)))
    walls = room.get("walls")
    if walls:
        t = float(walls["thickness"])
        cls = "scene_static" if walls.get("collide", True) else "scene_visual"
        for side in walls.get("sides", S.WALL_SIDES):
            along_x = side.startswith("y")  # y walls run along x
            lo, hi = (x0, x1) if along_x else (y0 - t, y1 + t)
            fixed = {"x_min": x0 - t / 2, "x_max": x1 + t / 2, "y_min": y0 - t / 2, "y_max": y1 + t / 2}[side]
            ops = sorted([o for o in walls.get("openings") or [] if o["wall"] == side], key=lambda o: o["span"][0])
            pieces = []  # (a, b, z0, z1)
            cur = lo
            for o in ops:
                a, b = o["span"]
                if a > cur:
                    pieces.append((cur, a, 0.0, H))
                if o["z"][0] > 0:
                    pieces.append((a, b, 0.0, o["z"][0]))
                if o["z"][1] < H:
                    pieces.append((a, b, o["z"][1], H))
                cur = b
            if cur < hi:
                pieces.append((cur, hi, 0.0, H))
            for k, (a, b, za, zb) in enumerate(pieces):
                c, h = (a + b) / 2, (b - a) / 2
                pos = [c, fixed, (za + zb) / 2] if along_x else [fixed, c, (za + zb) / 2]
                size = [h, t / 2, (zb - za) / 2] if along_x else [t / 2, h, (zb - za) / 2]
                name = f"wall_{side}" if len(pieces) == 1 else f"wall_{side}_{k}"
                out.append(ET.Element("geom", _attrs({"name": name, "type": "box", "class": cls, "pos": pos,
                                                      "size": size, "material": walls.get("material")})))
        bb = room.get("baseboard")
        if bb:
            bh, bt = bb["height"] / 2, bb["thickness"] / 2
            for side in walls.get("sides", S.WALL_SIDES):
                pos = {"x_min": [x0 + bt, cy, bh], "x_max": [x1 - bt, cy, bh], "y_min": [cx, y0 + bt, bh],
                       "y_max": [cx, y1 - bt, bh]}[side]
                size = [bt, hy, bh] if side.startswith("x") else [hx, bt, bh]
                out.append(ET.Element("geom", _attrs({"name": f"baseboard_{side}", "type": "box", "class": "scene_visual",
                                                      "pos": pos, "size": size, "material": bb.get("material")})))
    ceil = room.get("ceiling")
    if ceil is not None:
        t = float((walls or {}).get("thickness", cfg["default_ceiling_thickness"]))
        cls = "scene_static" if ceil.get("collide", False) else "scene_visual"
        out.append(ET.Element("geom", _attrs({"name": "ceiling", "type": "box", "class": cls, "pos": [cx, cy, H + t / 2],
                                              "size": [hx, hy, t / 2], "material": ceil.get("material")})))
    return out


def compose(spec: dict, cfg: dict, out_xml: str, robot: bool | None = None, start_pose: dict | None = None) -> dict:
    """Write the MJCF for `spec` to out_xml and return the scene meta dict.

    cfg: the `composer` config section (plain dict). robot: override cfg.robot.include.
    start_pose: override the robot start pose {xy, yaw, z?}.
    """
    S.validate_spec(spec)
    spec = S.normalize_spec(spec)
    out_xml = os.path.abspath(out_xml)
    out_dir = os.path.dirname(out_xml)
    os.makedirs(out_dir, exist_ok=True)
    rcfg = cfg["robot"]
    include_robot = rcfg["include"] if robot is None else robot

    if include_robot:
        robot_xml = repo_path(rcfg["mjcf"])
        root, angle = load_robot_root(robot_xml, out_dir)
        stripped = _strip_world(root, rcfg["strip_world_tags"])
        sp = start_pose or spec["robot"].get("start_pose") or rcfg["default_start_pose"]
        z = sp.get("z") if sp.get("z") is not None else rcfg.get("base_height")
        _set_robot_pose(root, rcfg["root_body"], sp["xy"], sp.get("yaw", 0.0), z)
    else:
        root, angle, stripped, sp, robot_xml = ET.Element("mujoco"), "radian", [], None, None
        ET.SubElement(root, "compiler", {"angle": "radian"})
    root.set("model", f"{spec['scene_id']}" + ("_g1" if include_robot else ""))
    comp = root.find("compiler")
    comp.attrib.update(_attrs(cfg["compiler"]))

    ctx = _Ctx(spec, cfg, out_dir, angle)
    # option / size / visual: robot values < composer config < spec.simulation
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(list(root).index(comp) + 1, opt)
    opt.attrib.update(_attrs(cfg["option"]))
    opt.attrib.update(_attrs((spec.get("simulation") or {}).get("option")))
    flags = dict(cfg.get("option_flags") or {})
    flags.update((spec.get("simulation") or {}).get("option_flags") or {})
    if flags:
        fl = opt.find("flag")
        if fl is None:
            fl = ET.SubElement(opt, "flag")
        fl.attrib.update(_attrs(flags))
    size = root.find("size")
    if size is None:
        size = ET.Element("size")
        root.insert(list(root).index(opt) + 1, size)
    size.attrib.update(_attrs(cfg["size"]))
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    vcfg = dict(cfg["visual"])
    hl = (spec.get("lighting") or {}).get("headlight")
    for sec in ("global", "quality", "map", "headlight"):
        attrs = dict(vcfg.get(sec) or {})
        if sec == "headlight" and hl:
            attrs.update(hl)
        if attrs:
            el = vis.find(sec)
            if el is None:
                el = ET.SubElement(vis, sec)
            el.attrib.update(_attrs(attrs))

    # Defaults for scene geoms.
    default = root.find("default")
    if default is None:
        default = ET.Element("default")
        root.insert(list(root).index(vis) + 1, default)
    for cname, gattrs in cfg["classes"].items():
        d = ET.SubElement(default, "default", {"class": cname})
        ET.SubElement(d, "geom", _attrs(gattrs))

    # Materials / textures.
    for name, m in spec["materials"].items():
        ma = {"name": name}
        for k in ("rgba", "specular", "shininess", "reflectance", "emission", "texrepeat", "texuniform"):
            if m.get(k) is not None:
                ma[k] = m[k]
        if m.get("texture"):
            tname = f"{name}_tex"
            ET.SubElement(ctx.asset, "texture", {"name": tname, "type": m.get("texture_type", "2d"),
                                                 "file": ctx.rel(S.resolve(spec, m["texture"]))})
            ma["texture"] = tname
        ET.SubElement(ctx.asset, "material", _attrs(ma))

    world = ET.Element("worldbody")
    for li in spec["lighting"]["lights"]:
        la = {k: li.get(k) for k in ("name", "pos", "dir", "diffuse", "specular", "ambient", "castshadow", "cutoff",
                                      "exponent", "attenuation")}
        t = li.get("type", "spot")
        if t == "directional":
            la["directional"] = True
        world.append(ET.Element("light", _attrs(la)))
    for c in spec["cameras"]:
        ca = {"name": c["name"], "pos": c["pos"], "fovy": c.get("fovy"), "resolution": c.get("resolution")}
        if "lookat" in c:
            ca["xyaxes"] = lookat_xyaxes(c["pos"], c["lookat"])
        elif "xyaxes" in c:
            ca["xyaxes"] = c["xyaxes"]
        else:
            ca["quat"] = c["quat"]
        world.append(ET.Element("camera", _attrs(ca)))
    for el in room_geoms(spec, cfg):
        world.append(el)
    objects_meta = {}
    for o in spec["objects"]:
        b = object_body(ctx, o)
        world.append(b)
        objects_meta[o["id"]] = dict(category=o["category"], raw_category=(o.get("semantic") or {}).get("raw_category"),
                                     dynamic=o["dynamic"], task_relevant=o["task_relevant"],
                                     root_body=o["body_name"], bodies=[e.get("name") for e in b.iter("body")])

    # Insert scene assets and bodies into the document.
    idx = list(root).index(default) + 1
    root.insert(idx, ctx.asset)
    wb = root.find("worldbody")
    if wb is None:
        root.append(world)
    else:
        for k, el in enumerate(list(world)):
            wb.insert(k, el)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(out_xml)
    # Pin the near plane in metres (MuJoCo's znear is relative to the model extent) so the ego view
    # keeps the same clip distance (hides the lens housing) whatever the room size.
    import mujoco

    model = mujoco.MjModel.from_xml_path(out_xml)
    near = vcfg.get("near_clip")
    if near:
        vis.find("map").set("znear", "%.6g" % (near / model.stat.extent))
        ET.ElementTree(root).write(out_xml)
        model = mujoco.MjModel.from_xml_path(out_xml)
    if include_robot and rcfg.get("camera"):
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, rcfg["camera"]) < 0:
            raise ValueError(f"camera {rcfg['camera']!r} not in the composed model")

    x0, y0, x1, y1 = S.room_rect(spec)
    meta = dict(scene_id=spec["scene_id"], xml=out_xml, spec_path=spec.get("_spec_path"),
                metadata=spec["metadata"], room=dict(rect=[x0, y0, x1, y1], height=spec["room"]["height"]),
                floor_geom=cfg["floor_geom_name"], objects=objects_meta, mujoco_only_geoms=ctx.mujoco_only,
                robot=dict(included=bool(include_robot), mjcf=robot_xml, root_body=rcfg["root_body"],
                           camera=rcfg.get("camera"), start_pose=sp, stripped_world_elements=stripped,
                           spec_start_pose=spec["robot"].get("start_pose"), staged=spec["robot"].get("staged")),
                counts=dict(nbody=model.nbody, ngeom=model.ngeom, nmesh=model.nmesh, nq=model.nq, nv=model.nv),
                timestep=float(model.opt.timestep))
    with open(os.path.splitext(out_xml)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_meta(xml_path: str) -> dict[str, Any]:
    with open(os.path.splitext(xml_path)[0] + "_meta.json") as f:
        return json.load(f)

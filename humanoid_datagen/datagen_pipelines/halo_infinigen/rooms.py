"""HALO's Infinigen rooms as push-datagen rooms (room.xml contract of datagen_pipelines/shared/compose.py), for the two Infinigen
baseline arms: our motions and our objects in HALO's procedural scenes.

    python -m humanoid_datagen.datagen_pipelines.halo_infinigen.rooms room   [scene ...]           (all 50 scenes by default)
    python -m humanoid_datagen.datagen_pipelines.halo_infinigen.rooms object <scene> <pid> [--arm ours|text]

room    outputs/push_box/infinigen/rooms/<scene>/room.xml: HALO's visual room parts (the same baked, unlit meshes and
        full-ambient headlight as the HALO-Infinigen data, episodes/_mjscene/<scene>/scene_with_robot.xml) shifted so
        the floor is z = 0, plus invisible box colliders (group 3) from scene_analysis.json: a floor plane, four walls
        around the floor rectangle (every scene is one rectangular room), and one box per obstacle asset's AABB,
        clipped to the room and to floor..ceiling. Skipped: skirting boards (the walls cover them), assets entirely
        above `max_low_z` (wall shelves' contents, lamps on high shelves), and boxes inside a larger obstacle box
        (shelf trinkets).
object  outputs/push_box/infinigen/rooms/<scene>/<pid>_<arm>/room.xml: that room plus the push object of our agent
        scene <scenes_root>/<pid>_s0/room.xml (body push_object with its geoms, meshes, textures and materials;
        asset files referenced by absolute path), so compose_room / datagen take it like an agent room.
"""
from __future__ import annotations

import json
import os
import re
import warnings

import mujoco
import numpy as np

from humanoid_datagen.utils.paths import repo_path
from humanoid_datagen.datagen_pipelines.shared.base import Stage

INF = "outputs/push_box/halo_infinigen"
OUT = "outputs/push_box/infinigen/rooms"
SCENES_ROOT = dict(ours="outputs/push_box/scenes", text="outputs/push_box/scenes_text")
WALL_T = 0.1          # wall collider thickness (m), outside the floor rectangle
max_low_z = 1.4       # an asset whose bottom is above this (m over the floor) cannot block the robot or the object
SKIP = ("skirtingboard",)


def scenes() -> list[str]:
    return sorted(os.listdir(repo_path(f"{INF}/episodes/_mjscene")))


def colliders(a: dict) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], np.ndarray, np.ndarray, float]:
    """Collider boxes (name, lo, hi) in floor-at-zero coordinates; floor rectangle lo/hi (xy) and ceiling height."""
    fz = float(a["floor_z"])
    st = {s["cls"]: s for s in a["assets"] if s["is_structural"]}
    flo, fhi = np.array(st["floor"]["bbox_min"][:2]), np.array(st["floor"]["bbox_max"][:2])
    ceil = float(st["ceiling"]["bbox_min"][2]) - fz
    boxes = []
    for s in a["assets"]:
        if not s["is_obstacle"] or s["cls"].startswith(SKIP):
            continue
        lo, hi = np.array(s["bbox_min"], float), np.array(s["bbox_max"], float)
        lo[2] -= fz
        hi[2] -= fz
        lo[:2], hi[:2] = np.maximum(lo[:2], flo), np.minimum(hi[:2], fhi)
        lo[2], hi[2] = max(lo[2], 0.0), min(hi[2], ceil)
        if np.any(hi - lo <= 0.005) or lo[2] > max_low_z:
            continue
        boxes.append((s["name"], lo, hi))
    # drop boxes inside another one (trinkets on a shelf, pillows on a bed)
    keep = []
    for i, (n, lo, hi) in enumerate(boxes):
        inside = any(j != i and np.all(lo >= l2 - 1e-3) and np.all(hi <= h2 + 1e-3) and np.prod(h2 - l2) > np.prod(hi - lo)
                     for j, (_, l2, h2) in enumerate(boxes))
        if not inside:
            keep.append((n, lo, hi))
    return keep, flo, fhi, ceil


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)[:60]


def room(scene: str) -> str:
    """Write rooms/<scene>/room.xml; returns its path."""
    out = repo_path(f"{OUT}/{scene}/room.xml")
    a = json.load(open(repo_path(f"{INF}/scenes/single/{scene}/scene_analysis.json")))
    fz = float(a["floor_z"])
    xml = open(repo_path(f"{INF}/episodes/_mjscene/{scene}/scene_with_robot.xml")).read()
    xml = xml.replace('  <include file="g1_29dof_ego.xml"/>\n', "")
    xml = xml.replace('<mujoco model="infinigen_scene_with_robot">', f'<mujoco model="infinigen_{scene}">')
    boxes, flo, fhi, ceil = colliders(a)
    g = ['<geom name="floor" type="plane" size="60 60 0.1" rgba="0.5 0.5 0.5 0" group="3"/>']
    c, h = (flo + fhi) / 2, (fhi - flo) / 2
    zc, zh = ceil / 2, ceil / 2
    for k, (px, py, sx, sy) in enumerate([(c[0], flo[1] - WALL_T / 2, h[0] + WALL_T, WALL_T / 2),
                                          (c[0], fhi[1] + WALL_T / 2, h[0] + WALL_T, WALL_T / 2),
                                          (flo[0] - WALL_T / 2, c[1], WALL_T / 2, h[1]),
                                          (fhi[0] + WALL_T / 2, c[1], WALL_T / 2, h[1])]):
        g.append(f'<geom name="wall_{k}" type="box" pos="{px:.4f} {py:.4f} {zc:.4f}" size="{sx:.4f} {sy:.4f} {zh:.4f}" '
                 f'rgba="0.5 0.5 0.5 0" group="3"/>')
    for k, (n, lo, hi) in enumerate(boxes):
        p, s = (lo + hi) / 2, (hi - lo) / 2
        g.append(f'<geom name="obs_{k:03d}_{_safe(n)}" type="box" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
                 f'size="{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}" rgba="0.8 0.3 0.3 0" group="3"/>')
    a_, b_ = xml.index("<worldbody>") + len("<worldbody>"), xml.index("</worldbody>")
    xml = (xml[:a_] + f'\n    <frame pos="0 0 {-fz:.4f}">{xml[a_:b_]}    </frame>\n    ' + "\n    ".join(g) + "\n  "
           + xml[b_:])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(xml)
    json.dump(dict(scene=scene, floor_z=fz, floor_lo=flo.tolist(), floor_hi=fhi.tolist(), ceiling=ceil,
                   n_obstacle_boxes=len(boxes), free_area_m2=a["occupancy"]["free_area_m2"]),
              open(os.path.join(os.path.dirname(out), "room_info.json"), "w"), indent=1)
    mujoco.MjModel.from_xml_path(out)  # compiles
    return out


def _absolute_assets(spec: mujoco.MjSpec, base: str) -> None:
    md = os.path.join(base, spec.meshdir or "")
    td = os.path.join(base, spec.texturedir or "")
    for m in spec.meshes:
        if m.file and not os.path.isabs(m.file):
            m.file = os.path.normpath(os.path.join(md, m.file))
    for t in spec.textures:
        if t.file and not os.path.isabs(t.file):
            t.file = os.path.normpath(os.path.join(td, t.file))
        if any(t.cubefiles):
            t.cubefiles = [os.path.normpath(os.path.join(td, f)) if f and not os.path.isabs(f) else f for f in t.cubefiles]
    for hf in spec.hfields:
        if hf.file and not os.path.isabs(hf.file):
            hf.file = os.path.normpath(os.path.join(base, hf.file))


def with_object(scene: str, pid: str, arm: str = "ours", body: str = "push_object") -> str:
    """Write rooms/<scene>/<pid>_<arm>/room.xml = the Infinigen room + our agent scene's push object."""
    src_xml = repo_path(f"{SCENES_ROOT[arm]}/{pid}_s0/room.xml")
    dst_xml = repo_path(f"{OUT}/{scene}/room.xml")
    if not os.path.exists(dst_xml):
        room(scene)
    src = mujoco.MjSpec.from_file(src_xml)
    _absolute_assets(src, os.path.dirname(src_xml))
    dst = mujoco.MjSpec.from_file(dst_xml)
    frame = dst.worldbody.add_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame.attach_body(src.body(body), "obj_", "")
    b = dst.body("obj_" + body)
    b.name = body
    b.pos = [0.0, 0.0, b.pos[2]]                      # compose_room stages the object; any free spot is fine here
    m = dst.compile()
    if sum(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE for j in range(m.njnt)) != 1:
        raise RuntimeError(f"{pid}: expected exactly one free joint")
    out = repo_path(f"{OUT}/{scene}/{pid}_{arm}/room.xml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dst.meshdir = dst.texturedir = "/"
    open(out, "w").write(dst.to_xml())
    mujoco.MjModel.from_xml_path(out)
    return out


class InfinigenRooms(Stage):
    """Datagen stage `rooms` of the HALO-Infinigen pipeline: room.xml per Infinigen scene (C["scenes"], or all)."""

    def run(self) -> None:
        C, tk = self.cfg, self.task
        for sc in C.get("scenes") or scenes():
            p = room(sc)
            print(sc, json.load(open(os.path.join(os.path.dirname(p), "room_info.json")))["n_obstacle_boxes"], "boxes")

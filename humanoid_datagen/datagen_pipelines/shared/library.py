"""Asset library tool: search ~7.6k annotated, metric-scale 3D assets (AI2-THOR artist assets + Objaverse home
objects, each with visual meshes, textures and convex colliders) and export any of them into the run dir.

    python -m humanoid_datagen.datagen_pipelines.shared.library search "a grey fabric armchair" [--k 8] [--category armchair]
    python -m humanoid_datagen.datagen_pipelines.shared.library export <uid> --name sofa [--scale 1.1 | --height 0.85] [--static]
    python -m humanoid_datagen.datagen_pipelines.shared.library categories

search  CLIP text query (laion ViT-L/14) against each asset's 3 rendered views and its description; prints uid,
        category, description, size [x, y, z] m (z up; front faces +x after export), mass, and writes a thumbnail
        sheet library/search_NN.png.
export  writes library/<name>/asset.xml: one body <name> (no joint) whose origin is the bottom centre of the asset
        and whose front faces +x, with the asset's meshes, textures and colliders copied to library/<name>/files
        (absolute paths, so the file can be included, attached with MjSpec, or copied from anywhere). Colliding
        geoms keep contype/conaffinity from the asset; --static makes them plain static world collision (1/1).
        Use it in room.xml, e.g.
            child = mujoco.MjSpec.from_file("library/sofa/asset.xml")
            frame = room.worldbody.add_frame(pos=[x, y, 0], quat=[...])
            frame.attach_body(child.body("sofa"), "sofa/", "")
        or for the pushed object: attach it inside your `push_object` body (which carries the freejoint) and set
        its mass / friction yourself.
Asset licenses: THOR CC BY 4.0, Objaverse per object (by / by-sa / cc0), printed with each result.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("ASSET_LIBRARY_DIR", os.path.join(HERE, "..", "..", "..", "library"))   # agent sandbox: <runtime>/library
CLIP = os.environ.get("ASSET_LIBRARY_CLIP", "laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
# per source: rotation that makes the export z-up / front +x (Unity Y-up exports)
FIX = {"thor": ([90, 0, 0], 90), "objaverse": ([90, 0, 0], 90)}


# ------------------------------------------------------------------------------------------------ index (driver side)
def export_index(index_pkl: str, out_dir: str, exclude: set[str] | None = None, assets_root: str | None = None):
    """index.pkl of the scene-agent asset cache -> library/{index.npz, meta.jsonl} (no project information).
    `exclude`: uids left out entirely (the test set's held-out assets, configs/eval/push/test_split.json).
    `assets_root`: local molmo-spaces-resources dir; the index stores each asset's xml as an absolute path from the
    machine that built it, so everything up to `molmo-spaces-resources/` is replaced by it."""
    import pickle
    d = pickle.load(open(index_pkl, "rb"))["data"]
    keep = np.array([u not in (exclude or set()) for u in d["uids"]])
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "index.npz"), img=d["img"][keep].astype(np.float16),
             txt=d["txt"][keep].astype(np.float16))
    with open(os.path.join(out_dir, "meta.jsonl"), "w") as f:
        for u, k in zip(d["uids"], keep):
            if not k:
                continue
            m = d["meta"][u]
            bb = m.get("boundingBox") or {}
            lic = m.get("license_info") or {}
            xml = d["xml"][u]
            if assets_root:
                xml = os.path.join(assets_root, xml.split("molmo-spaces-resources/", 1)[1])
            f.write(json.dumps(dict(uid=u, source=d["src"][u], xml=xml, category=m.get("category"),
                                    description=m.get("description"), size=[bb.get("x"), bb.get("y"), bb.get("z")],
                                    mass=m.get("mass"), license=lic.get("license") or "cc-by-4.0 (thor)",
                                    creator=lic.get("creator_display_name"))) + "\n")


def _meta():
    return [json.loads(l) for l in open(os.path.join(DATA, "meta.jsonl"))]


def _encode(texts):
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TQDM_DISABLE", "1")
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    m = CLIPModel.from_pretrained(CLIP).eval()
    t = CLIPTokenizer.from_pretrained(CLIP)
    tok = t(texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        e = m.get_text_features(**tok)
    e = getattr(e, "pooler_output", e).numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------------ export
def _rot(src):
    from scipy.spatial.transform import Rotation as R
    up, yaw = FIX[src]
    return (R.from_euler("z", yaw, degrees=True) * R.from_rotvec(np.radians(up))).as_quat(scalar_first=True)


def _walk(body):
    yield body
    for c in body.bodies:
        yield from _walk(c)


def build_asset(rec: dict, name: str, scale: float = 1.0, height: float | None = None, static: bool = False,
                files_dir: str | None = None) -> tuple[mujoco.MjSpec, np.ndarray]:
    """MjSpec with one body `name`: origin = bottom centre, z up, front +x. Returns (spec, size xyz)."""
    src_dir = os.path.dirname(rec["xml"])
    child = mujoco.MjSpec.from_file(rec["xml"])
    for el in list(child.meshes) + list(child.textures):
        if el.file and not os.path.isabs(el.file):
            el.file = os.path.join(src_dir, child.meshdir or "", el.file) if hasattr(el, "scale") else \
                os.path.join(src_dir, child.texturedir or "", el.file)
    top = child.worldbody.first_body()
    for b in _walk(top):
        for el in list(b.joints) + list(b.sites):   # no joints; sites are annotation markers (receptacles)
            child.delete(el)
    out = mujoco.MjSpec()
    out.compiler.degree = False
    body = out.worldbody.add_body(name=name)
    frame = body.add_frame(quat=_rot(rec["source"]))
    frame.attach_body(top, f"{name}/", "")
    # extent in the body frame (all geoms: visual meshes define what it looks like)
    m = out.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pts = []
    for g in range(m.ngeom):
        c, Rg, a = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_aabb[g]
        mid = m.geom_dataid[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and mid >= 0:   # exact: mesh vertices
            v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(c + v @ Rg.T)
        else:
            corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
            pts.append(c + (a[:3] + a[3:] * corners) @ Rg.T)
    pts = np.concatenate(pts)
    lo, hi = pts.min(0), pts.max(0)
    size = hi - lo
    if height:
        scale = float(height) / max(size[2], 1e-6)
    frame.pos = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    if scale != 1.0:
        for msh in out.meshes:
            msh.scale = np.asarray(msh.scale) * scale
        frame.pos = np.asarray(frame.pos) * scale
        for b in _walk(body):
            if b is not body:
                b.pos = np.asarray(b.pos) * scale
            for g in b.geoms:
                g.pos = np.asarray(g.pos) * scale
                if g.type != mujoco.mjtGeom.mjGEOM_MESH:
                    g.size = np.asarray(g.size) * scale
            for f in b.frames:
                if f is not frame:
                    f.pos = np.asarray(f.pos) * scale
        size = size * scale
    # mass: visual meshes weigh nothing; colliders share the annotated mass (scaled with volume)
    vis, col = [], []
    for b in _walk(body):
        for g in b.geoms:
            (col if (g.contype or g.conaffinity) else vis).append(g)
    for g in vis:
        g.mass, g.density = 0.0, 0.0
    m2 = out.compile()
    tot = float(m2.body_subtreemass[m2.body(name).id])
    want = float(rec.get("mass") or 0) * (scale ** 3 if scale != 1.0 else 1.0)
    if want > 0 and tot > 0:
        f = want / tot
        for g in col:
            if g.mass > 0:
                g.mass *= f
            else:
                g.density *= f
    if static:
        for b in _walk(body):
            for g in b.geoms:
                if g.contype or g.conaffinity:
                    g.contype, g.conaffinity = 1, 1
    if files_dir:
        os.makedirs(files_dir, exist_ok=True)
        for el in list(out.meshes) + list(out.textures):
            if el.file:
                dst = os.path.join(files_dir, f"{len(os.listdir(files_dir)):03d}_{os.path.basename(el.file)}")
                shutil.copyfile(el.file, dst)
                el.file = os.path.abspath(dst)
    return out, size


def export(uid: str, name: str, scale: float = 1.0, height: float | None = None, static: bool = False,
           root: str = "library") -> dict:
    rec = next((r for r in _meta() if r["uid"] == uid), None)
    if rec is None:
        raise SystemExit(f"unknown uid {uid!r}")
    d = os.path.join(root, name)
    if os.path.isdir(d):
        shutil.rmtree(d)
    spec, size = build_asset(rec, name, scale, height, static, files_dir=os.path.join(d, "files"))
    path = os.path.join(d, "asset.xml")
    with open(path, "w") as f:
        f.write(spec.to_xml())
    m = mujoco.MjSpec.from_file(path).compile()
    n_col = int(sum(1 for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]))
    mass = float(m.body_subtreemass[m.body(name).id])
    return dict(asset=os.path.abspath(path), body=name, uid=uid, category=rec["category"], size=np.round(size, 3).tolist(),
                mass_kg=round(mass, 2), n_colliders=n_col, annotated_mass_kg=rec["mass"], license=rec["license"],
                creator=rec["creator"], note="mass = annotated mass x scale^3 on the colliders; visual meshes are "
                "massless. Set your own mass / friction if you know better.")


# ------------------------------------------------------------------------------------------------ search
def _thumb(rec, cache):
    spec, size = build_asset(rec, "a")
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.1], rgba=[0.86, 0.86, 0.84, 1])
    spec.worldbody.add_light(pos=[1, -1, 4], dir=[-0.2, 0.2, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 320, 260
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 260, 320)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, size[2] / 2]
    cam.distance, cam.elevation, cam.azimuth = 0.6 + 1.6 * float(max(size)), -20, 200
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0                       # hide collider groups (THOR / Objaverse colliders are group 4)
    r.update_scene(d, cam, scene_option=opt)
    img = r.render().copy()
    r.close()
    return img, size


def search(query: str, k: int = 8, category: str | None = None, thumbs: bool = True) -> dict:
    meta = _meta()
    z = np.load(os.path.join(DATA, "index.npz"))
    q = _encode([f"a 3D model of {query}"])[0]
    score = (z["img"].astype(np.float32) @ q).max(1) + 0.5 * (z["txt"].astype(np.float32) @ q)
    if category:
        ok = np.array([(r["category"] or "").lower() == category.lower() for r in meta])
        score = np.where(ok, score, -np.inf)
    top = [int(i) for i in np.argsort(-score)[:k] if np.isfinite(score[i])]
    res = []
    for i in top:
        r = meta[i]
        res.append(dict(uid=r["uid"], score=round(float(score[i]), 3), category=r["category"], source=r["source"],
                        description=r["description"], annotated_size=[round(x, 3) if x else x for x in r["size"]],
                        annotated_mass_kg=r["mass"], license=r["license"]))
    out = dict(query=query, results=res)
    if thumbs and res:
        from PIL import Image, ImageDraw
        os.makedirs("library", exist_ok=True)
        n = len([f for f in os.listdir("library") if f.startswith("search_")])
        tiles, cache = [], {}
        for j, r in zip(top, res):
            try:
                img, size = _thumb(meta[j], cache)
                r["size"] = np.round(size, 3).tolist()
            except Exception as e:  # noqa: BLE001
                img = np.full((260, 320, 3), 255, np.uint8)
                r["thumb_error"] = str(e)[:120]
            im = Image.fromarray(img)
            ImageDraw.Draw(im).text((6, 6), f"{r['uid'][:24]}\n{r['category']}", fill=(0, 0, 0))
            tiles.append(im)
        cols = 4
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (320 * cols, 260 * rows), "white")
        for t, im in enumerate(tiles):
            sheet.paste(im, ((t % cols) * 320, (t // cols) * 260))
        p = os.path.join("library", f"search_{n:02d}.png")
        sheet.save(p)
        out["sheet"] = p
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="library", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=8)
    s.add_argument("--category")
    s.add_argument("--no-thumbs", action="store_true")
    e = sub.add_parser("export")
    e.add_argument("uid")
    e.add_argument("--name", required=True)
    e.add_argument("--scale", type=float, default=1.0)
    e.add_argument("--height", type=float, help="scale uniformly to this height (m)")
    e.add_argument("--static", action="store_true")
    sub.add_parser("categories")
    a = ap.parse_args(argv)
    if a.cmd == "search":
        r = search(a.query, a.k, a.category, not a.no_thumbs)
    elif a.cmd == "export":
        r = export(a.uid, a.name, a.scale, a.height, a.static)
    else:
        from collections import Counter
        r = dict(Counter(x["category"] for x in _meta()).most_common())
    print(json.dumps(r, indent=1))
    try:
        with open("tool_log.jsonl", "a") as f:
            f.write(json.dumps(dict(tool=f"library {a.cmd}", argv=sys.argv[1:])) + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    main()

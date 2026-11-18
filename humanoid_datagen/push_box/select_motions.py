"""Pick each prior's Kimodo-G1 push motions from oversampled candidates.

    python -m humanoid_datagen.push_box.select_motions --config_name push_box/motions
    (GPU job, after Kimodo: sbatch slurm_scripts/data/push_motions.sh; array task i = priors [i*per_task, (i+1)*per_task))

Per prior <root>/<pid>/: candidates in motion_candidates/c*.csv (Kimodo, kimodo.txt). A candidate is kept when
  - its palms meet the object's near face at least `min_top_clear` below the object's top (meta.json height_m;
    object.json size[2] for the first 54 priors) and
    no lower than `min_palm_frac` of its height (the same fit rule as the procedural arm's motion bank),
  - SONIC tracks it on an open floor without falling (head above run.FALL_HEAD_Z throughout),
  - its hands travel at least `min_push` m.
The first `keep` kept candidates (Kimodo sample order) become motions/m0..m{keep-1}.csv (+ .npz), what the scene agent
gets. motion_select.json records every candidate's numbers; motion_videos/m*.mp4 show each kept motion tracked by
SONIC (for review; not given to the agent).
"""
from __future__ import annotations

import glob
import json
import os
import shutil

import numpy as np


FPS = 30.0


def _video(model, tr: dict, ref, path: str):
    import imageio
    import mujoco
    from .cli import fwd_of
    d = mujoco.MjData(model)
    r = mujoco.Renderer(model, 360, 480)
    cam = mujoco.MjvCamera()
    pel = model.body("pelvis").id
    frames = []
    for k, q in enumerate(tr["qpos"][::2]):
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        cam.lookat[:] = d.xpos[pel]
        cam.distance, cam.elevation = 3.0, -10
        cam.azimuth = np.degrees(np.arctan2(*fwd_of(ref, min(2 * k, ref.T - 1))[::-1])) - 60
        r.update_scene(d, cam)
        frames.append(r.render().copy())
    r.close()
    imageio.mimsave(path, frames, fps=25)


def select_prior(pd: str, cfg: dict, rm, floor) -> dict:
    from .run import FALL_HEAD_Z, head_z, rollout
    from .sonic import Reference
    from .stage import stage
    meta = json.load(open(os.path.join(pd, "meta.json")))
    H = float(meta["height_m"]) if "height_m" in meta else \
        float(json.load(open(os.path.join(pd, "object.json")))["size"][2])  # first 54 priors: object card
    cands = sorted(glob.glob(os.path.join(pd, "motion_candidates", "c*.csv")))
    rows, kept = [], []
    for csv in cands:
        ref = Reference.from_qpos(np.loadtxt(csv, delimiter=","), FPS)
        st = stage(rm, ref, [0.1, 0.1, 0.1], friction=0.0, tip_safety=0.0)
        palm, push = float(st["contact_height"]), float(st["kinematic_push_dist"])
        fit = H - palm >= cfg["min_top_clear"] and palm >= cfg["min_palm_frac"] * H
        row = dict(candidate=os.path.basename(csv)[:-4], palm_height=round(palm, 3), push=round(push, 2), fits=fit)
        if fit and push >= cfg["min_push"]:  # track only what could be kept
            tr = rollout(floor, ref, "track", settle_s=0.0, tail_s=0.5)
            hz = float(head_z(floor, tr["qpos"]).min())
            row.update(min_head_z=round(hz, 3), tracked=hz > FALL_HEAD_Z)
            if row["tracked"] and len(kept) < cfg["keep"]:
                kept.append((csv, ref, tr))
                row["kept_as"] = f"m{len(kept) - 1}"
        rows.append(row)
    md, vd = os.path.join(pd, "motions"), os.path.join(pd, "motion_videos")
    for d in (md, vd):
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    for k, (csv, ref, tr) in enumerate(kept):
        for ext in (".csv", ".npz"):
            if os.path.exists(csv[:-4] + ext):
                shutil.copyfile(csv[:-4] + ext, os.path.join(md, f"m{k}{ext}"))
        if cfg["videos"]:
            _video(floor, tr, ref, os.path.join(vd, f"m{k}.mp4"))
    out = dict(object_height=H, rule={k: cfg[k] for k in ("min_top_clear", "min_palm_frac", "min_push", "keep")},
               n_candidates=len(cands), n_kept=len(kept), candidates=rows)
    json.dump(out, open(os.path.join(pd, "motion_select.json"), "w"), indent=1)
    return out


def main():
    from omegaconf import OmegaConf
    from humanoid_datagen.utils.config import parse_config
    from .compose import robot_floor_model, robot_only_model
    cfg = OmegaConf.to_container(parse_config("push_box/motions"), resolve=True)
    root = cfg["root"]
    pds = sorted(d for d in glob.glob(os.path.join(root, "p[0-9][0-9][0-9]")) if os.path.isdir(d))
    if os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        i, n = int(os.environ["SLURM_ARRAY_TASK_ID"]), int(cfg["per_task"])
        pds = pds[i * n:(i + 1) * n]
    rm, floor = robot_only_model(), robot_floor_model()
    for pd in pds:
        r = select_prior(pd, cfg, rm, floor)
        fits = sum(c["fits"] for c in r["candidates"])
        print(f"[select] {os.path.basename(pd)} H={r['object_height']:.2f}: {r['n_candidates']} candidates, {fits} fit, "
              f"kept {r['n_kept']}" + ("" if r["n_kept"] else "  NO MOTION: this prior will be skipped"), flush=True)


if __name__ == "__main__":
    main()

"""Pick each prior's Kimodo-G1 push motions from oversampled candidates.

    python humanoid_datagen/scripts/datagen.py --config_name datagen/push/agents/select
    (GPU job, after Kimodo: sbatch slurm_scripts/data/push_motions.sh; array task i = priors [i*per_task, (i+1)*per_task))

Per prior <root>/<pid>/: candidates in <candidate_dirs>/*.csv (Kimodo, kimodo.txt; default motion_candidates/c*.csv).
`reuse_tracking` takes a candidate's SONIC result from the existing motion_select.json instead of re-running it;
`priors` restricts the run to those prior ids. A candidate is kept when
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

from humanoid_datagen.utils.paths import repo_path
from humanoid_datagen.datagen_pipelines.shared.base import Stage

FPS = 30.0


def _video(model, tr: dict, ref, path: str):
    import imageio
    import mujoco
    from humanoid_datagen.datagen_pipelines.agents.cli import fwd_of
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


def select_prior(pd: str, cfg: dict, rm, floor, tk) -> dict:
    """tk: the task (its motion_fit decides which candidates fit the object)."""
    from humanoid_datagen.datagen_pipelines.shared.run import FALL_HEAD_Z, head_z, rollout
    from humanoid_datagen.utils.sonic import Reference
    meta = json.load(open(os.path.join(pd, "meta.json")))
    H = float(meta["height_m"]) if "height_m" in meta else \
        float(json.load(open(os.path.join(pd, "object.json")))["size"][2])  # first 54 priors: object card
    cands = [c for sub in cfg.get("candidate_dirs", ["motion_candidates"])
             for c in sorted(glob.glob(os.path.join(pd, sub, "*.csv")))]
    prev = {}
    sel = os.path.join(pd, "motion_select.json")
    if cfg.get("reuse_tracking") and os.path.exists(sel):  # SONIC results of an earlier selection run
        prev = {r["candidate"]: r for r in json.load(open(sel))["candidates"] if "tracked" in r}
    rows, kept = [], []
    for csv in cands:
        ref = Reference.from_qpos(np.loadtxt(csv, delimiter=","), FPS)
        f = tk.motion_fit(rm, ref, H)
        row = dict(candidate=os.path.basename(csv)[:-4], **{k: v for k, v in f.items() if k != "candidate_ok"})
        if f["candidate_ok"]:  # track only what could be kept
            name = row["candidate"]
            if name in prev:
                tr = None
                row.update(min_head_z=prev[name]["min_head_z"], tracked=prev[name]["tracked"])
            else:
                tr = rollout(floor, ref, "track", settle_s=0.0, tail_s=0.5)
                hz = float(head_z(floor, tr["qpos"]).min())
                row.update(min_head_z=round(hz, 3), tracked=hz > FALL_HEAD_Z)
            if row["tracked"] and len(kept) < cfg["keep"]:
                kept.append((csv, ref, tr))
                row["kept_as"] = f"m{len(kept) - 1}"
        rows.append(row)
    md, vd = os.path.join(pd, "motions"), os.path.join(pd, "motion_videos")
    for d in ((md, vd) if cfg["videos"] else (md,)):  # without videos, earlier review videos stay
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d)
    for k, (csv, ref, tr) in enumerate(kept):
        for ext in (".csv", ".npz"):
            if os.path.exists(csv[:-4] + ext):
                shutil.copyfile(csv[:-4] + ext, os.path.join(md, f"m{k}{ext}"))
        if cfg["videos"]:
            if tr is None:
                tr = rollout(floor, ref, "track", settle_s=0.0, tail_s=0.5)
            _video(floor, tr, ref, os.path.join(vd, f"m{k}.mp4"))
    out = dict(object_height=H, rule=dict(tk.params.get("motion_fit") or {}, keep=cfg["keep"]),
               n_candidates=len(cands), n_kept=len(kept), candidates=rows)
    json.dump(out, open(os.path.join(pd, "motion_select.json"), "w"), indent=1)
    return out


class SelectMotions(Stage):
    """Datagen stage `select`: priors of this array task (per_task chunk of SLURM_ARRAY_TASK_ID), else all."""

    def run(self) -> None:
        C, tk = self.cfg, self.task
        from humanoid_datagen.datagen_pipelines.shared.compose import robot_floor_model, robot_only_model
        rm, floor = robot_only_model(), robot_floor_model()
        from humanoid_datagen.datagen_pipelines.shared.priors import prior_dirs
        for pd in prior_dirs(C):
            r = select_prior(pd, C, rm, floor, tk)
            fits = sum(c["fits"] for c in r["candidates"])
            print(f"[select] {os.path.basename(pd)} H={r['object_height']:.2f}: {r['n_candidates']} candidates, {fits} fit, "
                  f"kept {r['n_kept']}" + ("" if r["n_kept"] else "  NO MOTION: this prior will be skipped"), flush=True)

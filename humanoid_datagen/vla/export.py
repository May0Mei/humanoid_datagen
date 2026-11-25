"""Export push demos to the HALO generator's dataset layout (ml-halo/generator/data/dataset.py):
<out_dir>/{data/episode_N.parquet, videos/observation.images.fpv_cam/episode_N.mp4 (symlink), meta/}.

    python humanoid_datagen/scripts/train.py --config_name train/push/export
    python humanoid_datagen/scripts/train.py --config_name train/push/export_halo_infinigen

`matched`: accepted datagen demos of several arms (datagen_pipelines/shared/datagen.py), with the same number per target category in
every arm; with `matched.total`, that many per arm spread uniformly over the shared scenes (same count per scene in
every arm).
`halo_episodes`: the HALO-baseline push episodes (datagen_pipelines/halo_infinigen/episodes.py; gate-accepted only).
(The walk-to export from a VLA config's `source_config` is in cache/walk_to/.)
"""
import collections
import glob
import json
import os
import random
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from humanoid_datagen.utils.paths import repo_path

FPS = 30


def write_parquet(path: str, q: np.ndarray) -> None:
    """q: (T, 36) qpos [xyz, wxyz, 29 joints]. Body 0 = pelvis, the only body the generator reads."""
    q = q.astype(np.float32)
    table = pa.table({
        "observation.joint_pos": q[:, 7:].tolist(),
        "observation.body_pos": q[:, None, :3].tolist(),
        "observation.body_quat": q[:, None, 3:7].tolist(),
    })
    pq.write_table(table, path)


def write_dataset(out: str, eps: list[dict], info: dict) -> None:
    """eps: dicts with motion (T, 36), video, text, demo."""
    if os.path.exists(out):
        shutil.rmtree(out)
    vid_dir = os.path.join(out, "videos", "observation.images.fpv_cam")
    for d in ("data", "meta", vid_dir):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    tasks = sorted({e["text"] for e in eps})
    with open(os.path.join(out, "meta", "tasks.jsonl"), "w") as f:
        f.writelines(json.dumps(dict(task_index=i, task=t)) + "\n" for i, t in enumerate(tasks))
    frames = 0
    with open(os.path.join(out, "meta", "episodes.jsonl"), "w") as ef, open(os.path.join(out, "meta", "sources.jsonl"), "w") as sf:
        for i, e in enumerate(eps):
            write_parquet(os.path.join(out, "data", f"episode_{i:06d}.parquet"), e["motion"])
            os.symlink(os.path.abspath(e["video"]), os.path.join(vid_dir, f"episode_{i:06d}.mp4"))
            ef.write(json.dumps(dict(episode_index=i, tasks=[e["text"]], length=len(e["motion"]))) + "\n")
            sf.write(json.dumps(dict(episode_index=i, demo=e["demo"])) + "\n")
            frames += len(e["motion"])
    with open(os.path.join(out, "meta", "info.json"), "w") as f:
        json.dump(dict(robot_type="g1", fps=FPS, total_episodes=len(eps), total_frames=frames, total_tasks=len(tasks), **info),
                  f, indent=1)
    print(f"{len(eps)} episodes, {frames} frames, {len(tasks)} instructions -> {out}")


def accepted_demos(root: str, camera: str, exclude=()) -> list[dict]:
    out = []
    for idx in sorted(glob.glob(os.path.join(repo_path(root), "*", "index.jsonl"))):
        if os.path.basename(os.path.dirname(idx)) in set(exclude or ()):
            continue
        for line in open(idx):
            r = json.loads(line)
            d = os.path.join(os.path.dirname(idx), r["demo_id"])
            if r.get("accepted") and os.path.exists(os.path.join(d, f"output_{camera}.mp4")):
                cat = json.load(open(os.path.join(d, "task.json")))["target"]["category"]
                out.append(dict(dir=d, demo=r["demo_id"], category=cat))
    return out


def export_matched(cfg) -> None:
    def load_motion_csv(path):  # 36-column qpos at 30 fps (models/vla/motion_chunk.py, now in cache/)
        q = np.loadtxt(path, delimiter=",", ndmin=2).astype(np.float32)
        if q.shape[1] != 36:
            raise ValueError(f"{path}: expected 36 columns, got {q.shape[1]}")
        return q
    m = cfg.matched
    demos = {arm: accepted_demos(root, m.camera, m.get("exclude_scenes")) for arm, root in m.arms.items()}
    if m.get("total"):
        return _export_per_scene(cfg, demos, load_motion_csv)
    by_cat = {arm: collections.defaultdict(list) for arm in demos}
    for arm, ds in demos.items():
        for d in ds:
            by_cat[arm][d["category"]].append(d)
    cats = sorted(set.intersection(*(set(b) for b in by_cat.values())))
    keep = {c: min(len(b[c]) for b in by_cat.values()) for c in cats}
    print("accepted:", {a: len(d) for a, d in demos.items()}, "kept per category:", keep)
    for arm, b in by_cat.items():
        rng = random.Random(int(m.seed))
        picked = [d for c in cats for d in rng.sample(sorted(b[c], key=lambda d: d["demo"]), keep[c])]
        eps = [dict(motion=load_motion_csv(os.path.join(d["dir"], "output.csv")),
                    video=os.path.join(d["dir"], f"output_{m.camera}.mp4"),
                    text=open(os.path.join(d["dir"], "prompt.txt")).read().strip(), demo=d["demo"]) for d in picked]
        write_dataset(os.path.join(repo_path(cfg.out_root), arm), eps,
                      dict(arm=arm, datagen_root=m.arms[arm], per_category=keep, accepted=len(demos[arm]),
                           exclude_scenes=list(m.get("exclude_scenes") or [])))


def _scene_quota(avail: dict, total: int, rng: random.Random) -> dict:
    """Spread `total` demos as evenly as possible over scenes, none above its `avail`: the largest level k with
    sum(min(avail, k)) <= total, then one more for a seeded random pick of the scenes that have more than k."""
    k = max(v for v in range(max(avail.values()) + 1) if sum(min(a, v) for a in avail.values()) <= total)
    q = {s: min(a, k) for s, a in avail.items()}
    rest = total - sum(q.values())
    if rest:
        for s in rng.sample(sorted(s for s, a in avail.items() if a > k), rest):
            q[s] += 1
    return q


def _export_per_scene(cfg, demos, load_motion_csv) -> None:
    """`matched.total` demos per arm, uniform over the scenes all arms share: every scene gets the same count (capped
    by the fewer accepted demos of the two arms), so the arms match per scene and hence per object category. Which
    scenes get one extra and which demos are picked are seeded by `matched.seed` (picks per scene, the same in every arm
    for a given scene and demo list)."""
    m = cfg.matched
    by_sc = {arm: collections.defaultdict(list) for arm in demos}
    for arm, ds in demos.items():
        for d in ds:
            by_sc[arm][d["demo"].split("__")[0]].append(d)
    scenes = sorted(set.intersection(*(set(b) for b in by_sc.values())))
    avail = {s: min(len(b[s]) for b in by_sc.values()) for s in scenes}
    total = int(m.total)
    assert sum(avail.values()) >= total, f"only {sum(avail.values())} matched demos for total {total}"
    quota = _scene_quota(avail, total, random.Random(int(m.seed)))
    print("accepted:", {a: len(d) for a, d in demos.items()}, f"{len(scenes)} scenes, quota:",
          dict(collections.Counter(quota.values())))
    for arm, b in by_sc.items():
        picked = [d for s in scenes
                  for d in random.Random(f"{m.seed}/{s}").sample(sorted(b[s], key=lambda d: d["demo"]), quota[s])]
        per_cat = collections.Counter(d["category"] for d in picked)
        eps = [dict(motion=load_motion_csv(os.path.join(d["dir"], "output.csv")),
                    video=os.path.join(d["dir"], f"output_{m.camera}.mp4"),
                    text=open(os.path.join(d["dir"], "prompt.txt")).read().strip(), demo=d["demo"]) for d in picked]
        write_dataset(os.path.join(repo_path(cfg.out_root), arm), eps,
                      dict(arm=arm, datagen_root=m.arms[arm], total=total, seed=int(m.seed), per_scene=quota,
                           per_category=dict(sorted(per_cat.items())), accepted=len(demos[arm]),
                           exclude_scenes=list(m.get("exclude_scenes") or [])))
        print(arm, "per category:", dict(sorted(per_cat.items())))


def export_halo_episodes(cfg) -> None:
    """HALO-baseline episodes (datagen_pipelines/halo_infinigen/episodes.py): <root>/episode_NNNN/{g1_qpos.npy, <video>, meta.json}. Keeps
    the episodes HALO's motion_check gate accepted (meta.json gate.accept), as HALO trains only on accepted motions.
    The motion is the scene-free qpos: the generator's targets are pelvis-relative to the window start and de-yawed
    (generator/data/traj_extract.py), so they equal those of the placed motion."""
    h = cfg.halo_episodes
    eps, n = [], 0
    for d in sorted(glob.glob(os.path.join(repo_path(h.root), "episode_*"))):
        meta = json.load(open(os.path.join(d, "meta.json")))
        n += 1
        if h.accepted_only and not meta["gate"]["accept"]:
            continue
        eps.append(dict(motion=np.load(os.path.join(d, "g1_qpos.npy")), video=os.path.join(d, h.video),
                        text=meta["instruction"], demo=os.path.basename(d)))
    print(f"{len(eps)} of {n} episodes kept")
    write_dataset(repo_path(cfg.out_dir), eps, dict(halo_episodes=h.root, accepted=len(eps), generated=n))


class HaloExport:
    """Demos -> HALO dataset (train config `step: {_target_: humanoid_datagen.vla.export.HaloExport, ...}`):
    `matched` datagen demos of several arms, or the HALO-baseline `halo_episodes`."""

    def __init__(self, **cfg):
        self.cfg = cfg

    def run(self) -> None:
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(self.cfg)
        if cfg.get("matched"):
            return export_matched(cfg)
        if cfg.get("halo_episodes"):
            return export_halo_episodes(cfg)
        raise ValueError("config needs `matched` or `halo_episodes`")

"""Rescore push-eval episodes with the hand rule (tasks/push/reward.py): replays each episode's saved trajectory
(qpos per 20 ms step) in its task model. Hand success = the eval's success AND the hand rule.

    python humanoid_datagen/scripts/eval.py --config_name eval/push/rescore
      -> <results_root>/hand_contact.<shard>.jsonl     (resumes; one line per episode)
"""
from __future__ import annotations

import glob
import json
import os
import warnings

import numpy as np

from humanoid_datagen.datagen_pipelines.shared.run import load_traj
from humanoid_datagen.tasks.push.reward import attribute, attribution_model, passes

warnings.simplefilter("ignore")


def rescore(roots, tasks, shard, tk):
    i, n = map(int, shard.split("/"))
    T = {t["task_id"]: t for t in map(json.loads, open(tasks))}
    eps = {}
    for root in roots:
        for r in map(json.loads, open(os.path.join(root, "episodes.jsonl"))):
            eps.setdefault(r["task_id"], []).append((root, r))
    outs = {root: os.path.join(root, f"hand_contact.{i}.jsonl") for root in roots}
    done = set()
    for root, p in outs.items():
        if os.path.exists(p):
            done |= {(root, (r := json.loads(l))["task_id"], r["seed"]) for l in open(p)}
    fh = {root: open(p, "a") for root, p in outs.items()}
    for tid in sorted(eps)[i::n]:
        todo = [(root, r) for root, r in eps[tid] if (root, tid, r["seed"]) not in done]
        if not todo:
            continue
        m0, _, _, _ = tk.compose_task(T[tid])
        m, obj, gp = attribution_model(m0, T[tid]["object_body"])
        for root, r in todo:
            try:
                z = load_traj(r["trajectory"])
                s = attribute(m, obj, gp, z["qpos"], z["obj_pos"], z["direction"], z["obj0"])
                s["hand_rule"] = passes(s, tk.hand_rule)
                s["hand_success"] = bool(r["success"] and s["hand_rule"])
            except Exception as e:  # noqa: BLE001
                s = dict(error=str(e))
            s.update(task_id=tid, seed=r["seed"], type=r["type"], source=r["source"], success=r["success"],
                     push_dist=r["push_dist"])
            fh[root].write(json.dumps(s) + "\n")
            fh[root].flush()
        print(tid, len(todo), flush=True)


def summary(roots, rule=None):
    for root in roots:
        R = [json.loads(l) for p in glob.glob(os.path.join(root, "hand_contact.*.jsonl")) for l in open(p)]
        ok = [r for r in R if "error" not in r]
        n = sum(1 for _ in open(os.path.join(root, "episodes.jsonl")))
        print(f"{root}: scored {len(ok)}/{n} ({len(R) - len(ok)} errors); success {sum(r['success'] for r in ok)}, "
              f"hand success {sum(r['hand_success'] for r in ok)} (rule {rule})")


def run(C: dict, tk) -> None:
    """Eval mode `rescore` (task.rescore): this shard (`shard` / `num_shards`, or SLURM_ARRAY_TASK_ID / _COUNT) of
    the episodes in C["rescore_roots"], then the totals over every shard written so far."""
    from humanoid_datagen.utils.paths import repo_path
    roots = [repo_path(r) for r in C["rescore_roots"]]
    sh = os.environ.get("SLURM_ARRAY_TASK_ID", C.get("shard", 0))
    ns = os.environ.get("SLURM_ARRAY_TASK_COUNT", C.get("num_shards", 1))
    rescore(roots, repo_path(C["tasks"]["file"]), f"{sh}/{ns}", tk)
    summary(roots, tk.hand_rule)

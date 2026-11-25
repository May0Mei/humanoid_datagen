"""Evaluation: an evaluator for a task, both instantiated from the run config.

    python humanoid_datagen/scripts/eval.py --config_name eval/push/grscenes_halo_ours

The config's `task` block builds the task and its `evaluator` block the evaluator (evaluators/), which gets the task;
`mode` names the evaluator method to call: build (draw the test set once: eval/<task>/build_grscenes), run (episodes
with the evaluator's planner), rescore (the task's rescoring of saved trajectories), viz (review videos).
Configs: humanoid_datagen/configs/eval/<task>/.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the repo root

from humanoid_datagen.utils.config import build, load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_name", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config_name)
    task = build(cfg.task)
    ev = build(cfg.evaluator, task=task)
    print(f"[eval] {type(task).__name__} / {type(ev).__name__}.{cfg.mode} ({a.config_name})", flush=True)
    getattr(ev, cfg.mode)()
    sys.stdout.flush()
    os._exit(0)          # MuJoCo / EGL teardown can hang at interpreter exit


if __name__ == "__main__":
    main()

"""Data generation: one stage of a pipeline for a task, both instantiated from the run config.

    python humanoid_datagen/scripts/datagen.py --config_name datagen/push/agents/demos

The config's `task` block builds the task (tasks/<name>/task.py) and its `stage` block the stage (a Stage subclass in
datagen_pipelines/), which gets the task; the script calls stage.run(). Configs:
humanoid_datagen/configs/datagen/<task>/<pipeline>/<stage>.yaml.
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
    stage = build(cfg.stage, task=task)
    print(f"[datagen] {type(task).__name__} / {type(stage).__name__} ({a.config_name})", flush=True)
    stage.run()


if __name__ == "__main__":
    main()

"""VLA training: one step, instantiated from the run config.

    python humanoid_datagen/scripts/train.py --config_name train/push/export      (demos -> HALO dataset)
    python humanoid_datagen/scripts/train.py --config_name train/push/halo_ours   (train the HALO generator)

The config's `step` block builds the step (vla/export.py HaloExport, vla/train.py HaloTrain); the script calls
run(). Configs: humanoid_datagen/configs/train/<task>/.
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
    step = build(cfg.step)
    print(f"[train] {type(step).__name__} ({a.config_name})", flush=True)
    step.run()


if __name__ == "__main__":
    main()

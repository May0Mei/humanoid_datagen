"""HALO generator training with HALO's unmodified trainer ($THIRD_PARTY_ROOT/ml-halo/generator/scripts/train.py) on a
HALO-format training config (train config `step: {_target_: humanoid_datagen.vla.train.HaloTrain, ...}`)."""
from __future__ import annotations

import subprocess
import sys

from humanoid_datagen.utils.paths import repo_path, third_party


class HaloTrain:
    def __init__(self, halo_config: str, gpus: int = 1, wandb_project: str = "humanoid_datagen_vla"):
        self.halo_config, self.gpus, self.wandb_project = halo_config, int(gpus), wandb_project

    def run(self) -> None:
        """One node; gpus > 1 through torchrun (data.batch_size in the HALO config is per GPU)."""
        launch = [sys.executable] if self.gpus <= 1 else ["torchrun", "--standalone", "--nproc_per_node", str(self.gpus)]
        cmd = launch + [third_party("ml-halo", "generator", "scripts", "train.py"), "--config", repo_path(self.halo_config),
                        "--wandb", "--wandb_project", self.wandb_project]
        raise SystemExit(subprocess.call(cmd, cwd=repo_path(".")))

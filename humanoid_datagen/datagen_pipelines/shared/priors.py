"""Prior folders (<priors_root>/p000..): the unit of work of the prior stages (videos, motions, select, leadin).
Light on purpose (stdlib only): the Wan2.2 and kimodo envs import it."""
from __future__ import annotations

import glob
import os

from humanoid_datagen.datagen_pipelines.shared.base import Stage
from humanoid_datagen.utils.paths import repo_path


def prior_dirs(C: dict) -> list[str]:
    """Prior folders under C["root"] (or `priors_root`): `priors` if set, then this array task's chunk of `per_task`
    (SLURM_ARRAY_TASK_ID; all priors without it)."""
    root = repo_path(C.get("root") or C["priors_root"])
    pds = sorted(d for d in glob.glob(os.path.join(root, "p[0-9][0-9][0-9]")) if os.path.isdir(d))
    if C.get("priors"):
        pds = [d for d in pds if os.path.basename(d) in set(C["priors"])]
    if os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        i, n = int(os.environ["SLURM_ARRAY_TASK_ID"]), int(C["per_task"])
        pds = pds[i * n:(i + 1) * n]
    return pds


class Prompts(Stage):
    """Datagen stage: prior folders from the task's prompt list (task.write_priors)."""

    def run(self) -> None:
        self.task.write_priors(self.cfg)

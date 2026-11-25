"""Base of every data-generation stage: a run config instantiates one (`stage: {_target_: ..., <settings>}`) with the
task (`task: {_target_: ...}`); scripts/datagen.py then calls run()."""
from __future__ import annotations


class Stage:
    def __init__(self, task, **cfg):
        self.task = task                # tasks.base.Task
        self.cfg = cfg                  # the stage's settings (the config's `stage` block without _target_)

    def run(self) -> None:
        raise NotImplementedError

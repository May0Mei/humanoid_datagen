"""The task of an agent session's tools: the agent batch driver puts the run config's task (task.to_config(), a Hydra
config with _target_) into $HDG_TASK as JSON, since the sandboxed session never sees the run config."""
from __future__ import annotations

import json
import os


def task_from_env():
    cfg = os.environ.get("HDG_TASK")
    if not cfg:
        raise ValueError("no task: $HDG_TASK (set by datagen_pipelines/agents/batch.py) is empty")
    from hydra.utils import instantiate
    return instantiate(json.loads(cfg))

"""Config loading. Every entry script takes --config_name and composes humanoid_datagen/configs/<name>.yaml with
Hydra; `build` turns a config block with `_target_` into its object."""
from __future__ import annotations

import os

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def load_config(config_name: str, overrides: list[str] | None = None):
    """Hydra composition of configs/<config_name>.yaml (a subpath like 'datagen/push/agents/demos') -> DictConfig."""
    import hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        return hydra.compose(config_name=config_name.removesuffix(".yaml"), overrides=overrides or [])


def build(node, **extra):
    """The object of a config block {_target_: <class path>, **settings}: the block is resolved to plain Python
    (dicts / lists) and the class called with `extra` (e.g. task=...) and the settings as keyword arguments."""
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    kw = OmegaConf.to_container(node, resolve=True) if not isinstance(node, dict) else dict(node)
    return get_class(kw.pop("_target_"))(**extra, **kw)

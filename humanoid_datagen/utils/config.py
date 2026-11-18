"""Shared config loading: every script takes --config_name and reads humanoid_datagen/configs/<name>.yaml via Hydra."""
import argparse
import os

import hydra
from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def load_config(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """Compose configs/<config_name>.yaml (may be a subpath like 'rl/g1_walk') with optional dotlist overrides."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.1"):
        cfg = hydra.compose(config_name=config_name.removesuffix(".yaml"), overrides=overrides or [])
    return cfg


def parse_config(default_config_name: str) -> DictConfig:
    """Standard CLI: --config_name NAME [key=value ...]."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default=default_config_name)
    args, rest = parser.parse_known_args()
    cfg = load_config(args.config_name, rest)
    print(OmegaConf.to_yaml(cfg))
    return cfg

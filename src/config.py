"""Central configuration loader.

The entire system is driven by ``config/config.yaml``. This module loads that
file into a nested object that supports attribute access (``cfg.dl.epochs``) and
validates the key invariants so that an invalid config fails fast instead of
producing silently-wrong experiments.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigNode(dict):
    """A ``dict`` with recursive attribute access.

    Nested dictionaries become ``ConfigNode`` instances and list items are
    wrapped as well, so ``cfg.datasets.skab.cv.n_splits`` works out of the box.
    """

    def __init__(self, data: dict | None = None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self._wrap(value)


def validate_config(cfg: ConfigNode) -> ConfigNode:
    """Validate the key invariants of a loaded config. Raises ``ValueError``."""
    if not cfg.get("seed_list"):
        raise ValueError("config: seed_list bos olamaz")

    fixed = cfg.fixed_params
    if fixed.window_size < 2:
        raise ValueError("config: fixed_params.window_size >= 2 olmali")
    if fixed.alphabet_size < 2:
        raise ValueError("config: fixed_params.alphabet_size >= 2 olmali")

    if any(w < 2 for w in cfg.param_grid.window_size):
        raise ValueError("config: param_grid.window_size degerleri >= 2 olmali")
    if any(a < 2 for a in cfg.param_grid.alphabet_size):
        raise ValueError("config: param_grid.alphabet_size degerleri >= 2 olmali")

    split = cfg.datasets.batadal.split
    total = split.train + split.val + split.test
    if not 0.99 <= total <= 1.01:
        raise ValueError(f"config: BATADAL split toplami 1.0 olmali (su an {total})")

    if cfg.preprocess.pca.n_components < 1:
        raise ValueError("config: pca.n_components >= 1 olmali")

    return cfg


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ConfigNode:
    """Load and validate the YAML config into a :class:`ConfigNode`."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return validate_config(ConfigNode(raw))


if __name__ == "__main__":  # hizli gozden gecirme
    print(load_config())

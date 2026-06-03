import pytest

from src.config import ConfigNode, load_config, validate_config


def test_load_config_has_expected_structure():
    cfg = load_config()
    assert cfg.fixed_params.window_size == 4
    assert cfg.fixed_params.alphabet_size == 3
    assert cfg.dl.epochs == 50
    assert cfg.dl.batch_size == 32
    assert cfg.dl.early_stopping.patience == 5
    assert cfg.datasets.skab.cv.n_splits == 5
    assert cfg.datasets.skab.target == "anomaly"
    assert cfg.datasets.batadal.target == "ATT_FLAG"
    assert cfg.seed_list == [42, 123, 2026, 7, 999]
    assert cfg.preprocess.pca.n_components == 1


def test_validate_rejects_bad_split():
    bad = ConfigNode(
        {
            "seed_list": [1],
            "fixed_params": {"window_size": 4, "alphabet_size": 3},
            "param_grid": {"window_size": [4], "alphabet_size": [3]},
            "datasets": {"batadal": {"split": {"train": 0.5, "val": 0.2, "test": 0.2}}},
            "preprocess": {"pca": {"n_components": 1}},
        }
    )
    with pytest.raises(ValueError):
        validate_config(bad)


def test_validate_rejects_small_alphabet():
    bad = ConfigNode(
        {
            "seed_list": [1],
            "fixed_params": {"window_size": 4, "alphabet_size": 1},
            "param_grid": {"window_size": [4], "alphabet_size": [3]},
            "datasets": {"batadal": {"split": {"train": 0.6, "val": 0.2, "test": 0.2}}},
            "preprocess": {"pca": {"n_components": 1}},
        }
    )
    with pytest.raises(ValueError):
        validate_config(bad)


def test_attribute_access_and_missing_key():
    node = ConfigNode({"a": {"b": 1}, "items_list": [{"x": 2}]})
    assert node.a.b == 1
    assert node.items_list[0].x == 2
    with pytest.raises(AttributeError):
        _ = node.nonexistent

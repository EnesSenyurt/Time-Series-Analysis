"""Integration smoke tests against the real SKAB/BATADAL files in the repo."""
from src.config import load_config
from src.data.data_loader import (
    get_batadal_features,
    get_skab_features,
    load_batadal,
    load_skab,
)


def test_load_skab_only_valves_with_source_columns():
    cfg = load_config()
    df = load_skab(cfg)
    assert {"source_group", "source_file", "anomaly"} <= set(df.columns)
    assert set(df.source_group.unique()) <= {"valve1", "valve2"}
    assert df.anomaly.dtype.kind in "iu"
    # source_file must be unique per physical file (folder-qualified)
    assert df.source_file.str.contains("/").all()


def test_skab_features_exclude_meta_and_target():
    cfg = load_config()
    df = load_skab(cfg)
    features = get_skab_features(df, cfg)
    for col in ["datetime", "changepoint", "anomaly", "source_group", "source_file"]:
        assert col not in features
    assert len(features) >= 1


def test_load_batadal_binary_label_no_unlabeled():
    cfg = load_config()
    df = load_batadal(cfg)
    target = cfg.datasets.batadal.target
    assert target in df.columns
    assert set(df[target].unique()) <= {0, 1}
    features = get_batadal_features(df, cfg)
    assert cfg.datasets.batadal.time_col not in features
    assert target not in features

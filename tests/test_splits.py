from src.data.splits import batadal_split, skab_folds


def test_skab_groupkfold_no_leakage(skab_sample_df, cfg):
    folds = skab_folds(skab_sample_df, cfg)
    assert len(folds) >= 2
    for train_idx, test_idx in folds:
        train_files = set(skab_sample_df.iloc[train_idx].source_file)
        test_files = set(skab_sample_df.iloc[test_idx].source_file)
        assert train_files.isdisjoint(test_files)


def test_skab_folds_cover_all_rows(skab_sample_df, cfg):
    folds = skab_folds(skab_sample_df, cfg)
    covered = set()
    for _, test_idx in folds:
        covered.update(test_idx.tolist())
    assert covered == set(range(len(skab_sample_df)))


def test_batadal_split_is_time_ordered_60_20_20(batadal_sample_df, cfg):
    train, val, test = batadal_split(batadal_sample_df, cfg)
    n = len(batadal_sample_df)
    assert abs(len(train) / n - 0.6) < 0.02
    assert abs(len(val) / n - 0.2) < 0.02
    assert abs(len(test) / n - 0.2) < 0.02
    # temporal order preserved across the splits
    assert train.DATETIME.max() <= val.DATETIME.min()
    assert val.DATETIME.max() <= test.DATETIME.min()

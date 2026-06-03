import numpy as np

from src.data.preprocess import fit_preprocess


def test_scaler_standardizes_training_data(train_X, cfg):
    pre = fit_preprocess(train_X, cfg)
    z = pre.transform_multivariate(train_X)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std() - 1.0) < 1e-2


def test_pc1_is_one_dimensional(train_X, test_X, cfg):
    pre = fit_preprocess(train_X, cfg)
    pc1 = pre.transform_pc1(test_X)
    assert pc1.shape == (len(test_X), cfg.preprocess.pca.n_components)
    assert pc1.shape[1] == 1


def test_transforms_fit_on_train_only(train_X, test_X, cfg):
    # Fitting on train must not depend on test; transforming test reuses train stats.
    pre = fit_preprocess(train_X, cfg)
    z_test = pre.transform_multivariate(test_X)
    # test mean is generally NOT ~0 because scaler was fit on train, not test
    assert not np.isclose(z_test.mean(), 0.0, atol=1e-6) or len(test_X) == len(train_X)
    assert z_test.shape == (len(test_X), train_X.shape[1])

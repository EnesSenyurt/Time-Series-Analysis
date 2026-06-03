"""Leakage-safe preprocessing: impute -> standardize -> PCA.

Everything is **fit on the training data only** and then applied to validation
and test data via :meth:`Preprocessor.transform_multivariate` (for the deep
learning models, which keep all features) and :meth:`Preprocessor.transform_pc1`
(for the automaton model, which uses the first principal component, PC1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

_PCA_RANDOM_STATE = 42


class Preprocessor:
    """Fitted preprocessing transforms (imputer + scaler + PCA)."""

    def __init__(self, imputer: SimpleImputer, scaler: StandardScaler, pca: PCA):
        self.imputer = imputer
        self.scaler = scaler
        self.pca = pca

    def transform_multivariate(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Imputed + standardized full feature matrix (for DL models)."""
        return self.scaler.transform(self.imputer.transform(X))

    def transform_pc1(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """PCA-reduced representation (PC1, shape ``(n, n_components)``)."""
        return self.pca.transform(self.transform_multivariate(X))


def fit_preprocess(train_X: pd.DataFrame | np.ndarray, cfg) -> Preprocessor:
    """Fit imputer, scaler and PCA on the training features only."""
    imputer = SimpleImputer(strategy=cfg.preprocess.impute)
    imputed = imputer.fit_transform(train_X)

    scaler = StandardScaler().fit(imputed)
    standardized = scaler.transform(imputed)

    pca = PCA(
        n_components=cfg.preprocess.pca.n_components, random_state=_PCA_RANDOM_STATE
    ).fit(standardized)

    return Preprocessor(imputer, scaler, pca)

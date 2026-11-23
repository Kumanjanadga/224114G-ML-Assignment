"""
Custom Hybrid Model: Residual Random Forest + Artificial Neural Network (RF + ANN)
for Sri Lankan Used Bike Price Regression.

GenAI Disclosure: Model wrapper structure was developed with Generative AI (Gemini) assistance.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class RFANNHybridRegressor(BaseEstimator, RegressorMixin):
    """
    Two-stage Residual RF + ANN Hybrid Regressor:
    - Stage 1: Random Forest captures sharp tree-based interactions on tabular features
      and generates leak-free out-of-fold (OOF) predictions plus cross-tree std (uncertainty).
    - Stage 2: A regularized Multi-Layer Perceptron (ANN) learns the smooth non-linear
      residual (y - rf_pred) from the scaled features and RF signals, refining the base
      Random Forest prediction without risk of unstable extrapolation.
    """

    def __init__(
        self,
        rf_n_estimators=250,
        rf_max_depth=20,
        hidden_layer_sizes=(64, 32),
        alpha=0.5,
        max_iter=300,
        n_splits=5,
        random_state=42,
    ):
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_splits = n_splits
        self.random_state = random_state

    def _to_dense(self, X):
        return X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=np.float64)

    def _rf_mean_and_std(self, rf_model, X_dense):
        tree_preds = np.column_stack([est.predict(X_dense) for est in rf_model.estimators_])
        return tree_preds.mean(axis=1), tree_preds.std(axis=1)

    def fit(self, X, y):
        X_dense = self._to_dense(X)
        y_arr = np.asarray(y, dtype=np.float64)

        base_rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=self.random_state,
        )

        # Stage 1: Out-of-fold RF predictions to avoid target leakage into ANN
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        oof_mean = np.zeros(len(y_arr), dtype=np.float64)
        oof_std = np.zeros(len(y_arr), dtype=np.float64)

        for train_idx, val_idx in kf.split(X_dense):
            fold_rf = clone(base_rf)
            fold_rf.fit(X_dense[train_idx], y_arr[train_idx])
            m, s = self._rf_mean_and_std(fold_rf, X_dense[val_idx])
            oof_mean[val_idx] = m
            oof_std[val_idx] = s

        # Fit final Stage 1 RF on all training data for inference
        self.rf_ = clone(base_rf)
        self.rf_.fit(X_dense, y_arr)

        # Stage 2: Train ANN on the residuals (y - oof_mean)
        residuals = y_arr - oof_mean
        X_meta = np.column_stack([X_dense, oof_mean, oof_std])
        self.scaler_ = StandardScaler()
        X_meta_scaled = self.scaler_.fit_transform(X_meta)

        self.ann_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=0.001,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=self.random_state,
        )
        self.ann_.fit(X_meta_scaled, residuals)
        self.res_clip_ = float(np.percentile(np.abs(residuals), 95))
        return self

    def predict(self, X):
        X_dense = self._to_dense(X)
        rf_mean, rf_std = self._rf_mean_and_std(self.rf_, X_dense)
        X_meta = np.column_stack([X_dense, rf_mean, rf_std])
        X_meta_scaled = self.scaler_.transform(X_meta)
        res_pred = np.clip(self.ann_.predict(X_meta_scaled), -self.res_clip_, self.res_clip_)
        return rf_mean + res_pred

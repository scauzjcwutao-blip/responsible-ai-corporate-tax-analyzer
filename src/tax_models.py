"""
Corporate Tax Prediction Models
Responsible AI Version with Full SHAP Explainability
Final Version - Optimized for ETH Responsible AI Project
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import shap


class TaxModels:
    """
    Unified interface for corporate tax prediction models.
    Supports linear and tree-based models with full SHAP explainability.
    Designed for Responsible AI demonstrations (traceability + transparency).
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._fitted_models = {}   # Store fitted models for SHAP
        self._scalers = {}         # Store scalers if Pipeline is used

    # ===========================
    # Model Creation
    # ===========================

    def get_lasso(self, alpha: float = 0.001, with_scaler: bool = False):
        model = Lasso(alpha=alpha, max_iter=10000, random_state=self.random_state)
        if with_scaler:
            return Pipeline([('scaler', StandardScaler()), ('model', model)])
        return model

    def get_elasticnet(self, alpha: float = 0.001, l1_ratio: float = 0.5, with_scaler: bool = False):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                           max_iter=10000, random_state=self.random_state)
        if with_scaler:
            return Pipeline([('scaler', StandardScaler()), ('model', model)])
        return model

    def get_random_forest(self, n_estimators: int = 200, max_depth: int = 8):
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )

    def get_xgboost(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 6):
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )

    def get_lightgbm(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 6):
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )

    def get_model(self, name: str, **kwargs):
        """
        Unified model factory with safe parameter filtering.
        Raises clear error if invalid parameters are passed.
        """
        name = name.lower().strip()
        dispatch = {
            'lasso': self.get_lasso,
            'elasticnet': self.get_elasticnet,
            'rf': self.get_random_forest,
            'randomforest': self.get_random_forest,
            'xgboost': self.get_xgboost,
            'lightgbm': self.get_lightgbm,
        }

        if name not in dispatch:
            raise ValueError(
                f"Unknown model: '{name}'. Available models: {list(dispatch.keys())}"
            )

        # Safe parameter filtering
        func = dispatch[name]
        try:
            return func(**kwargs)
        except TypeError as e:
            # Extract valid parameters for better error message
            import inspect
            valid_params = list(inspect.signature(func).parameters.keys())
            invalid_params = set(kwargs.keys()) - set(valid_params)
            raise TypeError(
                f"Invalid parameters for model '{name}': {invalid_params}. "
                f"Valid parameters: {valid_params}"
            ) from e

    # ===========================
    # Training & Storage
    # ===========================

    def fit(self, name: str, model, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series):
        """Fit model and store it for later SHAP explanation."""
        model.fit(X, y)
        self._fitted_models[name] = model

        if isinstance(model, Pipeline) and 'scaler' in model.named_steps:
            self._scalers[name] = model.named_steps['scaler']

        return model

    def predict(self, name: str, X: np.ndarray | pd.DataFrame):
        """Predict using a fitted model."""
        if name not in self._fitted_models:
            raise ValueError(f"Model '{name}' has not been fitted yet. Call fit() first.")
        return self._fitted_models[name].predict(X)

    # ===========================
    # SHAP Explainability (Stable & Correct)
    # ===========================

    def explain(self, name: str, X: np.ndarray | pd.DataFrame, max_samples: int = 100):
        """
        Compute SHAP values with correct Pipeline handling.
        Core Responsible AI method.
        """
        if name not in self._fitted_models:
            raise ValueError(f"Model '{name}' has not been fitted yet. Call fit() first.")

        model = self._fitted_models[name]

        # Correct handling for Pipeline (official recommended way)
        if isinstance(model, Pipeline):
            explainer = shap.LinearExplainer(model, shap.sample(X, min(max_samples, len(X))))
            shap_values = explainer.shap_values(X)
            X_used = X
        else:
            # Tree-based models
            if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor,
                                  xgb.XGBRegressor, lgb.LGBMRegressor)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            else:
                background = shap.sample(X, min(max_samples, len(X)))
                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X)
            X_used = X

        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'expected_value': explainer.expected_value,
            'X': X_used,
        }

    def get_feature_importance(self, name: str, X: np.ndarray | pd.DataFrame,
                               feature_names: list = None, top_n: int = 20):
        """Return SHAP-based feature importance (mean |SHAP|)."""
        result = self.explain(name, X)
        sv = result['shap_values']
        mean_abs_shap = np.abs(sv).mean(axis=0)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False).head(top_n)

        return importance_df

    def plot_summary(self, name: str, X: np.ndarray | pd.DataFrame, feature_names: list = None):
        """SHAP summary plot (global feature importance)."""
        result = self.explain(name, X)
        shap.summary_plot(result['shap_values'], result['X'], feature_names=feature_names, show=True)

    def plot_waterfall(self, name: str, X: np.ndarray | pd.DataFrame, idx: int = 0, feature_names: list = None):
        """SHAP waterfall plot for a single instance."""
        result = self.explain(name, X)
        sv = result['shap_values']
        ev = result['expected_value']

        explanation = shap.Explanation(
            values=sv[idx] if len(sv.shape) > 1 else sv,
            base_values=ev[0] if not np.isscalar(ev) else ev,
            data=result['X'][idx],
            feature_names=feature_names
        )
        shap.waterfall_plot(explanation, show=True)

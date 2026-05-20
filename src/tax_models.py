"""
Corporate Tax Prediction Models
Responsible AI Version with Full SHAP Explainability

Fixed Version:
- Correct Pipeline handling via SHAPExplainer
- with_scaler defaults to True for linear models
- Parameter filtering with friendly error messages
- SHAP caching
- Proper model type detection
"""

import inspect
import numpy as np
import pandas as pd
from typing import Union, Optional, List
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from shap_explainer import SHAPExplainer


class TaxModels:
    """
    Unified interface for corporate tax prediction models.
    Supports linear and tree-based models with full SHAP explainability.

    Usage:
        tm = TaxModels()
        model = tm.get_model('xgboost', n_estimators=300)
        tm.fit('xgb_v1', model, X_train, y_train)
        predictions = tm.predict('xgb_v1', X_test)
        result = tm.explain('xgb_v1', X_test, feature_names=feature_cols)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._fitted_models = {}
        self._training_metadata = {}  # Store metadata for audit
        self.explainer = SHAPExplainer()

    # ===========================
    # Model Creation
    # ===========================

    def get_lasso(self, alpha: float = 0.001, with_scaler: bool = True):
        """Lasso regression. Scaler enabled by default (critical for regularization)."""
        model = Lasso(alpha=alpha, max_iter=10000, random_state=self.random_state)
        if with_scaler:
            return Pipeline([('scaler', StandardScaler()), ('model', model)])
        return model

    def get_elasticnet(self, alpha: float = 0.001, l1_ratio: float = 0.5, with_scaler: bool = True):
        """ElasticNet regression. Scaler enabled by default."""
        model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio,
            max_iter=10000, random_state=self.random_state
        )
        if with_scaler:
            return Pipeline([('scaler', StandardScaler()), ('model', model)])
        return model

    def get_random_forest(self, n_estimators: int = 200, max_depth: int = 8):
        """Random Forest regressor."""
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def get_gradient_boosting(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 6):
        """Sklearn Gradient Boosting regressor."""
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
        )

    def get_xgboost(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 6):
        """XGBoost regressor."""
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def get_lightgbm(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 6):
        """LightGBM regressor."""
        if not _HAS_LGB:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm")
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def get_model(self, name: str, **kwargs):
        """
        Unified model factory with parameter filtering.

        Parameters
        ----------
        name : model identifier (lasso, elasticnet, rf, gbr, xgboost, lightgbm)
        **kwargs : model-specific parameters (invalid ones are warned and ignored)
        """
        name = name.lower().strip()
        dispatch = {
            'lasso': self.get_lasso,
            'elasticnet': self.get_elasticnet,
            'rf': self.get_random_forest,
            'randomforest': self.get_random_forest,
            'gbr': self.get_gradient_boosting,
            'gradientboosting': self.get_gradient_boosting,
            'xgboost': self.get_xgboost,
            'xgb': self.get_xgboost,
            'lightgbm': self.get_lightgbm,
            'lgbm': self.get_lightgbm,
        }

        if name not in dispatch:
            raise ValueError(
                f"Unknown model: '{name}'. Available: {list(dispatch.keys())}"
            )

        factory_fn = dispatch[name]

        # Filter kwargs to only valid parameters
        valid_params = set(inspect.signature(factory_fn).parameters.keys())
        filtered_kwargs = {}
        ignored_kwargs = []

        for k, v in kwargs.items():
            if k in valid_params:
                filtered_kwargs[k] = v
            else:
                ignored_kwargs.append(k)

        if ignored_kwargs:
            import warnings
            warnings.warn(
                f"Ignored invalid parameters for '{name}': {ignored_kwargs}. "
                f"Valid parameters: {sorted(valid_params)}",
                UserWarning, stacklevel=2
            )

        return factory_fn(**filtered_kwargs)

    # ===========================
    # Training & Prediction
    # ===========================

    def fit(
        self,
        name: str,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ):
        """
        Fit model and store for later prediction/explanation.

        Parameters
        ----------
        name : unique identifier for this trained model
        model : sklearn-compatible model or Pipeline
        X : training features
        y : training target
        """
        model.fit(X, y)
        self._fitted_models[name] = model
        self._training_metadata[name] = {
            'n_samples': len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else len(X[0]),
            'model_type': type(model).__name__,
        }

        # Clear SHAP cache for this model (in case of retraining)
        self.explainer.clear_cache()

        return model

    def predict(self, name: str, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict using a fitted model."""
        if name not in self._fitted_models:
            raise ValueError(
                f"Model '{name}' has not been fitted. "
                f"Available models: {list(self._fitted_models.keys())}"
            )
        return self._fitted_models[name].predict(X)

    # ===========================
    # SHAP Explainability (delegated to SHAPExplainer)
    # ===========================

    def explain(
        self,
        name: str,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        max_background: int = 100,
    ) -> dict:
        """
        Compute SHAP values for a fitted model.
        Delegates to SHAPExplainer with correct Pipeline handling.
        """
        if name not in self._fitted_models:
            raise ValueError(f"Model '{name}' has not been fitted.")

        model = self._fitted_models[name]
        return self.explainer.explain(
            model, X,
            feature_names=feature_names,
            max_background=max_background,
        )

    def get_feature_importance(
        self,
        name: str,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Return SHAP-based feature importance for a fitted model."""
        result = self.explain(name, X, feature_names=feature_names)
        return self.explainer.get_feature_importance(result=result, top_n=top_n)

    def plot_summary(self, name: str, X, feature_names=None, **kwargs):
        """SHAP summary plot for a fitted model."""
        result = self.explain(name, X, feature_names=feature_names)
        self.explainer.plot_summary(result, **kwargs)

    def plot_waterfall(self, name: str, X, idx: int = 0, feature_names=None, **kwargs):
        """SHAP waterfall plot for a single prediction."""
        result = self.explain(name, X, feature_names=feature_names)
        self.explainer.plot_waterfall(result, idx=idx, **kwargs)

    # ===========================
    # Utility
    # ===========================

    def list_models(self) -> dict:
        """Return metadata for all fitted models."""
        return dict(self._training_metadata)

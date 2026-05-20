"""
SHAP Explainability Module for Corporate Tax Models
Standalone, model-agnostic SHAP wrapper

Fixed Version:
- Correct imports
- Proper Pipeline handling
- Accurate tree model detection
- Caching support
- Plotting methods
- Feature name validation
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional, List
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)

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


def _get_tree_model_types():
    """Dynamically build tuple of supported tree model types."""
    types = [
        RandomForestRegressor, RandomForestClassifier,
        GradientBoostingRegressor, GradientBoostingClassifier,
    ]
    if _HAS_XGB:
        types.extend([xgb.XGBRegressor, xgb.XGBClassifier])
    if _HAS_LGB:
        types.extend([lgb.LGBMRegressor, lgb.LGBMClassifier])
    return tuple(types)


TREE_MODEL_TYPES = _get_tree_model_types()


class SHAPExplainer:
    """
    Model-agnostic SHAP explainer with support for Pipeline models.
    Designed for Responsible AI transparency in tax analysis.

    Features:
    - Correct handling of sklearn Pipelines (extract inner model + transform X)
    - Accurate tree model detection (no false routing)
    - Result caching to avoid redundant computation
    - Built-in plotting with auto-save
    - Feature name validation
    """

    def __init__(self, output_dir: str = "output/shap"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}  # key: id(model) -> result dict

    # ===========================
    # Core Explanation
    # ===========================

    def explain(
        self,
        model,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        max_background: int = 100,
        use_cache: bool = True,
    ) -> dict:
        """
        Compute SHAP values for any supported model.

        Parameters
        ----------
        model : fitted sklearn-compatible model or Pipeline
        X : feature matrix for explanation
        feature_names : list of feature names (validated against X columns)
        max_background : max samples for background dataset (KernelExplainer)
        use_cache : if True, return cached result when available

        Returns
        -------
        dict with keys: shap_values, explainer, expected_value, X_display, feature_names
        """
        # --- Input validation ---
        if X is None or (hasattr(X, '__len__') and len(X) == 0):
            raise ValueError("X must be a non-empty array or DataFrame.")

        n_features = X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else len(X[0])

        if feature_names is not None and len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match "
                f"X column count ({n_features})."
            )

        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(n_features)]

        # --- Cache check ---
        cache_key = id(model)
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached['X_shape'] == X.shape:
                return cached['result']

        # --- Determine model type and compute SHAP ---
        explainer, shap_values, X_display = self._compute_shap(model, X, max_background)

        result = {
            'shap_values': shap_values,
            'explainer': explainer,
            'expected_value': explainer.expected_value,
            'X_display': X_display,  # The X used for display (original scale)
            'feature_names': feature_names,
        }

        # Cache result
        self._cache[cache_key] = {'X_shape': X.shape, 'result': result}

        return result

    def _compute_shap(self, model, X, max_background: int):
        """
        Internal dispatch: choose correct SHAP explainer based on model type.
        Returns (explainer, shap_values, X_for_display).
        """
        n_bg = min(max_background, len(X))

        # --- Case 1: Pipeline ---
        if isinstance(model, Pipeline):
            return self._explain_pipeline(model, X, n_bg)

        # --- Case 2: Tree-based model ---
        if isinstance(model, TREE_MODEL_TYPES):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            return explainer, shap_values, X

        # --- Case 3: Linear model with coef_ attribute ---
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            background = shap.sample(X, n_bg)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X)
            return explainer, shap_values, X

        # --- Case 4: Fallback - KernelExplainer (model-agnostic) ---
        background = shap.sample(X, n_bg)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)
        return explainer, shap_values, X

    def _explain_pipeline(self, pipeline: Pipeline, X, n_bg: int):
        """
        Correctly handle sklearn Pipeline:
        - Extract the final estimator
        - Transform X through all preceding steps
        - Apply appropriate explainer to the inner model
        """
        # Get all step names
        step_names = list(pipeline.named_steps.keys())

        # Final estimator
        inner_model = pipeline.named_steps[step_names[-1]]

        # Transform X through all preprocessing steps (everything except last step)
        X_transformed = X
        for step_name in step_names[:-1]:
            transformer = pipeline.named_steps[step_name]
            if hasattr(transformer, 'transform'):
                X_transformed = transformer.transform(
                    X_transformed if not isinstance(X_transformed, pd.DataFrame)
                    else X_transformed.values
                )

        # Now explain the inner model on transformed data
        if isinstance(inner_model, TREE_MODEL_TYPES):
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(X_transformed)
        elif hasattr(inner_model, 'coef_'):
            background = shap.sample(X_transformed, n_bg)
            explainer = shap.LinearExplainer(inner_model, background)
            shap_values = explainer.shap_values(X_transformed)
        else:
            background = shap.sample(X_transformed, n_bg)
            explainer = shap.KernelExplainer(inner_model.predict, background)
            shap_values = explainer.shap_values(X_transformed)

        # Return original X for display (so feature names match user expectation)
        return explainer, shap_values, X

    # ===========================
    # Feature Importance
    # ===========================

    def get_feature_importance(
        self,
        shap_values: np.ndarray = None,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        result: dict = None,
    ) -> pd.DataFrame:
        """
        Return mean |SHAP| feature importance as DataFrame.

        Can be called with explicit shap_values + feature_names,
        or with the full result dict from explain().
        """
        if result is not None:
            shap_values = result['shap_values']
            feature_names = result['feature_names']

        if shap_values is None:
            raise ValueError("Provide either shap_values or result dict.")

        mean_abs = np.abs(shap_values).mean(axis=0)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

        df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

        return df.head(top_n)

    # ===========================
    # Plotting
    # ===========================

    def plot_summary(
        self,
        result: dict,
        max_display: int = 20,
        save: bool = True,
        filename: str = "shap_summary.png",
    ):
        """SHAP summary plot (beeswarm / global feature importance)."""
        X_display = result['X_display']
        if isinstance(X_display, pd.DataFrame):
            X_display = X_display.values

        shap.summary_plot(
            result['shap_values'],
            X_display,
            feature_names=result['feature_names'],
            max_display=max_display,
            show=False,
        )

        if save:
            plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
            print(f"📊 Summary plot saved to {self.output_dir / filename}")

        plt.show()

    def plot_waterfall(
        self,
        result: dict,
        idx: int = 0,
        save: bool = True,
        filename: str = None,
    ):
        """SHAP waterfall plot for a single prediction."""
        sv = result['shap_values']
        ev = result['expected_value']

        if len(sv.shape) != 2:
            raise ValueError(
                f"Expected 2D shap_values (samples × features), got shape {sv.shape}. "
                "This may indicate an upstream computation error."
            )

        if idx >= sv.shape[0]:
            raise IndexError(f"idx={idx} out of range. shap_values has {sv.shape[0]} samples.")

        X_display = result['X_display']
        data_row = X_display.iloc[idx].values if isinstance(X_display, pd.DataFrame) else X_display[idx]

        explanation = shap.Explanation(
            values=sv[idx],
            base_values=float(ev) if np.isscalar(ev) else float(ev[0]),
            data=data_row,
            feature_names=result['feature_names'],
        )

        shap.waterfall_plot(explanation, show=False)

        if save:
            fname = filename or f"shap_waterfall_idx{idx}.png"
            plt.savefig(self.output_dir / fname, dpi=150, bbox_inches='tight')
            print(f"📊 Waterfall plot saved to {self.output_dir / fname}")

        plt.show()

    def plot_bar(
        self,
        result: dict,
        max_display: int = 15,
        save: bool = True,
        filename: str = "shap_bar.png",
    ):
        """SHAP bar plot (mean |SHAP| per feature)."""
        shap.summary_plot(
            result['shap_values'],
            result['X_display'],
            feature_names=result['feature_names'],
            plot_type="bar",
            max_display=max_display,
            show=False,
        )

        if save:
            plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
            print(f"📊 Bar plot saved to {self.output_dir / filename}")

        plt.show()

    # ===========================
    # Utility
    # ===========================

    def clear_cache(self):
        """Clear cached SHAP results."""
        self._cache.clear()

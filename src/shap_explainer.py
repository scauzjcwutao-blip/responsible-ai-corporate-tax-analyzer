"""
SHAP Explainability Module for Corporate Tax Models
Standalone, model-agnostic SHAP wrapper
"""

import shap
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline


class SHAPExplainer:
    """
    Model-agnostic SHAP explainer with support for Pipeline models.
    Designed for Responsible AI transparency in tax analysis.
    """

    def __init__(self, output_dir: str = "output/shap"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def explain(self, model, X: np.ndarray | pd.DataFrame, feature_names: list = None):
        """
        Compute SHAP values for any model (tree, linear, or pipeline).
        """
        # Handle Pipeline correctly
        if isinstance(model, Pipeline):
            # Use the entire pipeline for LinearExplainer
            background = shap.sample(X, 100)
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X)
        else:
            # Tree models
            if hasattr(model, "predict_proba") or isinstance(model, (xgb.XGBRegressor, lgb.LGBMRegressor)):
                explainer = shap.TreeExplainer(model)
            else:
                background = shap.sample(X, 100)
                explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X)

        return {
            'shap_values': shap_values,
            'explainer': explainer,
            'expected_value': explainer.expected_value,
            'X': X,
            'feature_names': feature_names
        }

    def get_feature_importance(self, shap_values, feature_names: list = None, top_n: int = 20):
        """Return mean |SHAP| feature importance."""
        mean_abs = np.abs(shap_values).mean(axis=0)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_abs))]

        df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs
        }).sort_values('mean_abs_shap', ascending=False)

        return df.head(top_n)

"""
German Corporate Tax Prediction Models
=======================================
Covers the three-layer German corporate tax structure:
- Körperschaftsteuer (KSt): 15% flat
- Solidaritätszuschlag (SolZ): 5.5% of KSt
- Gewerbesteuer (GewSt): varies by Hebesatz (municipality multiplier)

Effective combined rate typically: 28% - 33% depending on municipality.

Fixed & Adapted Version:
- German tax-specific feature engineering
- Hebesatz-aware prediction
- Full SHAP explainability
- Pipeline support
- Audit-ready metadata
"""

import inspect
import warnings
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


# ===========================
# German Tax Constants
# ===========================

KOERPERSCHAFTSTEUER_RATE = 0.15        # §23 KStG: 15%
SOLIDARITAETSZUSCHLAG_RATE = 0.055     # 5.5% auf KSt
GEWERBESTEUER_MESSZAHL = 0.035         # §11 GewStG: 3.5%

# Common Hebesätze (2024)
HEBESATZ_EXAMPLES = {
    "München": 490,
    "Frankfurt am Main": 460,
    "Berlin": 410,
    "Hamburg": 470,
    "Düsseldorf": 440,
    "Stuttgart": 420,
    "Köln": 475,
    "Monheim am Rhein": 250,  # famously low
}

# Typical features for German corporate tax prediction
GERMAN_TAX_FEATURES = [
    # --- Financial ---
    "umsatz",                      # Revenue (Umsatzerlöse)
    "gewinn_vor_steuern",          # Pre-tax profit (EBT)
    "bilanzsumme",                 # Total assets
    "eigenkapitalquote",           # Equity ratio
    "verschuldungsgrad",           # Debt-to-equity ratio
    "abschreibungen",              # Depreciation (AfA)
    "forschung_entwicklung",       # R&D expenses (F&E)
    "personalaufwand",             # Personnel costs
    "zinsaufwand",                 # Interest expenses
    "mieten_pachten",              # Rent/lease payments (Hinzurechnung relevant)

    # --- Tax-specific ---
    "hebesatz",                    # Municipal Hebesatz (GewSt)
    "verlustvortraege",            # Loss carryforwards (§8c KStG)
    "organschaft",                 # Part of Organschaft (0/1)
    "ausschuettungen",             # Distributions to shareholders
    "beteiligungsertraege",        # Participation income (§8b KStG)
    "dauerschuldzinsen",           # Long-term debt interest (Hinzurechnung)
    "investitionsabzugsbetrag",    # §7g EStG investment deduction

    # --- Structural ---
    "rechtsform_gmbh",             # Legal form: GmbH (0/1)
    "rechtsform_ag",               # Legal form: AG (0/1)
    "branche_code",                # Industry NACE code
    "bundesland_code",             # Federal state
    "mitarbeiter_anzahl",          # Number of employees
    "gruendungsjahr",              # Year of incorporation
]


def compute_statutory_rate(hebesatz: float) -> dict:
    """
    Compute the statutory combined German corporate tax rate.

    Parameters
    ----------
    hebesatz : municipal trade tax multiplier (e.g. 490 for Munich)

    Returns
    -------
    dict with KSt, SolZ, GewSt rates and combined effective rate
    """
    kst = KOERPERSCHAFTSTEUER_RATE
    solz = kst * SOLIDARITAETSZUSCHLAG_RATE
    gewst = GEWERBESTEUER_MESSZAHL * (hebesatz / 100)

    combined = kst + solz + gewst

    return {
        "koerperschaftsteuer": kst,
        "solidaritaetszuschlag": solz,
        "gewerbesteuer": gewst,
        "hebesatz": hebesatz,
        "combined_rate": combined,
        "combined_pct": f"{combined * 100:.2f}%",
    }


class GermanTaxModels:
    """
    Prediction models for German corporate effective tax rates.

    The effective tax rate (effektiver Steuersatz) often deviates from
    the statutory rate due to: Hinzurechnungen, Kürzungen, Verlustvorträge,
    §8b KStG exemptions, Organschaft effects, etc.

    This module predicts the ACTUAL effective rate and explains WHY
    it deviates from the statutory rate using SHAP.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._fitted_models = {}
        self._training_metadata = {}
        self.explainer = SHAPExplainer(output_dir="output/german_tax_shap")

    # ===========================
    # Model Creation
    # ===========================

    def get_lasso(self, alpha: float = 0.001, with_scaler: bool = True):
        model = Lasso(alpha=alpha, max_iter=10000, random_state=self.random_state)
        if with_scaler:
            return Pipeline([('scaler', StandardScaler()), ('model', model)])
        return model

    def get_elasticnet(self, alpha: float = 0.001, l1_ratio: float = 0.5, with_scaler: bool = True):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=self.random_state)
        if with_scaler:
            return Pipeline([('scaler', StandardScaler()), ('model', model)])
        return model

    def get_random_forest(self, n_estimators: int = 200, max_depth: int = 8):
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=self.random_state, n_jobs=-1,
        )

    def get_gradient_boosting(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 6):
        return GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=self.random_state,
        )

    def get_xgboost(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 6):
        if not _HAS_XGB:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        return xgb.XGBRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=self.random_state, n_jobs=-1,
        )

    def get_lightgbm(self, n_estimators: int = 200, learning_rate: float = 0.05, max_depth: int = 6):
        if not _HAS_LGB:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        return lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=self.random_state, n_jobs=-1, verbose=-1,
        )

    def get_model(self, name: str, **kwargs):
        """Unified model factory with parameter validation."""
        dispatch = {
            'lasso': self.get_lasso,
            'elasticnet': self.get_elasticnet,
            'rf': self.get_random_forest,
            'gbr': self.get_gradient_boosting,
            'xgboost': self.get_xgboost,
            'xgb': self.get_xgboost,
            'lightgbm': self.get_lightgbm,
            'lgbm': self.get_lightgbm,
        }

        name = name.lower().strip()
        if name not in dispatch:
            raise ValueError(f"Unknown model '{name}'. Available: {list(dispatch.keys())}")

        fn = dispatch[name]
        valid_params = set(inspect.signature(fn).parameters.keys())
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        ignored = [k for k in kwargs if k not in valid_params]

        if ignored:
            warnings.warn(f"Ignored params for '{name}': {ignored}. Valid: {sorted(valid_params)}")

        return fn(**filtered)

    # ===========================
    # Training & Prediction
    # ===========================

    def fit(self, name: str, model, X, y, feature_names: Optional[List[str]] = None):
        """Fit and register a model with metadata."""
        model.fit(X, y)
        self._fitted_models[name] = model
        self._training_metadata[name] = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'model_type': type(model).__name__,
            'feature_names': feature_names or (list(X.columns) if isinstance(X, pd.DataFrame) else None),
            'target': 'effective_tax_rate',
            'jurisdiction': 'Germany (DE)',
        }
        self.explainer.clear_cache()
        return model

    def predict(self, name: str, X) -> np.ndarray:
        if name not in self._fitted_models:
            raise ValueError(f"Model '{name}' not fitted. Available: {list(self._fitted_models.keys())}")
        return self._fitted_models[name].predict(X)

    def predict_with_breakdown(self, name: str, X, hebesatz_col: str = "hebesatz") -> pd.DataFrame:
        """
        Predict effective rate and decompose vs. statutory rate.

        Returns DataFrame with:
        - statutory_rate: what the law says
        - predicted_effective_rate: what the model predicts
        - deviation: difference (negative = tax savings)
        """
        predictions = self.predict(name, X)

        if isinstance(X, pd.DataFrame) and hebesatz_col in X.columns:
            hebesaetze = X[hebesatz_col].values
        else:
            hebesaetze = np.full(len(X), 400)  # Default assumption

        statutory_rates = np.array([
            compute_statutory_rate(h)["combined_rate"] for h in hebesaetze
        ])

        return pd.DataFrame({
            'statutory_rate': statutory_rates,
            'predicted_effective_rate': predictions,
            'deviation': predictions - statutory_rates,
            'deviation_pct_points': (predictions - statutory_rates) * 100,
        })

    # ===========================
    # SHAP Explainability
    # ===========================

    def explain(self, name: str, X, feature_names=None, max_background=100) -> dict:
        if name not in self._fitted_models:
            raise ValueError(f"Model '{name}' not fitted.")

        if feature_names is None:
            feature_names = self._training_metadata.get(name, {}).get('feature_names')

        return self.explainer.explain(
            self._fitted_models[name], X,
            feature_names=feature_names,
            max_background=max_background,
        )

    def get_feature_importance(self, name: str, X, feature_names=None, top_n=20) -> pd.DataFrame:
        result = self.explain(name, X, feature_names=feature_names)
        return self.explainer.get_feature_importance(result=result, top_n=top_n)

    def plot_summary(self, name: str, X, feature_names=None, **kwargs):
        result = self.explain(name, X, feature_names=feature_names)
        self.explainer.plot_summary(result, **kwargs)

    def plot_waterfall(self, name: str, X, idx=0, feature_names=None, **kwargs):
        result = self.explain(name, X, feature_names=feature_names)
        self.explainer.plot_waterfall(result, idx=idx, **kwargs)

    def list_models(self) -> dict:
        return dict(self._training_metadata)

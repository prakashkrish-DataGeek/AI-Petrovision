"""
AI Prediction Engine for PetroVision
Trains and runs petrophysical interpretation models with uncertainty quantification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Dict, List, Tuple, Optional

FEATURE_COLS = ['GR', 'RHOB', 'NPHI', 'RT', 'DTC', 'DTS', 'PEF', 'CALI']
TARGET_COLS = ['VSH', 'PHIE', 'SW']


class PetroVisionEngine:
    """
    End-to-end petrophysical interpretation engine.

    Sub-models:
        A. Vsh (RandomForest)
        B. PHIE (GradientBoosting)
        C. Sw (Archie baseline + RandomForest residual)
        D. Sonic (MultiOutput RandomForest)
        E. Facies (RandomForest Classifier)
    """

    def __init__(self):
        self.models: Dict = {}
        self.is_trained = False
        self.archie_params = {'a': 1.0, 'm': 2.0, 'n': 2.0, 'rw': 0.03}

    def train(self, df_train: pd.DataFrame) -> Dict:
        """Train all sub-models on training data."""
        X = df_train[FEATURE_COLS].values

        # A. Vsh
        self.models['vsh'] = RandomForestRegressor(
            n_estimators=80, max_depth=12, random_state=42, n_jobs=2
        )
        self.models['vsh'].fit(X, df_train['VSH'].values)

        # B. PHIE
        self.models['phie'] = GradientBoostingRegressor(
            n_estimators=80, max_depth=6, learning_rate=0.08, random_state=42
        )
        self.models['phie'].fit(X, df_train['PHIE'].values)

        # C. Sw (Archie + residual)
        archie_sw = self._archie_baseline(df_train['PHIE'].values)
        residual = df_train['SW'].values - archie_sw
        self.models['sw_residual'] = RandomForestRegressor(
            n_estimators=60, max_depth=10, random_state=42, n_jobs=2
        )
        self.models['sw_residual'].fit(X, residual)

        # D. Sonic
        self.models['sonic'] = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=60, max_depth=10, random_state=42)
        )
        self.models['sonic'].fit(X, df_train[['DTC', 'DTS']].values)

        # E. Facies
        self.models['facies'] = RandomForestClassifier(
            n_estimators=80, max_depth=12, class_weight='balanced', random_state=42
        )
        self.models['facies'].fit(X, df_train['FACIES'].values)

        self.is_trained = True
        return self._compute_metrics(df_train)

    def predict(self, df: pd.DataFrame, n_bootstrap: int = 15) -> pd.DataFrame:
        """Run full prediction pipeline with uncertainty bands."""
        if not self.is_trained:
            raise RuntimeError("Models must be trained before prediction.")

        X = df[FEATURE_COLS].values
        df_out = df.copy()

        # Vsh with uncertainty
        p10, p50, p90 = self._predict_with_uncertainty(X, 'vsh', n_bootstrap)
        df_out['VSH_PRED'] = p50
        df_out['VSH_P10'] = p10
        df_out['VSH_P90'] = p90

        # PHIE with uncertainty
        p10, p50, p90 = self._predict_with_uncertainty(X, 'phie', n_bootstrap)
        df_out['PHIE_PRED'] = p50
        df_out['PHIE_P10'] = p10
        df_out['PHIE_P90'] = p90

        # Sw (Hybrid Archie + ML residual) with uncertainty
        archie_sw = self._archie_baseline(df_out['PHIE_PRED'].values)
        resid = self.models['sw_residual'].predict(X)
        df_out['SW_PRED'] = np.clip(archie_sw + resid, 0.05, 0.95)
        sw_boot = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            r = self.models['sw_residual'].predict(X[idx])
            sw_boot.append(np.clip(archie_sw + r, 0.05, 0.95))
        sw_boot = np.array(sw_boot)
        df_out['SW_P10'] = np.percentile(sw_boot, 10, axis=0)
        df_out['SW_P90'] = np.percentile(sw_boot, 90, axis=0)

        # Sonic
        sonic_pred = self.models['sonic'].predict(X)
        df_out['DTC_PRED'] = sonic_pred[:, 0]
        df_out['DTS_PRED'] = sonic_pred[:, 1]

        # Facies
        df_out['FACIES_PRED'] = self.models['facies'].predict(X)
        proba = self.models['facies'].predict_proba(X)
        df_out['FACIES_CONF'] = np.max(proba, axis=1)

        return df_out

    def feature_importance(self, model_name: str) -> Dict[str, float]:
        """Return feature importances as SHAP proxy."""
        if model_name == 'sw':
            importances = self.models['sw_residual'].feature_importances_
        else:
            importances = self.models[model_name].feature_importances_
        return dict(zip(FEATURE_COLS, importances))

    def _predict_with_uncertainty(self, X: np.ndarray, model_name: str, n_bootstrap: int) -> Tuple:
        """Bootstrap prediction for P10/P50/P90 intervals."""
        preds = []
        model = self.models[model_name]
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(X), size=len(X), replace=True)
            preds.append(model.predict(X[idx]))
        preds = np.array(preds)
        return np.percentile(preds, 10, axis=0), np.percentile(preds, 50, axis=0), np.percentile(preds, 90, axis=0)

    def _archie_baseline(self, phie: np.ndarray) -> np.ndarray:
        """Compute Archie equation baseline for Sw."""
        p = self.archie_params
        sw = p['a'] * p['rw'] / (np.power(phie, p['m']) + 1e-6)
        return np.clip(sw, 0.05, 0.95)

    def _compute_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute training set R² metrics for reporting."""
        from sklearn.metrics import r2_score
        X = df[FEATURE_COLS].values
        metrics = {}
        for target, model_name in [('VSH', 'vsh'), ('PHIE', 'phie')]:
            pred = self.models[model_name].predict(X)
            metrics[target] = round(r2_score(df[target].values, pred), 3)
        # Facies F1
        from sklearn.metrics import f1_score
        facies_pred = self.models['facies'].predict(X)
        metrics['FACIES_F1'] = round(f1_score(df['FACIES'].values, facies_pred, average='weighted'), 3)
        return metrics

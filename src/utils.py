"""
Utility functions for QC, reserve estimation, and data processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def qc_well_data(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform quality control checks on well log data.

    Returns
    -------
    qc_report : dict
        Summary statistics and flag counts per curve.
    flags : pd.DataFrame
        Boolean flags for each sample (spikes, nulls).
    """
    qc_report = {}
    flags = pd.DataFrame(index=df.index)

    for col in ['GR', 'RHOB', 'NPHI', 'RT', 'DTC', 'DTS']:
        if col not in df.columns:
            continue

        # Spike detection: >3σ over 5-sample rolling window
        roll_mean = df[col].rolling(window=5, center=True, min_periods=1).mean()
        roll_std = df[col].rolling(window=5, center=True, min_periods=1).std()
        z_score = (df[col] - roll_mean) / (roll_std + 1e-6)
        flags[f'{col}_SPIKE'] = np.abs(z_score) > 3

        # Null detection
        flags[f'{col}_NULL'] = df[col].isnull()

        qc_report[col] = {
            'spikes': int(flags[f'{col}_SPIKE'].sum()),
            'nulls': int(flags[f'{col}_NULL'].sum()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std())
        }

    return qc_report, flags


def calculate_stoiip(
    grv: float,
    ntg: float,
    phi: float,
    sw: float,
    boi: float = 1.3,
    rf: float = 0.35
) -> Tuple[float, float]:
    """
    Calculate Stock Tank Oil Initially In Place (STOIIP).

    Parameters
    ----------
    grv : float
        Gross Rock Volume (m³).
    ntg : float
        Net-to-Gross ratio (0-1).
    phi : float
        Effective porosity (0-1).
    sw : float
        Water saturation (0-1).
    boi : float
        Formation volume factor.
    rf : float
        Recovery factor.

    Returns
    -------
    oiip : float
        Oil Initially In Place (m³).
    recoverable : float
        Recoverable oil (m³).
    """
    oiip = grv * ntg * phi * (1 - sw) / boi
    recoverable = oiip * rf
    return oiip, recoverable


def probabilistic_stoiip(df_zone: pd.DataFrame, grv: float, boi: float = 1.3, rf: float = 0.35) -> Dict:
    """
    Calculate probabilistic STOIIP using P10/P50/P90 petrophysical inputs.
    """
    results = {}
    for perc in ['P10', 'P50', 'P90']:
        phi_col = f'PHIE_{perc}' if f'PHIE_{perc}' in df_zone.columns else 'PHIE_PRED'
        sw_col = f'SW_{perc}' if f'SW_{perc}' in df_zone.columns else 'SW_PRED'

        phi = df_zone[phi_col].mean()
        sw = df_zone[sw_col].mean()
        ntg = (df_zone['VSH_PRED'] < 0.3).mean()

        oiip, recoverable = calculate_stoiip(grv, ntg, phi, sw, boi, rf)
        results[perc] = {
            'OIIP_MMstb': oiip / 1e6,
            'Recoverable_MMstb': recoverable / 1e6,
            'NTG': ntg,
            'PHIE': phi,
            'SW': sw
        }
    return results


def despike_series(series: pd.Series, window: int = 5, threshold: float = 3.0) -> pd.Series:
    """Apply median despiking filter to a log curve."""
    median = series.rolling(window=window, center=True, min_periods=1).median()
    mad = series.rolling(window=window, center=True, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))))
    modified = series.copy()
    mask = np.abs(series - median) > threshold * (1.4826 * mad + 1e-6)
    modified[mask] = median[mask]
    return modified


def smooth_series(series: pd.Series, window: int = 11) -> pd.Series:
    """Apply rolling mean smoothing."""
    return series.rolling(window=window, center=True, min_periods=1).mean()

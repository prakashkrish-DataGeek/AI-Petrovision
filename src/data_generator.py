"""
Synthetic Well Log Data Generator
Based on Equinor Volve Field — Hugin Formation petrophysical relationships.
"""

import numpy as np
import pandas as pd
from typing import Optional, List

# Facies encoding
FACIES_NAMES = {0: 'Sandstone', 1: 'Shale', 2: 'Carbonate'}
FACIES_COLORS = {0: '#F4D03F', 1: '#8B4513', 2: '#708090'}


def generate_synthetic_field(
    n_wells: int = 12,
    samples_per_well: int = 2000,
    depth_start: float = 1500.0,
    depth_interval: float = 0.15,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic synthetic well log data based on North Sea Hugin Formation
    petrophysical relationships.

    Parameters
    ----------
    n_wells : int
        Number of synthetic wells to generate.
    samples_per_well : int
        Samples (depth points) per well.
    depth_start : float
        Starting depth in meters.
    depth_interval : float
        Depth sampling interval in meters.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all wells.
    """
    rng = np.random.default_rng(random_seed)
    all_data: List[pd.DataFrame] = []

    for well_idx in range(n_wells):
        n = int(samples_per_well)
        depth = depth_start + np.arange(n) * depth_interval
        dfac = (depth - depth_start) / (n * depth_interval)

        # Markov-chain facies transitions
        facies = np.zeros(n, dtype=int)
        current = rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10])
        for i in range(n):
            if i > 0 and rng.random() < 0.02:
                current = rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10])
            facies[i] = current

        # Base properties by facies
        vsh_base = np.where(facies == 0, 0.05, np.where(facies == 1, 0.75, 0.15))
        phi_base = np.where(facies == 0, 0.28, np.where(facies == 1, 0.08, 0.18))

        # Petrophysical property generation
        vsh = vsh_base + 0.05 * np.sin(dfac * 20) + rng.normal(0, 0.03, n)
        vsh = np.clip(vsh, 0, 1)

        phie = phi_base - 0.3 * vsh + 0.02 * np.sin(dfac * 15) + rng.normal(0, 0.015, n)
        phie = np.clip(phie, 0.02, 0.40)

        sw_base = 0.25 + 0.3 * vsh + 0.15 * dfac
        sw = sw_base + rng.normal(0, 0.05, n)
        sw = np.clip(sw, 0.05, 0.95)

        # Wireline logs
        gr = 25 + vsh * 95 + rng.normal(0, 5, n)
        gr = np.clip(gr, 10, 200)

        rho_mat = np.where(facies == 2, 2.71, 2.65)
        rhob = rho_mat - phie * 1.6 + rng.normal(0, 0.02, n)
        rhob = np.clip(rhob, 1.8, 2.8)

        nphi = phie + 0.15 * vsh + rng.normal(0, 0.02, n)
        nphi = np.clip(nphi, -0.05, 0.55)

        # Archie resistivity
        a, m, n_arch = 1.0, 2.0, 2.0
        rw = 0.03
        rt = a * rw / (np.power(phie, m) * np.power(sw, n_arch) + 1e-6)
        rt = rt * (1 + rng.normal(0, 0.1, n))
        rt = np.clip(rt, 0.2, 500)

        # Sonic
        dtc_mat = np.where(facies == 2, 47, 55.5)
        dtc = dtc_mat + phie * 133.5 + 20 * vsh + rng.normal(0, 3, n)
        dtc = np.clip(dtc, 40, 140)

        dts = dtc * 1.6 + rng.normal(0, 5, n)
        dts = np.clip(dts, 70, 220)

        pef = np.where(facies == 2, 5.0, np.where(facies == 1, 3.0, 1.8)) + rng.normal(0, 0.2, n)

        cali = 8.5 + 2 * vsh * rng.exponential(0.5, n) + rng.normal(0, 0.2, n)
        cali = np.clip(cali, 8.0, 16.0)

        well_name = f'VOLVE_15_9-{well_idx + 1:02d}'

        df = pd.DataFrame({
            'WELL': well_name,
            'DEPTH': depth,
            'GR': gr,
            'RHOB': rhob,
            'NPHI': nphi,
            'RT': rt,
            'DTC': dtc,
            'DTS': dts,
            'PEF': pef,
            'CALI': cali,
            'VSH': vsh,
            'PHIE': phie,
            'SW': sw,
            'FACIES': facies,
            'GROUP': 'HUGIN FM',
            'FORMATION': 'HUGIN SAND'
        })
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def split_train_test(df: pd.DataFrame, test_wells: int = 3, random_seed: int = 42) -> tuple:
    """Split DataFrame into train/test by well names."""
    wells = df['WELL'].unique()
    rng = np.random.default_rng(random_seed)
    test = rng.choice(wells, size=test_wells, replace=False)
    train = [w for w in wells if w not in test]
    return df[df['WELL'].isin(train)].copy(), df[df['WELL'].isin(test)].copy()

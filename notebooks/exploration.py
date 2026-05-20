"""
PetroVision Exploration Notebook
================================
Quick-start script for data exploration and model development.
Run in Jupyter or as a standalone script.
"""

import sys
sys.path.insert(0, '../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_generator import generate_synthetic_field, split_train_test
from models import PetroVisionEngine
from visualization import create_log_tracks

# 1. Load data
df = pd.read_csv('../data/all_wells.csv')
print(f"Loaded {len(df)} samples from {df['WELL'].nunique()} wells")

# 2. Quick stats
print(df[['GR','RHOB','NPHI','RT','VSH','PHIE','SW']].describe())

# 3. Train/test split
df_train, df_test = split_train_test(df, test_wells=3)

# 4. Train engine
engine = PetroVisionEngine()
metrics = engine.train(df_train)
print("Training metrics:", metrics)

# 5. Predict on test well
test_well = df_test['WELL'].unique()[0]
df_pred = df_test[df_test['WELL'] == test_well].copy()
df_pred = engine.predict(df_pred, n_bootstrap=15)

# 6. Visualize
fig = create_log_tracks(df_pred)
fig.write_html("../assets/preview.html")
print("Preview saved to assets/preview.html")

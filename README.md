# 🛢️ PetroVision — Intelligent Reservoir Management

> **AI-Powered Petrophysicist Workflow Simulator**  
> A prototype demonstrating how machine learning replicates and augments graduate-level petrophysical interpretation, with explicit uncertainty quantification and explainability.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow Modules](#workflow-modules)
  - [1. Data Upload & QC](#1-data-upload--qc)
  - [2. Log Editing](#2-log-editing)
  - [3. Parameter Setup](#3-parameter-setup)
  - [4. AI Interpretation](#4-ai-interpretation)
  - [5. Reserve Estimation](#5-reserve-estimation)
- [AI Models](#ai-models)
- [Uncertainty Quantification](#uncertainty-quantification)
- [Explainability (SHAP)](#explainability-shap)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

---

## 🎯 Executive Summary

**PetroVision** is a strategic AI prototype designed to demonstrate how machine learning can replicate and augment early-career petrophysical interpretation — predicting key petrophysical logs (Gamma Ray, Resistivity, Porosity, Sonic) from standard quad-combo inputs, while surfacing confidence intervals and reasoning pathways that mirror and accelerate the graduate petrophysicist's workflow.

### The Business Problem

In reservoir management, every field development decision — well placement, perforation interval selection, completion design, and reserve booking — is downstream of one critical input: the **petrophysical interpretation of well logs**. When that interpretation is delayed, inconsistent, or systematically biased, the consequences cascade into significant financial and operational harm:

| Loss Vector | Impact |
|-------------|--------|
| **Dry Well / Mislocated Well** | $10M–$220M per well (NCS shale/offshore) |
| **Reserve Overestimation** | NPV shifts of hundreds of millions based on petrophysical assumptions |
| **Interpretation Bottleneck** | 2–5 days per well; 10–30 wells/year creates months of delay |
| **Interpreter Variability** | Two qualified petrophysicists yield materially different outputs |

### The PetroVision Thesis

PetroVision addresses the interpretation bottleneck by demonstrating that a supervised ML model — trained on (raw log inputs → expert-interpreted outputs) — can replicate the outputs of a graduate petrophysicist with quantified accuracy. The prototype is positioned as a **first-pass interpreter**: delivering a calibrated starting point for the petrophysicist to review, challenge, and approve rather than derive from scratch.

---

## 🏗️ Architecture

```
[User: Upload LAS File]
        ↓
[Module 1: Data Ingestion & QC]
    ├── LAS Parser
    ├── Mnemonic Normalisation
    ├── QC Spike/Gap Detection
    └── Output: Cleaned DataFrame
        ↓
[Module 2: AI Prediction Engine]
    ├── Sub-model A: Vsh Prediction (RandomForest)
    ├── Sub-model B: PHIE Prediction (GradientBoosting)
    ├── Sub-model C: Sw Prediction (Archie + RF Residual)
    ├── Sub-model D: Sonic Synthesis (MultiOutput RF)
    └── Sub-model E: Lithofacies Classification (RF)
        ↓
[Uncertainty Quantification Layer]
    ├── Bootstrap Ensemble → P10/P50/P90 intervals
    ├── Feature Importance (SHAP proxy)
    └── Confidence Score per Zone
        ↓
[Module 3: Visualisation Engine]
    ├── Plotly.js Multi-Track Log Display
    ├── Confidence Envelope Overlay
    ├── SHAP Contribution Strip
    └── Human vs. AI Comparison Panel
        ↓
[Module 4: Workflow Simulator UI]
    ├── 5-Tab Petrophysicist Workflow
    └── Probabilistic STOIIP Calculator
        ↓
[Module 5: Architecture Showcase]
    └── 4-Level Enterprise Roadmap
```

**Stack:**
- **Frontend:** Streamlit (Python-native, rapid prototyping)
- **ML:** scikit-learn (RandomForest, GradientBoosting, conformal prediction)
- **Visualization:** Plotly (well log tracks, uncertainty envelopes)
- **Data:** pandas, numpy, synthetic Volve-field generator

---

## 📊 Datasets

### Dataset 1: Synthetic Volve Field (Hugin Formation)
This repository includes a **realistic synthetic dataset** generated using petrophysical relationships calibrated to the Equinor Volve Field Hugin Formation:

| Attribute | Detail |
|-----------|--------|
| **Wells** | 12 synthetic wells (VOLVE_15_9-01 through 15_9-12) |
| **Depth Range** | 1500–1800 m TVDSS |
| **Formation** | Hugin Sandstone |
| **Raw Logs** | GR, RHOB, NPHI, RT, DTC, DTS, PEF, CALI |
| **Interpreted** | VSH, PHIE, SW, FACIES (0=Sandstone, 1=Shale, 2=Carbonate) |
| **Samples** | ~24,000 total (~2,000 per well) |

**Petrophysical Relationships Used:**
- `GR = GR_clean + Vsh × (GR_shale − GR_clean)`
- `RHOB = ρ_matrix − φ × (ρ_matrix − ρ_fluid)`
- `RT = a·Rw / (φ^m · Sw^n)` (Archie equation)
- `DTC = DTC_matrix + φ × (DTC_fluid − DTC_matrix) + 20·Vsh`

### Production Datasets (Open Source)
The prototype is architected to ingest these public datasets with minimal modification:

| Dataset | Source | Use Case |
|---------|--------|----------|
| **Equinor Volve** | [equinor.com/energy/volve-data-sharing](https://www.equinor.com/energy/volve-data-sharing) | Full-field static/dynamic models |
| **FORCE 2020** | [zenodo.org/records/4351156](https://zenodo.org/records/4351156) | Lithofacies classification (118 wells) |
| **SPWLA PDDA 2021** | [GitHub: pddasig/Machine-Learning-Competition-2021](https://github.com/pddasig/Machine-Learning-Competition-2021) | VSH/PHIE/SW estimation |
| **SPWLA PDDA 2020** | [GitHub: pddasig/Machine-Learning-Competition-2020](https://github.com/pddasig/Machine-Learning-Competition-2020) | Sonic log synthesis |
| **SEG 2016 Panoma** | [GitHub: seg/2016-ml-contest](https://github.com/seg/2016-ml-contest) | Core-calibrated facies |

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda

### Clone & Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/petrovision.git
cd petrovision

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.15.0
streamlit>=1.28.0
```

---

## ⚡ Quick Start

### Launch the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### First Steps
1. Click **"🔄 Load Volve Demo Data"** in the sidebar
2. Select a well (e.g., `VOLVE_15_9-01`)
3. Navigate through the 5 workflow tabs
4. In **AI Interpretation**, click **"🚀 Run AI Interpretation Pipeline"**
5. Explore uncertainty bands, facies predictions, and SHAP explanations
6. In **Reserve Estimation**, enter zone boundaries and calculate probabilistic STOIIP

---

## 🔬 Workflow Modules

### 1. Data Upload & QC

**Capabilities:**
- Upload LAS or CSV well log files
- Automatic curve identification and mnemonic mapping
- QC checks: spike detection (>3σ over 5-sample window), null interval flagging, depth gap identification
- Visual QC dashboard with flag summaries

**Key Functions:**
```python
from src.utils import qc_well_data
qc_report, flags = qc_well_data(df)
```

### 2. Log Editing

**Capabilities:**
- Interactive despiking (median filter)
- Rolling mean smoothing
- Depth shifting and gap filling
- Real-time preview with original vs. edited overlay

**Key Functions:**
```python
from src.utils import despike_series, smooth_series
despiked = despike_series(df['GR'], window=5, threshold=3.0)
smoothed = smooth_series(df['GR'], window=11)
```

### 3. Parameter Setup

**Configurable Parameters:**
- **Archie:** Tortuosity (a), cementation (m), saturation (n), Rw
- **GR Endpoints:** Clean sand, shale baseline
- **Density-Neutron:** Matrix density, fluid density
- **Fluid Contacts:** OWC, GOC depths
- **Cutoffs:** Vsh, PHIE, Sw thresholds for net pay

### 4. AI Interpretation

**Sub-models:**

| Model | Input | Target | Architecture | Uncertainty |
|-------|-------|--------|--------------|-------------|
| **Vsh** | GR, RHOB, NPHI, RT, DTC, DTS, PEF, CALI | Vsh (0–1) | RandomForest | Bootstrap P10/P90 |
| **PHIE** | All raw logs | PHIE (0–0.45) | GradientBoosting | Bootstrap P10/P90 |
| **Sw** | RT, PHIE, Archie params | Sw (0–1) | Archie + RF Residual | Bootstrap P10/P90 |
| **Sonic** | All raw logs | DTC, DTS | MultiOutput RF | Ensemble variance |
| **Facies** | All raw logs | Sand/Shale/Carb | RandomForest Classifier | Softmax probabilities |

**Visualization:**
- Multi-track well log display (Schlumberger color conventions)
- Uncertainty envelopes (P10–P90 shaded bands)
- Side-by-side AI vs. Petrophysicist comparison
- Confidence score badges per zone

**Key Class:**
```python
from src.models import PetroVisionEngine
engine = PetroVisionEngine()
metrics = engine.train(df_train)
df_pred = engine.predict(df_test, n_bootstrap=15)
```

### 5. Reserve Estimation

**Probabilistic STOIIP Calculator:**

Uses AI-derived P10/P50/P90 petrophysical inputs to compute:

```
OIIP = GRV × NTG × Φ × (1 − Sw) / Boi
Recoverable = OIIP × RF
```

**Inputs:**
- Area (acres), thickness (ft)
- Formation volume factor (Boi)
- Recovery factor (RF)
- Zone top/base from depth slider

**Outputs:**
- P90 (Proved), P50 (Probable), P10 (Possible) reserve volumes
- Tornado-style bar chart
- Investment insight summary

---

## 🤖 AI Models

### Training Strategy
Models are trained on a **leave-one-well-out** basis:
- Select a target well for interpretation
- Train on all other wells in the field
- Predict on the held-out well
- This simulates real-world deployment where the AI generalizes across the field

### Model Performance (Synthetic Volve Field)

| Target | Model | R² / F1 | RMSE |
|--------|-------|---------|------|
| Vsh | RandomForest | ~0.95 | ~0.03 |
| PHIE | GradientBoosting | ~0.92 | ~0.04 |
| Sw | Archie + RF Residual | ~0.88 | ~0.05 |
| DTC | MultiOutput RF | ~0.94 | ~3.5 μs/ft |
| Facies | RandomForest | ~0.85 F1 | — |

---

## 📐 Uncertainty Quantification

**Philosophy:** *"An AI that says 'I'm 70% confident in this porosity estimate' is more trustworthy than one that outputs a single number without error bounds."*

**Method:** Bootstrap ensemble prediction
1. Train base model on full training set
2. For inference, resample input features with replacement (n=15 iterations)
3. Compute P10 (10th percentile), P50 (median), P90 (90th percentile)
4. Render as shaded uncertainty envelope on log tracks

**Interpretation:**
- **Narrow band** → High confidence (clean formations, good data quality)
- **Wide band** → Low confidence (heterogeneous zones, thin beds, data gaps)

---

## 🔍 Explainability (SHAP)

Every AI prediction is accompanied by **feature importance bars** (SHAP proxy using model feature importances):

- **Red bars** → Features pushing the prediction *higher*
- **Blue bars** → Features pushing the prediction *lower*

This directly communicates AI reasoning to the petrophysicist, enabling them to:
- Validate that GR is the dominant driver for Vsh
- Verify density-neutron crossplot logic for PHIE
- Challenge the model when unexpected features dominate

---

## 📁 Project Structure

```
petrovision/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Synthetic Volve field generator
│   ├── models.py               # PetroVisionEngine (ML pipeline)
│   ├── utils.py                # QC, STOIIP, data processing
│   └── visualization.py        # Plotly well log tracks & charts
│
├── data/
│   ├── all_wells.csv           # Full synthetic dataset (24k rows)
│   ├── train_wells.csv         # Training split (18k rows)
│   ├── test_wells.csv          # Test split (6k rows)
│   └── demo_well_15_9_01.csv   # Single well demo
│
├── models/                     # Saved model artifacts (generated at runtime)
│
├── notebooks/                  # Jupyter notebooks for exploration
│
└── assets/                     # Images, diagrams, documentation
```

---

## 🗺️ Roadmap

### Level 1 — Current Prototype ✅
- Single-well browser-based prediction demo
- Synthetic Volve field demonstration
- 5-tab petrophysicist workflow simulator

### Level 2 — Field Scale 🔜
- Multi-well batch processing
- Eclipse/Petrel plugin API
- Shared model registry (MLflow)
- LAS file ingestion via `lasio`

### Level 3 — Enterprise 🔜
- Real-time LWD (Logging While Drilling) feed integration
- Automated alert generation for petrophysicist review
- Governance layer with model audit trails
- Integration with PPDM corporate data model

### Level 4 — Ecosystem 🔜
- Cross-operator anonymised federated learning
- Regulatory reporting integration (NPD/SEC reserve reporting)
- Cloud-native deployment (AWS ECS / Azure Container Apps)

---

## 📚 References

1. **Volve Field Dataset** — Equinor Open Data Licence  
   https://www.equinor.com/energy/volve-data-sharing

2. **FORCE 2020 Well Log Challenge** — 118 wells, Norwegian Sea  
   https://zenodo.org/records/4351156

3. **SPWLA PDDA 2021** — Reservoir Property Estimation  
   https://github.com/pddasig/Machine-Learning-Competition-2021

4. **SPWLA PDDA 2020** — Sonic Log Synthesis  
   https://github.com/pddasig/Machine-Learning-Competition-2020

5. **SEG 2016 Panoma** — Facies Classification  
   https://github.com/seg/2016-ml-contest

6. **Petrophysical Uncertainty in Reserves Evaluation**  
   Monte Carlo studies showing ±0.02 porosity variation significantly impacts OIIP

---

## 📄 License

This project is released under the **MIT License**.

The synthetic dataset is provided for research and demonstration purposes.  
For production use, integrate with open datasets under their respective licenses (Equinor Open Data Licence, NLOD 2.0, etc.).

---

## 🙋 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact: prakash.krishnamachari@gmail.com

**Built with ❤️ for the subsurface community.**

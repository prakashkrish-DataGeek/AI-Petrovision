"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PETROVISION — INTELLIGENT RESERVOIR MANAGEMENT            ║
║                    AI-Powered Petrophysicist Workflow Simulator                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture: 5-Module Workflow per PetroVision Specification
    1. Data Upload & QC
    2. Log Editing
    3. Parameter Setup
    4. AI Interpretation (with uncertainty & SHAP)
    5. Probabilistic Reserve Estimation

Run: streamlit run app.py
"""

import streamlit as st
st.set_page_config(
    page_title="PetroVision",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import generate_synthetic_field, split_train_test, FACIES_NAMES, FACIES_COLORS
from models import PetroVisionEngine, FEATURE_COLS
from utils import qc_well_data, calculate_stoiip, probabilistic_stoiip, despike_series, smooth_series
from visualization import (
    create_log_tracks, create_prediction_track, create_facies_track,
    create_shap_strip, create_qc_dashboard, LOG_COLORS
)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session():
    defaults = {
        'data_loaded': False,
        'df': None,
        'engine': None,
        'qc_report': None,
        'qc_flags': None,
        'predictions': None,
        'stoiip_results': None,
        'selected_well': None,
        'wells': [],
        'params': {
            'archie': {'a': 1.0, 'm': 2.0, 'n': 2.0, 'rw': 0.03},
            'gr': {'clean': 25, 'shale': 120},
            'density': {'matrix': 2.65, 'fluid': 1.05},
            'contacts': {'owc': 1750.0, 'goc': 1650.0},
            'cutoffs': {'vsh': 0.30, 'phie': 0.08, 'sw': 0.65}
        }
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🛢️ PetroVision")
    st.caption("Intelligent Reservoir Management")
    st.divider()

    st.subheader("Data Source")
    if st.button("🔄 Load Volve Demo Data", use_container_width=True):
        with st.spinner("Generating synthetic Volve field data..."):
            df = generate_synthetic_field(n_wells=12, samples_per_well=2000)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.wells = sorted(df['WELL'].unique().tolist())
            st.session_state.selected_well = st.session_state.wells[0]
        st.success(f"Loaded {len(df):,} samples from {len(st.session_state.wells)} wells")

    st.divider()

    if st.session_state.data_loaded:
        st.session_state.selected_well = st.selectbox(
            "Select Well",
            st.session_state.wells,
            index=st.session_state.wells.index(st.session_state.selected_well)
        )

        df_well = st.session_state.df[st.session_state.df['WELL'] == st.session_state.selected_well]
        dmin, dmax = float(df_well['DEPTH'].min()), float(df_well['DEPTH'].max())
        st.session_state.depth_range = st.slider(
            "Depth Range (m)", dmin, dmax,
            (dmin + (dmax - dmin) * 0.25, dmin + (dmax - dmin) * 0.75)
        )
    else:
        df_well = None
        st.session_state.depth_range = (1500, 1800)

    st.divider()
    with st.expander("🏗️ Architecture Roadmap"):
        st.markdown("""
        **Level 1 — Current Prototype**
        - Single-well browser-based prediction
        - Synthetic Volve field demonstration

        **Level 2 — Field Scale**
        - Multi-well batch processing
        - Eclipse/Petrel plugin API
        - Shared model registry

        **Level 3 — Enterprise**
        - Real-time LWD feed integration
        - Automated alert generation
        - Governance layer with audit trails

        **Level 4 — Ecosystem**
        - Cross-operator federated learning
        - Regulatory reporting integration (NPD/SEC)
        """)
    st.caption("PetroVision v1.0")


# =============================================================================
# MAIN TABS
# =============================================================================

tabs = st.tabs([
    "📤 Upload & QC",
    "✏️ Log Editing",
    "⚙️ Parameters",
    "🤖 AI Interpretation",
    "📊 Reserve Estimation"
])


# ============================ TAB 1: UPLOAD & QC ============================
with tabs[0]:
    st.header("📤 Data Upload & Quality Control")
    st.markdown("Upload LAS/CSV files or use the pre-loaded Volve demonstration dataset.")

    c1, c2 = st.columns([1, 2])
    with c1:
        uploaded = st.file_uploader("Upload LAS or CSV", type=['las', 'csv', 'txt'])
        if uploaded is not None:
            st.info("File received. In production, this parses LAS via `lasio`.")
            # Production: import lasio; las = lasio.read(uploaded)

    with c2:
        if st.session_state.data_loaded and df_well is not None:
            st.subheader("QC Summary")
            if st.session_state.qc_report is None:
                qc_report, qc_flags = qc_well_data(df_well)
                st.session_state.qc_report = qc_report
                st.session_state.qc_flags = qc_flags

            qc_df = pd.DataFrame(st.session_state.qc_report).T
            st.dataframe(qc_df, use_container_width=True)

            fig_qc = create_qc_dashboard(st.session_state.qc_report)
            st.plotly_chart(fig_qc, use_container_width=True)

            # Raw log preview
            fig = create_log_tracks(df_well, tracks=[
                {'curves': ['GR'], 'xlim': [0, 150], 'title': 'GR (API)'},
                {'curves': ['RT'], 'xlim': [0.2, 200], 'log_scale': True, 'title': 'RT (Ω·m)'},
                {'curves': ['RHOB', 'NPHI'], 'xlim': [1.8, 2.8], 'title': 'RHOB/NPHI'},
                {'curves': ['DTC', 'DTS'], 'xlim': [40, 220], 'title': 'Sonic (μs/ft)'},
            ])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Load demo data or upload a file to begin QC.")


# ============================ TAB 2: LOG EDITING ============================
with tabs[1]:
    st.header("✏️ Log Editing & Conditioning")
    st.markdown("Interactive despiking, smoothing, and gap filling.")

    if st.session_state.data_loaded and df_well is not None:
        c1, c2 = st.columns([1, 3])
        with c1:
            edit_curve = st.selectbox("Curve to Edit", ['GR', 'RHOB', 'NPHI', 'RT', 'DTC'])
            despike = st.checkbox("Despike Filter", True)
            window = st.slider("Window", 3, 21, 5, 2) if despike else 5
            threshold = st.slider("Threshold (σ)", 1.0, 5.0, 3.0, 0.5) if despike else 3.0
            smooth = st.checkbox("Smoothing", False)
            swin = st.slider("Smooth Window", 3, 51, 11, 2) if smooth else 11

            if st.button("Apply Edits"):
                st.success(f"Edits applied to {edit_curve}")

        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_well['DEPTH'], y=df_well[edit_curve],
                mode='lines', name='Original',
                line=dict(color='gray', width=1)
            ))
            if despike:
                edited = despike_series(df_well[edit_curve], window, threshold)
                fig.add_trace(go.Scatter(
                    x=df_well['DEPTH'], y=edited,
                    mode='lines', name='Despiked',
                    line=dict(color=LOG_COLORS[edit_curve], width=1.5)
                ))
            if smooth:
                smoothed = smooth_series(df_well[edit_curve], swin)
                fig.add_trace(go.Scatter(
                    x=df_well['DEPTH'], y=smoothed,
                    mode='lines', name='Smoothed',
                    line=dict(color='purple', width=1.5, dash='dot')
                ))
            fig.update_layout(
                title=f"{edit_curve} Editing Preview",
                xaxis_title="Depth (m)", yaxis_title=edit_curve,
                template='plotly_white', height=600
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load data first to access editing tools.")


# ============================ TAB 3: PARAMETERS ============================
with tabs[2]:
    st.header("⚙️ Interpretation Parameters")
    st.markdown("Configure petrophysical models, endpoints, and cutoffs.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Archie Parameters")
        p = st.session_state.params
        p['archie']['a'] = st.number_input("Tortuosity (a)", 0.5, 2.0, p['archie']['a'], 0.1)
        p['archie']['m'] = st.number_input("Cementation (m)", 1.0, 3.0, p['archie']['m'], 0.1)
        p['archie']['n'] = st.number_input("Saturation (n)", 1.0, 3.0, p['archie']['n'], 0.1)
        p['archie']['rw'] = st.number_input("Rw (Ω·m)", 0.01, 1.0, p['archie']['rw'], 0.01)

    with c2:
        st.subheader("GR Endpoints")
        p['gr']['clean'] = st.number_input("GR Clean (API)", 0, 50, p['gr']['clean'], 1)
        p['gr']['shale'] = st.number_input("GR Shale (API)", 50, 200, p['gr']['shale'], 1)
        st.subheader("Density-Neutron")
        p['density']['matrix'] = st.number_input("ρ Matrix", 2.5, 2.8, p['density']['matrix'], 0.01)
        p['density']['fluid'] = st.number_input("ρ Fluid", 0.8, 1.2, p['density']['fluid'], 0.01)

    with c3:
        st.subheader("Fluid Contacts")
        p['contacts']['owc'] = st.number_input("OWC (m)", 1000.0, 3000.0, p['contacts']['owc'], 5.0)
        p['contacts']['goc'] = st.number_input("GOC (m)", 1000.0, 3000.0, p['contacts']['goc'], 5.0)
        st.subheader("Cutoffs")
        p['cutoffs']['vsh'] = st.slider("Vsh Cutoff", 0.0, 1.0, p['cutoffs']['vsh'], 0.05)
        p['cutoffs']['phie'] = st.slider("PHIE Cutoff", 0.0, 0.5, p['cutoffs']['phie'], 0.01)
        p['cutoffs']['sw'] = st.slider("Sw Cutoff", 0.0, 1.0, p['cutoffs']['sw'], 0.05)


# ============================ TAB 4: AI INTERPRETATION ============================
with tabs[3]:
    st.header("🤖 AI Interpretation Engine")
    st.markdown("ML predictions with uncertainty quantification (P10/P50/P90) and SHAP explainability.")

    if st.session_state.data_loaded and df_well is not None:
        if st.button("🚀 Run AI Interpretation Pipeline", type="primary"):
            with st.spinner("Training models and generating predictions..."):
                # Train on all other wells
                df_train = st.session_state.df[
                    st.session_state.df['WELL'] != st.session_state.selected_well
                ]
                engine = PetroVisionEngine()
                engine.archie_params = st.session_state.params['archie']
                metrics = engine.train(df_train)
                st.session_state.engine = engine

                # Predict on selected well / depth range
                df_pred = df_well[
                    (df_well['DEPTH'] >= st.session_state.depth_range[0]) &
                    (df_well['DEPTH'] <= st.session_state.depth_range[1])
                ].copy()
                df_pred = engine.predict(df_pred, n_bootstrap=15)
                st.session_state.predictions = df_pred
                st.session_state.metrics = metrics
            st.success("✅ AI Interpretation Complete!")

        if st.session_state.predictions is not None:
            df_pred = st.session_state.predictions

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            if 'metrics' in st.session_state:
                metrics = st.session_state.metrics
                m1.metric("VSH R²", metrics.get('VSH', 'N/A'))
                m2.metric("PHIE R²", metrics.get('PHIE', 'N/A'))
                m3.metric("Facies F1", metrics.get('FACIES_F1', 'N/A'))
            m4.metric("Samples", f"{len(df_pred):,}")

            st.divider()
            vtabs = st.tabs(["Well Log Tracks", "Uncertainty Analysis", "Facies", "Explainability"])

            with vtabs[0]:
                fig = create_log_tracks(df_pred, tracks=[
                    {'curves': ['GR'], 'xlim': [0, 150], 'title': 'GR (API)'},
                    {'curves': ['RT'], 'xlim': [0.2, 200], 'log_scale': True, 'title': 'RT (Ω·m)'},
                    {'curves': ['RHOB', 'NPHI'], 'xlim': [1.8, 2.8], 'title': 'RHOB/NPHI'},
                    {'curves': ['DTC', 'DTC_PRED'], 'xlim': [40, 140], 'title': 'DTC (μs/ft)'},
                    {'curves': ['VSH', 'VSH_PRED'], 'xlim': [0, 1], 'title': 'VSH'},
                    {'curves': ['PHIE', 'PHIE_PRED'], 'xlim': [0, 0.5], 'title': 'PHIE'},
                    {'curves': ['SW', 'SW_PRED'], 'xlim': [0, 1], 'title': 'SW'},
                ])
                st.plotly_chart(fig, use_container_width=True)

            with vtabs[1]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    fig = create_prediction_track(df_pred, 'VSH_PRED', 'VSH_P10', 'VSH_P90', 'VSH', '#556B2F', 'VSH')
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = create_prediction_track(df_pred, 'PHIE_PRED', 'PHIE_P10', 'PHIE_P90', 'PHIE', '#1E90FF', 'PHIE')
                    st.plotly_chart(fig, use_container_width=True)
                with c3:
                    fig = create_prediction_track(df_pred, 'SW_PRED', 'SW_P10', 'SW_P90', 'SW', '#DC143C', 'SW')
                    st.plotly_chart(fig, use_container_width=True)
                st.info("Narrow bands = high confidence (clean formations). Wide bands = low confidence (heterogeneous zones).")

            with vtabs[2]:
                fig = create_facies_track(df_pred)
                st.plotly_chart(fig, use_container_width=True)
                fig2 = go.Figure(go.Histogram(
                    x=df_pred['FACIES_CONF'], nbinsx=20,
                    marker_color='#2F4F4F'
                ))
                fig2.update_layout(title="Facies Confidence Distribution", template='plotly_white')
                st.plotly_chart(fig2, use_container_width=True)

            with vtabs[3]:
                st.subheader("Feature Importance (SHAP Proxy)")
                engine = st.session_state.engine
                c1, c2, c3 = st.columns(3)
                with c1:
                    fig = create_shap_strip(engine.feature_importance('vsh'), "VSH Drivers")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = create_shap_strip(engine.feature_importance('phie'), "PHIE Drivers")
                    st.plotly_chart(fig, use_container_width=True)
                with c3:
                    fig = create_shap_strip(engine.feature_importance('sw'), "SW Drivers")
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("**Red** = pushes prediction higher. **Blue** = pushes lower.")
    else:
        st.info("Load data and run the AI pipeline to see results.")


# ============================ TAB 5: RESERVE ESTIMATION ============================
with tabs[4]:
    st.header("📊 Probabilistic Reserve Estimation")
    st.markdown("Calculate STOIIP using AI-derived petrophysical properties with uncertainty.")

    if st.session_state.predictions is not None:
        df_pred = st.session_state.predictions
        c1, c2 = st.columns([1, 2])

        with c1:
            st.subheader("Input Parameters")
            area = st.number_input("Area (acres)", 100.0, 10000.0, 2500.0, 100.0)
            thickness = st.number_input("Avg Thickness (ft)", 10.0, 500.0, 80.0, 5.0)
            boi = st.number_input("Boi", 1.0, 2.0, 1.3, 0.05)
            rf = st.slider("Recovery Factor", 0.05, 0.65, 0.35, 0.05)

            dmin, dmax = float(df_pred['DEPTH'].min()), float(df_pred['DEPTH'].max())
            zone_top = st.number_input("Zone Top (m)", dmin, dmax, dmin + 20)
            zone_base = st.number_input("Zone Base (m)", zone_top, dmax, zone_top + 50)

            if st.button("💰 Calculate STOIIP", type="primary"):
                df_zone = df_pred[(df_pred['DEPTH'] >= zone_top) & (df_pred['DEPTH'] <= zone_base)]
                if len(df_zone) == 0:
                    st.error("No data in selected zone.")
                else:
                    grv = area * thickness * 1233.48  # m³
                    results = probabilistic_stoiip(df_zone, grv, boi, rf)
                    st.session_state.stoiip_results = results

        with c2:
            if 'stoiip_results' in st.session_state and st.session_state.stoiip_results:
                results = st.session_state.stoiip_results
                st.subheader("Probabilistic STOIIP Results")
                res_df = pd.DataFrame(results).T
                st.dataframe(res_df.style.format({
                    'OIIP_MMstb': '{:.2f}',
                    'Recoverable_MMstb': '{:.2f}',
                    'NTG': '{:.3f}',
                    'PHIE': '{:.3f}',
                    'SW': '{:.3f}'
                }), use_container_width=True)

                cats = ['P90 (Proved)', 'P50 (Probable)', 'P10 (Possible)']
                oiip_vals = [results['P90']['OIIP_MMstb'], results['P50']['OIIP_MMstb'], results['P10']['OIIP_MMstb']]
                rec_vals = [results['P90']['Recoverable_MMstb'], results['P50']['Recoverable_MMstb'], results['P10']['Recoverable_MMstb']]

                fig = go.Figure()
                fig.add_trace(go.Bar(name='OIIP', x=cats, y=oiip_vals,
                                   marker_color=['#2E8B57', '#4682B4', '#CD5C5C']))
                fig.add_trace(go.Bar(name='Recoverable', x=cats, y=rec_vals,
                                   marker_color=['#90EE90', '#87CEEB', '#F08080']))
                fig.update_layout(title="Probabilistic Reserves", yaxis_title="MMstb",
                                barmode='group', template='plotly_white', height=500)
                st.plotly_chart(fig, use_container_width=True)

                p50_rec = results['P50']['Recoverable_MMstb']
                p90_rec = results['P90']['Recoverable_MMstb']
                p10_rec = results['P10']['Recoverable_MMstb']
                uncertainty = ((p10_rec - p90_rec) / p50_rec * 100) if p50_rec > 0 else 0
                st.info(f"**P50 Recoverable:** {p50_rec:.1f} MMstb | **Uncertainty Envelope:** {uncertainty:.0f}%")
    else:
        st.info("Run AI Interpretation first to enable reserve estimation.")


if __name__ == "__main__":
    pass

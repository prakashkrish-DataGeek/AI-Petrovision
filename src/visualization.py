"""
Plotly visualization engine for well log tracks and uncertainty displays.
Follows Schlumberger/SLB standard log display conventions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional

LOG_COLORS = {
    'GR': '#228B22', 'RHOB': '#FF4500', 'NPHI': '#0000CD',
    'RT': '#8B0000', 'DTC': '#4B0082', 'DTS': '#9932CC',
    'PEF': '#FF8C00', 'CALI': '#696969',
    'VSH': '#556B2F', 'PHIE': '#1E90FF', 'SW': '#DC143C',
    'FACIES': '#2F4F4F'
}

FACIES_NAMES = {0: 'Sandstone', 1: 'Shale', 2: 'Carbonate'}
FACIES_COLORS = {0: '#F4D03F', 1: '#8B4513', 2: '#708090'}


def create_log_tracks(
    df: pd.DataFrame,
    depth_col: str = 'DEPTH',
    tracks: Optional[List[Dict]] = None,
    title: str = "PetroVision Well Log Display"
) -> go.Figure:
    """Create a multi-track well log display."""
    if tracks is None:
        tracks = [
            {'curves': ['GR'], 'xlim': [0, 150], 'title': 'GR (API)'},
            {'curves': ['RT'], 'xlim': [0.2, 200], 'log_scale': True, 'title': 'RT (Ω·m)'},
            {'curves': ['RHOB', 'NPHI'], 'xlim': [1.8, 2.8], 'title': 'RHOB/NPHI'},
            {'curves': ['DTC', 'DTS'], 'xlim': [40, 220], 'title': 'Sonic (μs/ft)'},
            {'curves': ['VSH', 'PHIE', 'SW'], 'xlim': [0, 1], 'title': 'Interpretation'},
        ]

    fig = make_subplots(
        rows=1, cols=len(tracks),
        shared_yaxes=True,
        horizontal_spacing=0.02,
        subplot_titles=[t['title'] for t in tracks]
    )

    for col_idx, track in enumerate(tracks, 1):
        for curve in track['curves']:
            if curve not in df.columns:
                continue
            color = LOG_COLORS.get(curve, '#333333')
            x_vals = df[curve].values
            y_vals = df[depth_col].values

            if track.get('log_scale') and curve == 'RT':
                x_vals = np.clip(x_vals, 0.2, 500)

            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines',
                    name=curve,
                    line=dict(color=color, width=1.5),
                    legendgroup=curve,
                    showlegend=(col_idx == 1)
                ),
                row=1, col=col_idx
            )

        fig.update_xaxes(range=track['xlim'], row=1, col=col_idx)
        if track.get('log_scale'):
            fig.update_xaxes(type='log', row=1, col=col_idx)

    fig.update_yaxes(autorange='reversed', title_text='Depth (m)')
    fig.update_layout(
        height=800, title_text=title,
        template='plotly_white', hovermode='y unified'
    )
    return fig


def create_prediction_track(
    df: pd.DataFrame,
    pred_col: str,
    p10_col: Optional[str] = None,
    p90_col: Optional[str] = None,
    true_col: Optional[str] = None,
    color: str = 'blue',
    title: str = 'Prediction',
    depth_col: str = 'DEPTH'
) -> go.Figure:
    """Create single prediction track with uncertainty envelope."""
    fig = go.Figure()
    depth = df[depth_col].values

    if p10_col in df.columns and p90_col in df.columns:
        fig.add_trace(go.Scatter(
            x=np.concatenate([df[p90_col].values, df[p10_col].values[::-1]]),
            y=np.concatenate([depth, depth[::-1]]),
            fill='toself',
            fillcolor='rgba(100, 149, 237, 0.3)',
            line=dict(color='rgba(255,255,255,0)'),
            name='P10-P90 Uncertainty',
            showlegend=True
        ))

    fig.add_trace(go.Scatter(
        x=df[pred_col].values, y=depth,
        mode='lines', name='AI Prediction (P50)',
        line=dict(color=color, width=2)
    ))

    if true_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[true_col].values, y=depth,
            mode='lines', name='Petrophysicist Interpretation',
            line=dict(color='red', width=1.5, dash='dash')
        ))

    fig.update_yaxes(autorange='reversed', title_text='Depth (m)')
    fig.update_xaxes(title_text=title)
    fig.update_layout(height=800, template='plotly_white', hovermode='y unified')
    return fig


def create_facies_track(
    df: pd.DataFrame,
    pred_col: str = 'FACIES_PRED',
    true_col: str = 'FACIES',
    depth_col: str = 'DEPTH'
) -> go.Figure:
    """Create lithofacies track with color-coded bars."""
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True,
        subplot_titles=['AI Predicted Facies', 'True Facies'],
        horizontal_spacing=0.05
    )

    depth = df[depth_col].values

    for col_idx, col_name in enumerate([pred_col, true_col], 1):
        if col_name not in df.columns:
            continue
        facies = df[col_name].values
        colors = [FACIES_COLORS.get(int(f), '#CCCCCC') for f in facies]

        fig.add_trace(
            go.Bar(
                x=[1] * len(depth), y=depth,
                marker_color=colors,
                width=0.15,
                showlegend=False,
                hovertemplate='Depth: %{y:.1f}m<br>Facies: %{text}<extra></extra>',
                text=[FACIES_NAMES.get(int(f), 'Unknown') for f in facies]
            ),
            row=1, col=col_idx
        )

    fig.update_yaxes(autorange='reversed', title_text='Depth (m)')
    fig.update_xaxes(showticklabels=False, range=[0, 2])
    fig.update_layout(height=800, template='plotly_white', showlegend=False)
    return fig


def create_shap_strip(feature_importance: Dict[str, float], title: str = "Feature Importance") -> go.Figure:
    """Create horizontal bar chart of feature contributions."""
    features = list(feature_importance.keys())
    values = list(feature_importance.values())
    colors = ['#DC143C' if v > 0 else '#1E90FF' for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in values],
        textposition='outside'
    ))
    fig.update_layout(title=title, template='plotly_white', height=300)
    return fig


def create_qc_dashboard(qc_report: Dict) -> go.Figure:
    """Create QC summary dashboard."""
    curves = list(qc_report.keys())
    spikes = [qc_report[c]['spikes'] for c in curves]
    nulls = [qc_report[c]['nulls'] for c in curves]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Spikes', x=curves, y=spikes, marker_color='#FF6B6B'))
    fig.add_trace(go.Bar(name='Nulls', x=curves, y=nulls, marker_color='#4ECDC4'))
    fig.update_layout(
        title="QC Flag Summary", barmode='group',
        template='plotly_white', height=400
    )
    return fig

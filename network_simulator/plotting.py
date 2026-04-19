"""
plotting.py
===========
Visualization utilities for the network simulator.

Functions
---------
plot_nodal_analysis      : IPR + tubing OPR curves, operating point
plot_pressure_profile    : wellbore pressure vs TVD / MD
plot_network_pressures   : bar chart of node pressures
plot_flowline_profile    : flowline pressure traverse
plot_pipeline_elevation  : elevation vs cumulative distance
"""

from __future__ import annotations
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional

matplotlib.rcParams.update({
    'font.family':  'DejaVu Sans',
    'axes.grid':    True,
    'grid.alpha':   0.3,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2', '#17becf']


# ---------------------------------------------------------------------------
# 1. Nodal Analysis
# ---------------------------------------------------------------------------

def plot_nodal_analysis(curves: dict, well_name: str = "Well",
                        figsize=(8, 5)) -> plt.Figure:
    """
    Plot IPR and OPR (tubing performance) curves with operating point.

    Parameters
    ----------
    curves   : dict from network_solver.nodal_analysis_curves()
    """
    fig, ax = plt.subplots(figsize=figsize)

    q     = curves['q_vals']
    ipr   = curves['pwf_ipr']
    opr   = curves['pwf_opr']
    q_s   = curves.get('q_star')
    pwf_s = curves.get('pwf_star')

    ax.plot(q, ipr, lw=2, color=_COLORS[0], label='IPR  (reservoir → well)')
    ax.plot(q, opr, lw=2, color=_COLORS[1], label='OPR  (tubing + WHP)')

    if q_s is not None and pwf_s is not None:
        ax.scatter([q_s], [pwf_s], s=120, zorder=5, color='red',
                   label=f'Operating point\nq = {q_s:,.0f} STB/D\npwf = {pwf_s:,.0f} psi')
        ax.axvline(q_s, ls='--', lw=0.8, color='red', alpha=0.5)
        ax.axhline(pwf_s, ls='--', lw=0.8, color='red', alpha=0.5)

    ax.set_xlabel('Liquid rate,  q  (STB/D)', fontsize=11)
    ax.set_ylabel('Bottom-hole flowing pressure,  p_wf  (psia)', fontsize=11)
    ax.set_title(f'Nodal Analysis – {well_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Wellbore pressure profile (TVD-based)
# ---------------------------------------------------------------------------

def plot_pressure_profile(profile: List[dict],
                          well_name: str = "Well",
                          show_md: bool = False,
                          figsize=(6, 7)) -> plt.Figure:
    """
    Plot pressure vs TVD (or MD) for a wellbore pressure traverse.

    profile : list of dicts with keys tvd_ft, md_ft, pressure_psi
    """
    tvd  = [pt['tvd_ft']       for pt in profile]
    md   = [pt['md_ft']        for pt in profile]
    p    = [pt['pressure_psi'] for pt in profile]

    x_vals = md if show_md else tvd
    x_label = 'Measured Depth (ft)' if show_md else 'True Vertical Depth (ft)'

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(p, x_vals, lw=2, color=_COLORS[0], marker='o', ms=3)
    ax.invert_yaxis()
    ax.set_xlabel('Pressure (psia)', fontsize=11)
    ax.set_ylabel(x_label, fontsize=11)
    ax.set_title(f'Wellbore Pressure Profile – {well_name}', fontsize=12, fontweight='bold')
    ax.annotate(f"Surface: {p[0]:.0f} psia", xy=(p[0], x_vals[0]),
                xytext=(p[0] + 50, x_vals[0] + 50), fontsize=8, color='gray')
    ax.annotate(f"BHP: {p[-1]:.0f} psia", xy=(p[-1], x_vals[-1]),
                xytext=(p[-1] + 50, x_vals[-1] - 100), fontsize=8, color='gray')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Network node pressures
# ---------------------------------------------------------------------------

def plot_network_pressures(node_pressures: Dict[str, float],
                           node_rates:     Dict[str, float] = None,
                           figsize=(10, 4)) -> plt.Figure:
    """Bar chart of pressures at all network nodes."""
    names  = list(node_pressures.keys())
    pressures = [node_pressures[n] for n in names]
    rates     = [node_rates.get(n, 0) if node_rates else 0 for n in names]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Pressure bars
    ax = axes[0]
    bars = ax.barh(names, pressures, color=_COLORS[0], alpha=0.8, edgecolor='white')
    for bar, p in zip(bars, pressures):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f'{p:.0f}', va='center', fontsize=8)
    ax.set_xlabel('Pressure (psia)', fontsize=10)
    ax.set_title('Node Pressures', fontsize=11, fontweight='bold')

    # Rate bars
    ax2 = axes[1]
    bars2 = ax2.barh(names, rates, color=_COLORS[1], alpha=0.8, edgecolor='white')
    for bar, r in zip(bars2, rates):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 f'{r:.0f}', va='center', fontsize=8)
    ax2.set_xlabel('Liquid Rate (STB/D)', fontsize=10)
    ax2.set_title('Cumulative Flow Rates', fontsize=11, fontweight='bold')

    fig.suptitle('Network Solution Summary', fontsize=12, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Flowline pressure traverse
# ---------------------------------------------------------------------------

def plot_flowline_profile(profiles: Dict[str, List[dict]],
                          figsize=(9, 4)) -> plt.Figure:
    """Plot pressure along each flowline segment."""
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, prof) in enumerate(profiles.items()):
        L  = [pt['L_cum_ft']    for pt in prof]
        p  = [pt['pressure_psi'] for pt in prof]
        ax.plot(L, p, lw=2, color=_COLORS[i % len(_COLORS)], label=name, marker='o', ms=3)

    ax.set_xlabel('Cumulative pipe length (ft)', fontsize=11)
    ax.set_ylabel('Pressure (psia)', fontsize=11)
    ax.set_title('Flowline Pressure Traverse', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Pipeline elevation profile (schematic)
# ---------------------------------------------------------------------------

def plot_pipeline_elevation(profile_xy: List[tuple],
                             pipe_name: str = "Flowline",
                             figsize=(9, 3)) -> plt.Figure:
    """
    Schematic elevation view of the pipeline.
    profile_xy : list of (x_ft, z_ft) pairs
    """
    x = [pt[0] for pt in profile_xy]
    z = [pt[1] for pt in profile_xy]

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(x, min(z) - 20, z, alpha=0.15, color=_COLORS[0])
    ax.plot(x, z, lw=2, color=_COLORS[0])
    ax.set_xlabel('Horizontal distance (ft)', fontsize=10)
    ax.set_ylabel('Elevation (ft)', fontsize=10)
    ax.set_title(f'Pipeline Elevation Profile – {pipe_name}', fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Comprehensive single-well dashboard
# ---------------------------------------------------------------------------

def plot_well_dashboard(well_name: str,
                        curves:  dict,
                        profile: List[dict],
                        figsize=(13, 5)) -> plt.Figure:
    """
    Two-panel figure: nodal analysis (left) + wellbore pressure profile (right).
    """
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    # --- Left: nodal analysis
    ax1 = fig.add_subplot(gs[0])
    q     = curves['q_vals']
    ipr   = curves['pwf_ipr']
    opr   = curves['pwf_opr']
    q_s   = curves.get('q_star')
    pwf_s = curves.get('pwf_star')

    ax1.plot(q, ipr, lw=2, color=_COLORS[0], label='IPR')
    ax1.plot(q, opr, lw=2, color=_COLORS[1], label='OPR (tubing)')
    if q_s:
        ax1.scatter([q_s], [pwf_s], s=120, color='red', zorder=5,
                    label=f'q* = {q_s:,.0f} STB/D\npwf* = {pwf_s:,.0f} psi')
        ax1.axvline(q_s,   ls='--', lw=0.8, color='red', alpha=0.5)
        ax1.axhline(pwf_s, ls='--', lw=0.8, color='red', alpha=0.5)
    ax1.set_xlabel('q  (STB/D)')
    ax1.set_ylabel('p_wf  (psia)')
    ax1.set_title('Nodal Analysis')
    ax1.legend(fontsize=8)
    ax1.set_xlim(left=0); ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    # --- Right: pressure profile
    ax2 = fig.add_subplot(gs[1])
    tvd = [pt['tvd_ft']       for pt in profile]
    p   = [pt['pressure_psi'] for pt in profile]
    ax2.plot(p, tvd, lw=2, color=_COLORS[2], marker='o', ms=3)
    ax2.invert_yaxis()
    ax2.set_xlabel('Pressure (psia)')
    ax2.set_ylabel('TVD (ft)')
    ax2.set_title('Wellbore Pressure Profile')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Well Performance Dashboard – {well_name}',
                 fontsize=13, fontweight='bold')
    return fig

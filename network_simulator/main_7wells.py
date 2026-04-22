"""
main_7wells.py
==============
7-well gathering network simulation.

Topology
--------
  Well_1 → WH_1 ─ FL_1 ─┐
  Well_2 → WH_2 ─ FL_2 ─┤─ M1 ─ FL_M1_M2 ─┐
  Well_3 → WH_3 ─ FL_3 ─┘                   │
                                              ├─ M2 (FixedRateSink, 45,000 STB/D)
  Well_4 → WH_4 ─ FL_4 ─────────────────────┤
                                              │
  Well_5 → WH_5 ─ FL_5 ─┐                   │
  Well_6 → WH_6 ─ FL_6 ─┤─ M3 ─ FL_M3_M2 ──┘
  Well_7 → WH_7 ─ FL_7 ─┘

Usage
-----
    python main_7wells.py

Survey CSV files required in same directory:
    w1_survey.csv … w7_survey.csv
    (any of the three formats: MD+TVD, MD+INC, or MD+TVD+INC)
"""

import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fluid_correlations  import BlackOilFluid
from network_components  import (
    SurveyPoint, Well, FlowlinePipe,
    Manifold, FixedRateSink
)
from network_solver      import ProductionNetwork, nodal_analysis_curves
from survey_reader       import read_survey

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728',
          '#9467bd','#8c564b','#e377c2']

# ============================================================
# 1.  Fluid
# ============================================================
fluid = BlackOilFluid(
    api          = 45.0,
    mu_o_res_cp  = 0.902,
    sg_gas       = 0.6636,
    gor_scf_stb  = 800.0,
    pb_psia      = 3800.0,
    bo_pb        = 1.2,
    wc           = 0.05,
    T_F          = 150.0,
    p_res_psia   = 3800.0,
    p_sep_psia   = 14.54,
)

print("=" * 62)
print("BLACK-OIL FLUID PROPERTIES  (T = 150 °F)")
print("=" * 62)
print(f"  Bubble point = {fluid.bubble_point_psia():.0f} psia")
print(f"  {'p (psia)':>10}  {'Bo (bbl/STB)':>13}  {'μo (cp)':>9}  {'ρmix (lbm/ft³)':>15}")
print(f"  {'─'*52}")
for p_t in [500, 1000, 2000, 3000, 3800, 4000]:
    print(f"  {p_t:>10}  {fluid.oil_fvf(p_t):>13.4f}"
          f"  {fluid.oil_viscosity_cp(p_t):>9.3f}"
          f"  {fluid.mixture_density_lbmft3(p_t):>15.2f}")

# ============================================================
# 2.  Wells
# ============================================================
WELL_PARAMS = dict(
    ID_in      = 2.922, #2.922,     # 3½″ tubing
    eps_in     = 0.001,
    J_stbd_psi = 5.9,
    Pr_psia    = 3800,
    # Pwh_psia   = 1200,
)

SURVEY_DIR = os.path.dirname(os.path.abspath(__file__))

wells = []
for i in range(1, 8):
    csv_path = os.path.join(SURVEY_DIR, f'w{1}_test.csv')
    survey   = read_survey(csv_path, verbose=True)
    
    w = Well(name=f"Well_{i}", fluid=fluid, survey=survey, **WELL_PARAMS)
    wells.append(w)

well_1, well_2, well_3, well_4, well_5, well_6, well_7 = wells

print(f"\n{'─'*62}")
print("WELL GEOMETRY (min-curvature)")
print(f"{'─'*62}")
for w in wells:
    segs = [f"{s.inc_start:.0f}°→{s.inc_end:.0f}°" for s in w.segments]
    print(f"  {w.name}: MD={w.MD_ft:.0f}ft  TVD={w.TVD_ft:.0f}ft  "
          f"AOF={w.q_from_aof():.0f} STB/D  segs={segs}")

# ============================================================
# 3.  Flowlines
# ============================================================
# Wells 1-3 → M1
fl_1 = FlowlinePipe("FL_1", "WH_1", "M1", fluid,
                    [(0, 0),  (700, 5)],           ID_in=4.5, eps_in=0.001)
fl_2 = FlowlinePipe("FL_2", "WH_2", "M1", fluid,
                    [(0, 4),  (660, 5)],            ID_in=4.5, eps_in=0.001)
fl_3 = FlowlinePipe("FL_3", "WH_3", "M1", fluid,
                    [(0, -30),(600, 5)],            ID_in=4.5, eps_in=0.001)

# Well 4 → M2 (direct)
fl_4 = FlowlinePipe("FL_4", "WH_4", "M2", fluid,
                    [(0, 15), (600, 15)],           ID_in=4.5, eps_in=0.001)

# Wells 5-7 → M3
fl_5 = FlowlinePipe("FL_5", "WH_5", "M3", fluid,
                    [(0, 17), (420, 27)],           ID_in=4.5, eps_in=0.001)
fl_6 = FlowlinePipe("FL_6", "WH_6", "M3", fluid,
                    [(0, 5),  (290, 27)],           ID_in=4.5, eps_in=0.001)
fl_7 = FlowlinePipe("FL_7", "WH_7", "M3", fluid,
                    [(0, 5),  (177, 27)],           ID_in=4.5, eps_in=0.001)

# Trunk lines → M2
fl_m1_m2 = FlowlinePipe("FL_M1_M2", "M1", "M2", fluid,
                         [(0,5),(600,7),(890,17),(1322,17),(2824,15)],
                         ID_in=8, eps_in=0.001)
fl_m3_m2 = FlowlinePipe("FL_M3_M2", "M3", "M2", fluid,
                         [(0,27),(1640,33),(3885,24),(5312,12),(8057,15)],
                         ID_in=8, eps_in=0.001)

# ============================================================
# 4.  Build & solve network
# ============================================================
print(f"\n{'═'*62}")
print("NETWORK SOLVE  –  Fixed-rate sink M2 = 45,000 STB/D")
print(f"{'═'*62}")

net = ProductionNetwork("7-Well Gathering System to M2")

for w, wh in zip(wells, ["WH_1","WH_2","WH_3","WH_4","WH_5","WH_6","WH_7"]):
    net.add_well(w, wh)

for fl in [fl_1, fl_2, fl_3, fl_4, fl_5, fl_6, fl_7, fl_m1_m2, fl_m3_m2]:
    net.add_flowline(fl)

net.add_manifold(Manifold("M1"))
net.add_manifold(Manifold("M3"))
net.set_fixed_rate_sink(FixedRateSink("M2", q_target_stbd=45000,
                                       p_outlet_psia=1202.54))

results = net.solve(verbose=True)

# ============================================================
# 5.  Nodal analysis – all 7 wells
# ============================================================
print(f"\n{'─'*62}")
print("NODAL ANALYSIS")
print(f"{'─'*62}")
curves_all = {}
for w in wells:
    c = nodal_analysis_curves(w, q_min=0,
                               q_max=w.q_from_aof() * 0.9, npts=80)
    curves_all[w.name] = c
    qs = c.get('q_star'); ps = c.get('pwf_star')
    if qs:
        print(f"  {w.name}: q* = {qs:.0f} STB/D   pwf* = {ps:.0f} psia")

# ============================================================
# 6.  Figures
# ============================================================
print(f"\n{'─'*62}")
print("GENERATING FIGURES")
print(f"{'─'*62}")


# ── Fig 1: All-wells nodal analysis ─────────────────────────────────────────
fig1, axes = plt.subplots(2, 4, figsize=(18, 8))
fig1.suptitle("Nodal Analysis – 7 Wells  (WHP = 1200 psia, T = 150 °F, Pr = 3800 psia)",
              fontsize=12, fontweight='bold')
axes_list = list(axes.flat)
for ax, (wname, c), col in zip(axes_list, curves_all.items(), COLORS):
    q = c['q_vals']; ipr = c['pwf_ipr']; opr = c['pwf_opr']
    ax.plot(q, ipr, lw=2, color=col,         label='IPR')
    ax.plot(q, opr, lw=2, color=col, ls='--', label='OPR')
    qs = c.get('q_star'); ps = c.get('pwf_star')
    if qs:
        ax.scatter([qs], [ps], s=80, color='red', zorder=5,
                   label=f'q*={qs:.0f}\npwf*={ps:.0f}')
        ax.axvline(qs, ls=':', lw=0.8, color='red', alpha=0.5)
        ax.axhline(ps, ls=':', lw=0.8, color='red', alpha=0.5)
    ax.set_title(wname, fontsize=10, fontweight='bold')
    ax.set_xlabel('q (STB/D)', fontsize=8)
    ax.set_ylabel('pwf (psia)', fontsize=8)
    
    ax.legend(fontsize=6); ax.set_xlim(left=0, right=8000); ax.set_ylim(bottom=0)

    ax.grid(True, alpha=0.3)
# Hide unused 8th panel
axes_list[-1].set_visible(False)
fig1.tight_layout()
fig1.savefig(f"{OUT}/fig1_nodal_all_wells.png", dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  ✓ fig1_nodal_all_wells.png")


# ── Fig 2: Wellbore pressure profiles ────────────────────────────────────────
fig2, axes = plt.subplots(2, 4, figsize=(18, 9))
fig2.suptitle("Wellbore Pressure Profiles – 7 Wells  (constrained rates)",
              fontsize=12, fontweight='bold')
for ax, wr, col in zip(list(axes.flat), results['well_results'], COLORS):
    prof = wr['profile']
    if not prof:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        continue
    tvd = [pt['tvd_ft']       for pt in prof]
    p   = [pt['pressure_psi'] for pt in prof]
    ax.plot(p, tvd, lw=2, color=col, marker='o', ms=3)
    ax.invert_yaxis()
    ax.set_xlabel('Pressure (psia)', fontsize=8)
    ax.set_ylabel('TVD (ft)', fontsize=8)
    ax.set_title(f"{wr['well_name']}\nq={wr['q_stbd']:.0f} STB/D  "
                 f"pwf={wr['pwf_psia']:.0f} psi", fontsize=9, fontweight='bold')
    ax.annotate(f"WHP={p[0]:.0f}", xy=(p[0], tvd[0]),
                xytext=(p[0]+20, tvd[0]+80), fontsize=6, color='gray')
    ax.annotate(f"BHP={p[-1]:.0f}", xy=(p[-1], tvd[-1]),
                xytext=(p[-1]+20, tvd[-1]-80), fontsize=6, color='gray')
    ax.grid(True, alpha=0.3)
axes[1,3].set_visible(False)
fig2.tight_layout()
fig2.savefig(f"{OUT}/fig2_wellbore_profiles.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  ✓ fig2_wellbore_profiles.png")


# ── Fig 3: Network node pressures ────────────────────────────────────────────
np_dict = results['node_pressures']
nr_dict = results['node_rates']
node_order = ["WH_1","WH_2","WH_3","WH_4","WH_5","WH_6","WH_7",
              "M1","M3","M2"]
labels  = [n for n in node_order if n in np_dict]
pvals   = [np_dict[n]          for n in labels]
qvals   = [nr_dict.get(n, 0.0) for n in labels]
bar_col = (['#4e9af1']*7 + ['#f0a500','#f0a500','#e74c3c'])[:len(labels)]

fig3, axes = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle("Network Solution – Node Pressures & Rates", fontsize=12, fontweight='bold')

ax = axes[0]
bars = ax.barh(labels, pvals, color=bar_col, edgecolor='white', alpha=0.85)
for bar, v in zip(bars, pvals):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}', va='center', fontsize=8)
ax.set_xlabel('Pressure (psia)'); ax.set_title('Node Pressures')
ax.axvline(fluid.pb_psia, ls='--', color='red', alpha=0.5, label=f'Pb={fluid.pb_psia:.0f} psia')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='x')

ax2 = axes[1]
bars2 = ax2.barh(labels, qvals, color=bar_col, edgecolor='white', alpha=0.85)
for bar, v in zip(bars2, qvals):
    ax2.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
             f'{v:.0f}', va='center', fontsize=8)
ax2.set_xlabel('Rate (STB/D)'); ax2.set_title('Cumulative Flow Rates')
ax2.grid(True, alpha=0.2, axis='x')

fig3.tight_layout()
fig3.savefig(f"{OUT}/fig3_node_summary.png", dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  ✓ fig3_node_summary.png")


# ── Fig 4: Flowline pressure traverses ───────────────────────────────────────
fp = results['flowline_profiles']
fig4, axes = plt.subplots(3, 3, figsize=(14, 10))
fig4.suptitle("Flowline Pressure Traverses", fontsize=12, fontweight='bold')
fl_names = ["FL_1","FL_2","FL_3","FL_4","FL_5","FL_6","FL_7",
            "FL_M1_M2","FL_M3_M2"]
for ax, name, col in zip(axes.flat, fl_names, COLORS + ['#555','#888']):
    if name not in fp:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(name, fontsize=9)
        continue
    prof = fp[name]
    L = [pt['L_cum_ft']    for pt in prof]
    p = [pt['pressure_psi'] for pt in prof]
    ax.plot(L, p, lw=2, color=col, marker='o', ms=4)
    ax.set_title(name, fontsize=9, fontweight='bold')
    ax.set_xlabel('Dist. (ft)', fontsize=7)
    ax.set_ylabel('Pressure (psia)', fontsize=7)
    ax.grid(True, alpha=0.3)
fig4.tight_layout()
fig4.savefig(f"{OUT}/fig4_flowline_traverses.png", dpi=150, bbox_inches='tight')
plt.close(fig4)
print("  ✓ fig4_flowline_traverses.png")


# ── Fig 5: Well trajectory map ────────────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(9, 8))
for w, col in zip(wells, COLORS):
    x_list, z_list = [0.0], [0.0]
    x_cum = tvd_cum = 0.0
    for seg in w.segments:
        i1 = math.radians(seg.inc_start)
        i2 = math.radians(seg.inc_end)
        beta = math.acos(max(-1.0, min(1.0, math.cos(i2-i1))))
        rf   = 2/beta*math.tan(beta/2) if abs(beta) > 1e-9 else 1.0
        x_cum  += seg.dmd/2*(math.sin(i1)+math.sin(i2))*rf
        tvd_cum += seg.dmd/2*(math.cos(i1)+math.cos(i2))*rf
        x_list.append(x_cum)
        z_list.append(tvd_cum)
    ax5.plot(x_list, z_list, lw=2, color=col,
             label=f"{w.name}  TVD={w.TVD_ft:.0f}ft", marker='o', ms=4)
ax5.invert_yaxis()
ax5.set_xlabel("Horizontal Displacement (ft)", fontsize=11)
ax5.set_ylabel("TVD (ft)", fontsize=11)
ax5.set_title("Well Survey Trajectories – 7 Wells", fontsize=12, fontweight='bold')
ax5.legend(fontsize=8); ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(f"{OUT}/fig5_well_trajectories.png", dpi=150, bbox_inches='tight')
plt.close(fig5)
print("  ✓ fig5_well_trajectories.png")


# ── Fig 6: Network schematic ──────────────────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(14, 6))
ax6.set_xlim(-0.5, 12); ax6.set_ylim(-0.5, 4); ax6.axis('off')
ax6.set_title("7-Well Gathering Network – M2 Fixed-Rate Sink (45,000 STB/D)",
              fontsize=12, fontweight='bold')

pos = {
    "WH_1": (0, 3.5), "WH_2": (0, 2.8), "WH_3": (0, 2.1),
    "M1":   (2.8, 2.8),
    "WH_4": (0, 1.4),
    "M2":   (7.5, 1.9),
    "WH_5": (0, 0.7), "WH_6": (0, 0.0), "WH_7": (0, -0.4),
    "M3":   (2.8, 0.3),
}
node_color = {
    **{f"WH_{i}": "#a9dfbf" for i in range(1,8)},
    "M1": "#f9e79f", "M3": "#f9e79f",
    "M2": "#ec7063",
}
for n, (x, y) in pos.items():
    c = plt.Circle((x, y), 0.28, color=node_color.get(n,'#ccc'),
                   ec='#444', lw=1.2, zorder=3)
    ax6.add_patch(c)
    p_val = np_dict.get(n, 0)
    q_val = nr_dict.get(n, 0)
    ax6.text(x, y+0.05, n, ha='center', va='center',
             fontsize=6.5, fontweight='bold', zorder=4)
    ax6.text(x, y-0.12, f"{p_val:.0f} psia", ha='center',
             fontsize=5.5, color='#333', zorder=4)

edges_draw = [
    ("WH_1","M1"),("WH_2","M1"),("WH_3","M1"),
    ("WH_4","M2"),
    ("WH_5","M3"),("WH_6","M3"),("WH_7","M3"),
    ("M1","M2"),("M3","M2"),
]
for u, v in edges_draw:
    xu, yu = pos[u]; xv, yv = pos[v]
    ax6.annotate("", xy=(xv, yv), xytext=(xu, yu),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='#444'))
    mx, my = (xu+xv)/2, (yu+yv)/2
    q_edge = nr_dict.get(u, 0)
    ax6.text(mx, my+0.12, f"{q_edge:.0f} STB/D",
             ha='center', fontsize=5.5, color='#1a5276')

fig6.tight_layout()
fig6.savefig(f"{OUT}/fig6_network_schematic.png", dpi=150, bbox_inches='tight')
plt.close(fig6)
print("  ✓ fig6_network_schematic.png")


# ── Fig 7: Fluid PVT ─────────────────────────────────────────────────────────
p_range = list(range(100, 4200, 100))
fig7, axes = plt.subplots(2, 2, figsize=(12, 8))
fig7.suptitle(f"Black-Oil PVT  (API={fluid.api}, GOR={fluid.gor_scf_stb} scf/STB,"
              f" Pb={fluid.pb_psia:.0f} psia, T=150°F)",
              fontsize=11, fontweight='bold')

axes[0,0].plot(p_range, [fluid.oil_fvf(p) for p in p_range], color=COLORS[0], lw=2)
axes[0,0].axvline(fluid.pb_psia, ls='--', color='gray', alpha=0.6, label='Pb')
axes[0,0].set_title('Bo (bbl/STB)'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
axes[0,0].set_xlabel('p (psia)')

axes[0,1].plot(p_range, [fluid.oil_viscosity_cp(p) for p in p_range], color=COLORS[1], lw=2)
axes[0,1].axvline(fluid.pb_psia, ls='--', color='gray', alpha=0.6)
axes[0,1].scatter([fluid.p_res_psia], [fluid.mu_o_res_cp], s=80, color='red',
                   zorder=5, label=f'Measured {fluid.mu_o_res_cp} cp')
axes[0,1].set_title('μo (cp)'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
axes[0,1].set_xlabel('p (psia)')

axes[1,0].plot(p_range, [fluid.solution_gor(p) for p in p_range], color=COLORS[2], lw=2)
axes[1,0].axvline(fluid.pb_psia, ls='--', color='gray', alpha=0.6)
axes[1,0].set_title('Rs (scf/STB)'); axes[1,0].grid(True, alpha=0.3)
axes[1,0].set_xlabel('p (psia)')

axes[1,1].plot(p_range, [fluid.mixture_density_lbmft3(p) for p in p_range],
               color=COLORS[3], lw=2, label=f'Mix (wc={fluid.wc:.0%})')
axes[1,1].plot(p_range, [fluid.oil_density_lbmft3(p) for p in p_range],
               color=COLORS[0], lw=1.5, ls='--', label='Oil')
axes[1,1].set_title('Density (lbm/ft³)'); axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3); axes[1,1].set_xlabel('p (psia)')

fig7.tight_layout()
fig7.savefig(f"{OUT}/fig7_fluid_pvt.png", dpi=150, bbox_inches='tight')
plt.close(fig7)
print("  ✓ fig7_fluid_pvt.png")


# ============================================================
# 7.  Summary table
# ============================================================
print(f"\n{'═'*70}")
print("FINAL PRODUCTION SUMMARY")
print(f"{'═'*70}")
print(f"\n  {'Well':<10} {'q (STB/D)':>10} {'q_oil':>10} {'q_water':>9} "
      f"{'pwf (psia)':>11} {'pwh (psia)':>11} {'TVD (ft)':>9}")
print(f"  {'─'*72}")
total_q = total_oil = total_water = 0.0
for wr in results['well_results']:
    q_oil   = wr['q_stbd'] * (1 - fluid.wc)
    q_water = wr['q_stbd'] * fluid.wc
    total_q     += wr['q_stbd']
    total_oil   += q_oil
    total_water += q_water
    print(f"  {wr['well_name']:<10} {wr['q_stbd']:>10.1f} {q_oil:>10.1f} "
          f"{q_water:>9.1f} {wr['pwf_psia']:>11.1f} "
          f"{wr['pwh_psia']:>11.1f} {wr['tvd_ft']:>9.0f}")
print(f"  {'─'*72}")
print(f"  {'TOTAL':<10} {total_q:>10.1f} {total_oil:>10.1f} {total_water:>9.1f}")
print(f"\n  Water cut        : {fluid.wc*100:.0f}%")
print(f"  M1 pressure      : {np_dict.get('M1', 0):>8.1f} psia")
print(f"  M3 pressure      : {np_dict.get('M3', 0):>8.1f} psia")
print(f"  M2 inlet pressure: {np_dict.get('M2', 0):>8.1f} psia")
print(f"  M2 target rate   : 45,000.0 STB/D")
print(f"  M2 delivered     : {nr_dict.get('M2', 0):>8.1f} STB/D")
print(f"\n✓ All figures saved to {OUT}/")

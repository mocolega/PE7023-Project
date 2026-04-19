"""
network_solver.py
=================
Network solver — generalised tree topology.

Key insight: WHP(q) is NON-MONOTONE for multiphase wells.
At very low rates: heavy liquid column → low WHP.
At moderate rates: gas lightens column → WHP rises to a peak.
At high rates: friction dominates → WHP falls back to atmospheric.

Only the FALLING (right) branch q ∈ [q_peak, q_max] is physically
stable. The solver bisects exclusively on this branch.

Boundary conditions
-------------------
SeparatorNode  : fixed p_sep  → rates determined by IPR + tubing balance
FixedRateSink  : fixed q_total → find common WHP* on stable branch
                 such that Σ q_i(WHP*) = q_target
"""

from __future__ import annotations
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from network_components import (
    Node, ReservoirNode, SeparatorNode, ManifoldNode,
    FixedRateSink, Well, FlowlinePipe, Choke, Manifold
)


class ProductionNetwork:

    def __init__(self, name: str = "Network"):
        self.name        = name
        self.nodes:      Dict[str, Node]        = {}
        self.wells:      List[Well]             = []
        self.flowlines:  List[FlowlinePipe]     = []
        self.chokes:     List[Choke]            = []
        self.manifolds:  List[Manifold]         = []
        self._sink_node: Optional[Node]         = None

        self.node_pressure:     Dict[str, float] = {}
        self.node_rate:         Dict[str, float] = {}
        self.well_results:      List[dict]        = []
        self.flowline_profiles: Dict[str, list]   = {}

    # ── Registration ─────────────────────────────────────────────────────────

    def _ensure_node(self, name, p_fixed=None, cls=Node):
        if name not in self.nodes:
            self.nodes[name] = cls(name, p_fixed)
        elif p_fixed is not None:
            self.nodes[name].pressure_psi = p_fixed
        return self.nodes[name]

    def set_separator(self, sep: SeparatorNode):
        self._sink_node = sep
        self.nodes[sep.name] = sep

    def set_fixed_rate_sink(self, sink: FixedRateSink):
        self._sink_node = sink
        self.nodes[sink.name] = sink

    def add_well(self, well: Well, wellhead_node: str):
        self.wells.append(well)
        well.wellhead_node = wellhead_node
        self._ensure_node(wellhead_node)

    def add_flowline(self, fl: FlowlinePipe):
        self.flowlines.append(fl)
        self._ensure_node(fl.u)
        self._ensure_node(fl.v)

    def add_choke(self, chk: Choke):
        self.chokes.append(chk)
        self._ensure_node(chk.u)
        self._ensure_node(chk.v)

    def add_manifold(self, mfld: Manifold):
        self.manifolds.append(mfld)
        self.nodes[mfld.name] = mfld.node

    # ── Graph ────────────────────────────────────────────────────────────────

    def _build_graph(self):
        downstream, upstream, edge_map = {}, defaultdict(list), {}
        for fl in self.flowlines:
            downstream[fl.u] = fl.v
            upstream[fl.v].append(fl.u)
            edge_map[(fl.u, fl.v)] = fl
        for chk in self.chokes:
            downstream[chk.u] = chk.v
            upstream[chk.v].append(chk.u)
            edge_map[(chk.u, chk.v)] = chk
        return downstream, upstream, edge_map

    def _topological_order(self, upstream, sink_name):
        order, visited = [], set([sink_name])
        queue = deque([sink_name])
        while queue:
            n = queue.popleft()
            order.append(n)
            for u in upstream.get(n, []):
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
        return order  # sink-first; reverse for source-first

    # ── Well WHP helpers ──────────────────────────────────────────────────────

    def _whp_at_q(self, well: Well, q: float) -> Tuple[float, list]:
        """WHP(q) from tubing traverse. Returns (whp_psia, profile)."""
        q   = max(q, 0.1)
        pwf = well.pwf_from_q(q)
        try:
            dp, prof = well.tubing_dp_total_psi(q, pwf)
            return pwf - dp, prof
        except Exception:
            return pwf, []

    def _find_stable_branch(self, well: Well,
                            n_scan: int = 40) -> Tuple[float, float, float]:
        """
        Scan WHP(q) to find the stable (falling) branch.

        Returns
        -------
        (q_peak, whp_peak, q_max)
        q_peak  : rate at maximum WHP (top of hump)
        whp_peak: maximum WHP value
        q_max   : rate where WHP → atmospheric (14.7 psia)
        """
        aof   = well.q_from_aof()
        q_max = aof  # start assuming q_max = AOF

        # Find q_max: highest q where WHP > 14.7 psia
        q_lo, q_hi = 1.0, aof
        whp_lo, _ = self._whp_at_q(well, q_lo)
        whp_hi, _ = self._whp_at_q(well, q_hi)

        if whp_lo <= 14.7:
            # Well can barely flow even at minimum rate
            return 1.0, whp_lo, 1.0

        if whp_hi > 14.7:
            q_max = q_hi  # WHP still positive at AOF
        else:
            # Bisect to find q_max (where WHP → 14.7)
            for _ in range(60):
                m = 0.5 * (q_lo + q_hi)
                whp_m, _ = self._whp_at_q(well, m)
                if whp_m > 14.7:
                    q_lo = m
                else:
                    q_hi = m
                if q_hi - q_lo < 1.0:
                    break
            q_max = q_lo

        # Scan [1, q_max] to find q_peak (maximum WHP)
        best_q, best_whp = 1.0, -1e9
        step = q_max / n_scan
        for i in range(n_scan + 1):
            q = max(1.0, step * i)
            if q > q_max:
                break
            whp, _ = self._whp_at_q(well, q)
            if whp > best_whp:
                best_whp = whp
                best_q   = q

        # Refine q_peak with golden section in [best_q-step, best_q+step]
        lo = max(1.0,   best_q - step)
        hi = min(q_max, best_q + step)
        gr = (math.sqrt(5) - 1) / 2
        for _ in range(40):
            q1 = hi - gr * (hi - lo)
            q2 = lo + gr * (hi - lo)
            w1, _ = self._whp_at_q(well, q1)
            w2, _ = self._whp_at_q(well, q2)
            if w1 < w2:
                lo = q1
            else:
                hi = q2
            if hi - lo < 0.5:
                break
        q_peak   = 0.5 * (lo + hi)
        whp_peak, _ = self._whp_at_q(well, q_peak)

        return q_peak, whp_peak, q_max

    def _q_at_whp_stable(self, well: Well,
                          p_wh: float,
                          q_peak: float,
                          whp_peak: float,
                          q_max: float) -> float:
        """
        Find q on the STABLE branch [q_peak, q_max] such that WHP(q) = p_wh.
        If p_wh > whp_peak: well cannot flow at that back-pressure → q = 0.
        If p_wh ≤ 14.7: well flows at q_max.
        """
        if p_wh > whp_peak:
            return 0.0
        if p_wh <= 14.7:
            return q_max

        # Bisect on [q_peak, q_max] — WHP is monotone decreasing here
        lo, hi = q_peak, q_max
        for _ in range(60):
            m = 0.5 * (lo + hi)
            whp_m, _ = self._whp_at_q(well, m)
            if whp_m > p_wh:
                lo = m   # need higher q to lower WHP
            else:
                hi = m
            if hi - lo < 0.5:
                break
        return 0.5 * (lo + hi)

    # ── Rate accumulation & pressure march ───────────────────────────────────

    def _accumulate_rates(self, q_at_wh, src_first, downstream, sink_name):
        node_rate = dict(q_at_wh)
        for node in src_first:
            if node == sink_name:
                continue
            ds = downstream.get(node)
            if ds is not None:
                node_rate[ds] = node_rate.get(ds, 0.0) + node_rate.get(node, 0.0)
        return node_rate

    def _march_pressures(self, node_rate, src_first, downstream,
                         edge_map, sink_name, verbose):
        p = dict(self.node_pressure)

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Flowline pressure traverse")
            print(f"{'─'*60}")

        for node in src_first:
            if node == sink_name:
                continue
            ds   = downstream.get(node)
            pipe = edge_map.get((node, ds)) if ds else None
            if pipe is None:
                continue
            p_in = p.get(node)
            q_fl = node_rate.get(node, 0.0)
            if p_in is None or q_fl <= 0:
                continue

            if isinstance(pipe, FlowlinePipe):
                dp, prof = pipe.dp_psi(q_fl, p_in)
                self.flowline_profiles[pipe.name] = prof
            else:
                dp = pipe.dp_psi(q_fl, p_in)

            p_out = p_in - dp
            p[ds]  = min(p.get(ds, p_out), p_out)

            if verbose:
                print(f"  {pipe.name:14s}  {node:8s}→{ds:8s}"
                      f"  p_in={p_in:7.1f}  dp={dp:6.1f}"
                      f"  p_out={p_out:7.1f} psia"
                      f"  q={q_fl:7.0f} STB/D")
        return p

    # ── Fixed-pressure sink ───────────────────────────────────────────────────

    def _solve_fixed_pressure(self, verbose):
        sep_name = self._sink_node.name
        p_sep    = self._sink_node.pressure_psi

        downstream, upstream, edge_map = self._build_graph()
        src_first = list(reversed(self._topological_order(upstream, sep_name)))

        self.node_pressure[sep_name] = p_sep

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  STEP 1 – Well operating points  (fixed Pr, fixed WHP)")
            print(f"{'─'*60}")

        q_at_wh = {}
        for well in self.wells:
            whn = well.wellhead_node
            try:
                q_w, pwf_w = well.solve_operating_point()
                _, tp       = well.tubing_dp_total_psi(q_w, pwf_w)
                pwh_w       = tp[0]['pressure_psi']
            except Exception as e:
                if verbose:
                    print(f"  [WARN] {well.name}: {e}")
                q_w, pwf_w, pwh_w, tp = 0.0, well.Pr, well.Pwh or 0.0, []

            self.node_pressure[whn] = pwh_w
            q_at_wh[whn] = q_at_wh.get(whn, 0.0) + q_w
            self.node_rate[whn] = q_at_wh[whn]
            self.well_results.append({
                'well_name': well.name, 'q_stbd': q_w,
                'pwf_psia': pwf_w, 'pwh_psia': pwh_w,
                'tvd_ft': well.TVD_ft, 'profile': tp,
            })
            if verbose:
                print(f"  {well.name:10s}  q={q_w:8.1f} STB/D"
                      f"  pwf={pwf_w:7.1f}  pwh={pwh_w:7.1f} psia")

        node_rate = self._accumulate_rates(q_at_wh, src_first, downstream, sep_name)
        self.node_rate.update(node_rate)
        p = self._march_pressures(node_rate, src_first, downstream,
                                  edge_map, sep_name, verbose)
        self.node_pressure.update(p)
        return self._package_results()

    # ── Fixed-rate sink ───────────────────────────────────────────────────────

    def _solve_fixed_rate(self, verbose):
        """
        Fixed Pr per well + fixed q_target at sink.

        Algorithm
        ---------
        1. For each well, find the stable branch: (q_peak, WHP_peak, q_max).
           WHP_peak is the MAXIMUM wellhead pressure this well can deliver.

        2. Bisect on a common back-pressure p_wh* ∈ [14.7, min(WHP_peak)]
           such that Σ q_i(p_wh*) = q_target, where each q_i is found on
           the STABLE (monotone-decreasing) branch of each well's WHP curve.

        3. Compute constrained rates, profiles, and march through flowlines.
        """
        sink_name = self._sink_node.name
        q_target  = self._sink_node.q_target_stbd

        downstream, upstream, edge_map = self._build_graph()
        src_first = list(reversed(self._topological_order(upstream, sink_name)))

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  FIXED-RATE SINK  '{sink_name}'")
            print(f"  Boundary conditions: Pr per well + q_target={q_target:,.1f} STB/D")
            print(f"{'─'*60}")

        # ── Step 1: stable branch per well ────────────────────────────────
        if verbose:
            print(f"\n  STEP 1 – Stable branch analysis per well")
            print(f"{'─'*60}")
            print(f"  {'Well':10s}  {'AOF':>8}  {'q_peak':>8}  {'WHP_peak':>10}  {'q_max':>8}")
            print(f"  {'─'*52}")

        stable: Dict[str, dict] = {}
        for well in self.wells:
            q_pk, whp_pk, q_mx = self._find_stable_branch(well)
            stable[well.name] = {
                'q_peak':   q_pk,
                'whp_peak': whp_pk,
                'q_max':    q_mx,
            }
            if verbose:
                print(f"  {well.name:10s}  {well.q_from_aof():>8.0f}"
                      f"  {q_pk:>8.0f}  {whp_pk:>10.1f}  {q_mx:>8.0f}")

        # Maximum total deliverable (all wells at q_max on stable branch)
        q_total_max = sum(s['q_max'] for s in stable.values())
        # Min of all WHP_peaks (common p_wh* cannot exceed the most constrained well)
        whp_peak_min = min(s['whp_peak'] for s in stable.values())

        if verbose:
            print(f"\n  Total q_max = {q_total_max:,.1f} STB/D")
            print(f"  Min WHP_peak = {whp_peak_min:.1f} psia")
            print(f"  Target       = {q_target:,.1f} STB/D")

        if q_total_max < q_target:
            if verbose:
                print(f"\n  [WARN] Target {q_target:,.1f} > deliverable {q_total_max:,.1f}.")
                print(f"  Running at maximum deliverable rate.")
            q_target = q_total_max * 0.999

        # ── Step 2: bisect on p_wh* ───────────────────────────────────────
        if verbose:
            print(f"\n  STEP 2 – Bisect for common stable-branch WHP*")
            print(f"{'─'*60}")

        def total_q_at_whp(p_wh: float) -> Tuple[float, Dict[str, float]]:
            rates, total = {}, 0.0
            for well in self.wells:
                s   = stable[well.name]
                q_w = self._q_at_whp_stable(
                    well, p_wh, s['q_peak'], s['whp_peak'], s['q_max'])
                rates[well.name] = q_w
                total += q_w
            return total, rates

        # Bracket: low p_wh → high total rate, high p_wh → low total rate
        p_lo, p_hi = 14.7, whp_peak_min
        tot_lo, _  = total_q_at_whp(p_lo)
        tot_hi, _  = total_q_at_whp(p_hi)

        if verbose:
            print(f"  p_wh={p_lo:.1f}: total={tot_lo:.1f} STB/D")
            print(f"  p_wh={p_hi:.1f}: total={tot_hi:.1f} STB/D")

        final_rates: Dict[str, float] = {}
        p_wh_star = p_lo

        if tot_hi >= q_target:
            # Target achievable even at the peak — use p_wh = whp_peak_min
            p_wh_star = p_hi
            _, final_rates = total_q_at_whp(p_wh_star)
        elif tot_lo <= q_target:
            # Target exceeds maximum — already handled above
            p_wh_star = p_lo
            _, final_rates = total_q_at_whp(p_wh_star)
        else:
            # Normal bisect: total is monotone decreasing in p_wh on stable branch
            for iteration in range(80):
                p_mid = 0.5 * (p_lo + p_hi)
                tot_mid, rates_mid = total_q_at_whp(p_mid)
                if abs(tot_mid - q_target) < 1.0:
                    p_wh_star   = p_mid
                    final_rates = rates_mid
                    break
                if tot_mid > q_target:
                    p_lo = p_mid  # too much flow → raise WHP
                else:
                    p_hi = p_mid  # too little flow → lower WHP
            else:
                p_wh_star   = 0.5 * (p_lo + p_hi)
                _, final_rates = total_q_at_whp(p_wh_star)

        if verbose:
            print(f"  Common WHP* = {p_wh_star:.2f} psia")

        # ── Step 3: well profiles at final rates ──────────────────────────
        if verbose:
            print(f"\n  STEP 3 – Well solutions")
            print(f"{'─'*60}")

        q_at_wh = {}
        for well in self.wells:
            whn  = well.wellhead_node
            q_w  = final_rates.get(well.name, 0.0)
            pwf_w = well.pwf_from_q(max(q_w, 0.1))

            try:
                _, tp  = well.tubing_dp_total_psi(max(q_w, 0.1), pwf_w)
                pwh_w  = tp[0]['pressure_psi']
            except Exception:
                tp, pwh_w = [], pwf_w

            self.node_pressure[whn] = pwh_w
            q_at_wh[whn]  = q_at_wh.get(whn, 0.0) + q_w
            self.node_rate[whn] = q_at_wh[whn]

            self.well_results.append({
                'well_name':  well.name,
                'q_stbd':     q_w,
                'pwf_psia':   pwf_w,
                'pwh_psia':   pwh_w,
                'tvd_ft':     well.TVD_ft,
                'profile':    tp,
                'p_wh_star':  p_wh_star,
            })
            if verbose:
                status = "" if q_w > 1 else "  [below stable threshold]"
                print(f"  {well.name:10s}  q={q_w:8.1f} STB/D"
                      f"  pwf={pwf_w:7.1f}  pwh={pwh_w:7.1f} psia{status}")

        # ── Step 4: accumulate + march ─────────────────────────────────────
        node_rate = self._accumulate_rates(q_at_wh, src_first,
                                           downstream, sink_name)
        self.node_rate.update(node_rate)
        p = self._march_pressures(node_rate, src_first, downstream,
                                  edge_map, sink_name, verbose)
        self.node_pressure.update(p)

        q_del  = node_rate.get(sink_name, sum(q_at_wh.values()))
        p_sink = self.node_pressure.get(sink_name, 0.0)
        self._sink_node.pressure_psi = p_sink

        if verbose:
            print(f"\n  ✓ Delivered = {q_del:,.1f} STB/D"
                  f"  (target = {q_target:,.1f} STB/D)")
            print(f"  ✓ Sink inlet pressure = {p_sink:.1f} psia")

        return self._package_results()

    # ── Package ───────────────────────────────────────────────────────────────

    def _package_results(self):
        for n, p in self.node_pressure.items():
            if n in self.nodes:
                self.nodes[n].pressure_psi = p

        print(f"\n{'═'*60}")
        print(f"  NETWORK SOLUTION")
        print(f"{'═'*60}")
        print(f"  {'Node':<20} {'Pressure (psia)':>16} {'Rate (STB/D)':>14}")
        print(f"  {'─'*52}")
        for n, p in sorted(self.node_pressure.items()):
            q = self.node_rate.get(n, 0.0)
            print(f"  {n:<20} {p:>16.1f} {q:>14.1f}")

        return {
            'well_results':       self.well_results,
            'node_pressures':     self.node_pressure,
            'node_rates':         self.node_rate,
            'flowline_profiles':  self.flowline_profiles,
        }

    # ── Entry point ───────────────────────────────────────────────────────────

    def solve(self, verbose: bool = True) -> dict:
        if self._sink_node is None:
            raise RuntimeError("No sink registered.")
        if isinstance(self._sink_node, FixedRateSink):
            return self._solve_fixed_rate(verbose)
        else:
            return self._solve_fixed_pressure(verbose)


# ── Nodal analysis ────────────────────────────────────────────────────────────

def nodal_analysis_curves(well, q_min=0.0, q_max=None, npts=80) -> dict:
    """Generate IPR and OPR (tubing performance) curves."""
    if q_max is None:
        q_max = well.q_from_aof() * 0.98

    net    = ProductionNetwork()
    q_vals = [q_min + (q_max - q_min) * i / (npts - 1) for i in range(npts)]
    p_wh   = well.Pwh or 0.0

    pwf_ipr, pwf_opr, whp_tubing = [], [], []

    for q in q_vals:
        pwf_ipr.append(well.pwf_from_q(max(q, 0.1)))
        # OPR: pwf needed to deliver q against fixed WHP
        try:
            dp, _ = well.tubing_dp_total_psi(max(q, 0.1),
                                              well.pwf_from_q(max(q, 0.1)))
            pwf_opr.append(dp + p_wh)
        except Exception:
            pwf_opr.append(well.pwf_from_q(max(q, 0.1)))

        # Tubing performance: actual WHP delivered (for nodal)
        whp, _ = net._whp_at_q(well, max(q, 0.1))
        whp_tubing.append(whp)

    # Operating point where IPR WHP = tubing WHP
    q_star = pwf_star = None
    try:
        # Find where tubing WHP crosses p_wh (if Pwh is set)
        if well.Pwh is not None:
            def F(q):
                whp, _ = net._whp_at_q(well, max(q, 0.1))
                return whp - well.Pwh
            fa, fb = F(q_min + 1), F(q_max)
            if fa * fb < 0:
                a, b = q_min + 1, q_max
                for _ in range(80):
                    m = 0.5 * (a + b)
                    fm = F(m)
                    if abs(fm) < 0.5:
                        break
                    if fa * fm <= 0:
                        b = m
                    else:
                        a, fa = m, fm
                q_star   = 0.5 * (a + b)
                pwf_star = well.pwf_from_q(q_star)
    except Exception:
        pass

    return {
        'q_vals':     q_vals,
        'pwf_ipr':    pwf_ipr,
        'pwf_opr':    pwf_opr,
        'whp_tubing': whp_tubing,
        'q_star':     q_star,
        'pwf_star':   pwf_star,
    }

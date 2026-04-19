"""
network_components.py
=====================
Modular building blocks for the production network simulator.

Components
----------
- Node            : junction / boundary condition holder
- WellSegment     : single tubing segment with trajectory (survey data)
- Well            : composite wellbore (chain of WellSegments) + IPR
- FlowlinePipe    : surface flowline segment (H&B or B&B depending on angle)
- Choke           : fixed or variable positive-choke pressure drop
- Manifold        : simple junction (no pressure drop, just mass balance)
- ReservoirNode   : fixed-pressure boundary (Pr)
- SeparatorNode   : fixed-pressure sink

All pressure-drop elements expose:
    dp_psi(q_stbd, p_in_psia) -> float
so the solver can march pressure along a path.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from fluid_correlations import BlackOilFluid
from multiphase_correlations import get_correlation, HagedornBrown, BeggsBrill

G    = 32.174
GC   = 32.174
PSF_TO_PSI = 1.0 / 144.0


# ===========================================================================
# Node
# ===========================================================================

@dataclass
class Node:
    """
    Network junction.

    pressure_psi   : fixed if boundary condition, None if unknown
    """
    name: str
    pressure_psi: Optional[float] = None

    # For mass-balance bookkeeping
    source_rate_stbd: float = 0.0   # positive = injection into node


class ReservoirNode(Node):
    """Fixed-pressure reservoir boundary."""
    def __init__(self, name: str, pressure_psi: float):
        super().__init__(name, pressure_psi)


class SeparatorNode(Node):
    """Fixed-pressure sink."""
    def __init__(self, name: str, pressure_psi: float):
        super().__init__(name, pressure_psi)


class FixedRateSink(Node):
    """
    Fixed-rate sink node (e.g. separator, compressor, export terminal
    with a contractual or mechanical throughput limit).

    Instead of prescribing a backpressure, this node accepts a target
    total liquid rate [STB/D].  The solver will:
      1. Throttle individual wells (via virtual choke) so that the
         sum of well rates equals q_target_stbd.
      2. Compute the resulting inlet (manifold) pressure from the
         pipe pressure-drop equations at that constrained rate.

    The outlet pressure is recorded after solve as `pressure_psi`
    (it becomes a result, not an input).

    Parameters
    ----------
    name          : node identifier
    q_target_stbd : total liquid rate to be delivered [STB/D]
    p_outlet_psia : known or assumed outlet pressure; used only as a
                    reference for reporting (e.g. flare/export pressure).
                    Does NOT constrain the solve — set to None if unknown.
    """
    def __init__(self, name: str,
                 q_target_stbd: float,
                 p_outlet_psia: Optional[float] = None):
        super().__init__(name, pressure_psi=p_outlet_psia)
        self.q_target_stbd  = q_target_stbd
        self.p_outlet_psia  = p_outlet_psia   # reference only


class ManifoldNode(Node):
    """
    Simple junction – no pressure loss.
    Pressure unknown (to be solved) unless explicitly fixed.
    """
    def __init__(self, name: str, pressure_psi: Optional[float] = None):
        super().__init__(name, pressure_psi)


# ===========================================================================
# Well Trajectory Segment
# ===========================================================================

@dataclass
class SurveyPoint:
    """Single measured-depth survey point."""
    md_ft: float         # measured depth from surface [ft]
    inc_deg: float       # inclination from vertical [°]
    az_deg: float = 0.0  # azimuth (not used in pressure calc but stored)


@dataclass
class WellSegment:
    """
    One segment of the wellbore between two survey points.
    Uses minimum-curvature method for TVD.

    Attributes
    ----------
    md_start, md_end  : measured depths [ft]
    inc_start, inc_end: inclinations from vertical [°]
    dmd               : measured depth increment [ft]
    dtvd              : true vertical depth increment [ft] (positive = deeper)
    avg_inc_from_horiz: average inclination from horizontal [°]
    """
    md_start:  float
    md_end:    float
    inc_start: float
    inc_end:   float

    def __post_init__(self):
        self.dmd  = self.md_end - self.md_start

        # Minimum curvature TVD increment
        i1 = math.radians(self.inc_start)
        i2 = math.radians(self.inc_end)
        beta = math.acos(max(-1.0, min(1.0, math.cos(i2 - i1))))
        if abs(beta) < 1e-9:
            rf = 1.0
        else:
            rf = 2.0 / beta * math.tan(beta / 2.0)
        self.dtvd = self.dmd / 2.0 * (math.cos(i1) + math.cos(i2)) * rf

        # Average inclination from horizontal (for correlation swap)
        avg_inc_from_vert = (self.inc_start + self.inc_end) / 2.0
        self.avg_inc_from_horiz = 90.0 - avg_inc_from_vert  # deg from horizontal


class Well:
    """
    Complete wellbore: vertical + deviated sections defined by survey.
    Connected to a reservoir via a linear (Vogel / PI) IPR.

    Parameters
    ----------
    name          : well name
    fluid         : BlackOilFluid
    survey        : list of SurveyPoint from surface (MD=0) to TD
    ID_in         : tubing inner diameter [in]
    eps_in        : roughness [in]
    J_stbd_psi    : productivity index [STB/D/psi]
    Pr_psia       : static reservoir pressure
    Pwh_psia      : fixed wellhead pressure (outflow boundary)
    """

    def __init__(self,
                 name: str,
                 fluid: BlackOilFluid,
                 survey: List[SurveyPoint],
                 ID_in: float,
                 eps_in: float,
                 J_stbd_psi: float,
                 Pr_psia: float,
                 Pwh_psia: Optional[float] = None):
        self.name       = name
        self.fluid      = fluid
        self.survey     = survey
        self.ID_in      = ID_in
        self.eps_in     = eps_in
        self.J          = J_stbd_psi
        self.Pr         = Pr_psia
        self.Pwh        = Pwh_psia

        # Build segments from survey
        self.segments: List[WellSegment] = []
        for k in range(len(survey) - 1):
            seg = WellSegment(
                md_start  = survey[k].md_ft,
                md_end    = survey[k+1].md_ft,
                inc_start = survey[k].inc_deg,
                inc_end   = survey[k+1].inc_deg,
            )
            self.segments.append(seg)

        self.TVD_ft = sum(s.dtvd for s in self.segments)
        self.MD_ft  = sum(s.dmd  for s in self.segments)

    # -- IPR ----------------------------------------------------------------
    # ipr_model : 'linear' or 'vogel'
    # 'vogel'   : used when Pr <= Pb (reservoir at or below bubble point)
    #             q = qmax * [1 - 0.2*(pwf/Pr) - 0.8*(pwf/Pr)^2]
    #             where qmax = J*Pr/1.8   (Standing 1970, Pr=Pb case)
    # 'linear'  : q = J*(Pr - pwf)        (Pr > Pb)

    def _ipr_model(self) -> str:
        """Auto-select IPR model based on Pr vs Pb."""
        if hasattr(self, '_ipr_override'):
            return self._ipr_override
        Pb = self.fluid.bubble_point_psia()
        return 'vogel' if self.Pr < Pb else 'linear'

    def set_ipr_model(self, model: str):
        """Manually override IPR model: 'linear' or 'vogel'."""
        assert model in ('linear', 'vogel'), "model must be 'linear' or 'vogel'"
        self._ipr_override = model

    def pwf_from_q(self, q_stbd: float) -> float:
        """Compute pwf from rate using the appropriate IPR model."""
        if self._ipr_model() == 'vogel':
            # Invert Vogel: q = qmax*(1 - 0.2*x - 0.8*x²) where x = pwf/Pr
            # Solve quadratic: 0.8x² + 0.2x + (q/qmax - 1) = 0
            qmax = self.J * self.Pr / 1.8
            q    = max(0.0, min(q_stbd, qmax * 0.9999))
            ratio = q / max(qmax, 1e-6)
            # 0.8x² + 0.2x - (1 - ratio) = 0
            disc = 0.04 + 4 * 0.8 * (1.0 - ratio)
            x    = (-0.2 + math.sqrt(max(disc, 0.0))) / (2 * 0.8)
            x    = max(0.0, min(1.0, x))
            return x * self.Pr
        else:
            return max(self.Pr - q_stbd / self.J, 0.0)

    def q_from_pwf(self, pwf_psia: float) -> float:
        """Compute rate from pwf using the appropriate IPR model."""
        if self._ipr_model() == 'vogel':
            qmax = self.J * self.Pr / 1.8
            x    = pwf_psia / self.Pr
            return max(qmax * (1.0 - 0.2*x - 0.8*x**2), 0.0)
        else:
            return max(self.J * (self.Pr - pwf_psia), 0.0)

    def q_from_aof(self) -> float:
        """AOF at pwf = 0."""
        if self._ipr_model() == 'vogel':
            return self.J * self.Pr / 1.8
        else:
            return self.J * self.Pr

    # -- Tubing pressure profile (bottom-up march) -------------------------
    def tubing_dp_total_psi(self, q_stbd: float, p_bhp: float,
                             n_sub: int = 3) -> Tuple[float, List[dict]]:
        """
        March from BHP upward through all segments.
        Returns (total_dp_psi, profile_list).

        profile_list : list of dicts with keys
            md_ft, tvd_ft, pressure_psi, inc_deg
        """
        # Start from bottom
        p_curr = p_bhp
        tvd_cum = self.TVD_ft   # start at bottom TVD

        profile = [{'md_ft':   self.MD_ft,
                    'tvd_ft':  self.TVD_ft,
                    'pressure_psi': p_curr,
                    'inc_deg': self.segments[-1].inc_end if self.segments else 0}]

        # March upward (segments reversed: bottom to top)
        for seg in reversed(self.segments):
            # Sub-divide each survey segment for accuracy
            dmd_sub = seg.dmd / n_sub
            # Linearly interpolate inclination within segment
            for sub in range(n_sub):
                frac_a = sub / n_sub
                frac_b = (sub + 1) / n_sub
                inc_a  = seg.inc_start + frac_a * (seg.inc_end - seg.inc_start)
                inc_b  = seg.inc_start + frac_b * (seg.inc_end - seg.inc_start)
                sub_seg = WellSegment(0, dmd_sub, inc_a, inc_b)

                # Inclination from horizontal for correlation selection
                theta_from_horiz = sub_seg.avg_inc_from_horiz  # + = uphill

                # Get appropriate correlation
                corr = get_correlation(self.fluid, self.ID_in,
                                       theta_from_horiz, self.eps_in)

                # dP/dL at mid-segment pressure (floor p to avoid singularities)
                p_safe = max(p_curr, 14.7)
                try:
                    dpdl = corr.dpdl_psi_ft(q_stbd, p_safe)
                except (OverflowError, ZeroDivisionError, ValueError):
                    dpdl = 0.0

                # MD-based dp (correlation gives per ft of pipe length)
                dp_seg = dpdl * dmd_sub

                p_curr -= dp_seg   # pressure drops as we go upward
                p_curr = max(p_curr, 14.7)   # floor at atmospheric
                tvd_cum -= sub_seg.dtvd

                profile.append({'md_ft':   self.MD_ft - (seg.md_end - seg.md_start) * (1 - frac_b),
                                'tvd_ft':  max(tvd_cum, 0),
                                'pressure_psi': p_curr,
                                'inc_deg': inc_b})

        total_dp = p_bhp - p_curr
        profile.reverse()   # now top-to-bottom
        return total_dp, profile

    def wellhead_pressure(self, q_stbd: float) -> float:
        """Given q, compute wellhead pressure from IPR + tubing traverse."""
        p_bhp = self.pwf_from_q(q_stbd)
        dp_tub, _ = self.tubing_dp_total_psi(q_stbd, p_bhp)
        return p_bhp - dp_tub

    def solve_operating_point(self, p_wh_fixed: Optional[float] = None,
                               q_min: float = 1.0, q_max: float = 20000.0,
                               tol: float = 0.5) -> Tuple[float, float]:
        """
        Bisect to find q* where WHP from tubing traverse = p_wh_fixed.
        Returns (q_star, pwf_star).
        If p_wh_fixed is None, uses self.Pwh.
        """
        p_wh = p_wh_fixed if p_wh_fixed is not None else self.Pwh
        if p_wh is None:
            raise ValueError(f"Well '{self.name}': wellhead pressure not set.")

        # Cap q_max at AOF
        q_max = min(q_max, self.q_from_aof() * 0.999)

        def F(q):
            return self.wellhead_pressure(q) - p_wh

        fa, fb = F(q_min), F(q_max)
        if fa * fb > 0:
            q_max_try = q_max * 2
            fb2 = F(q_max_try)
            if fa * fb2 <= 0:
                q_max, fb = q_max_try, fb2
            else:
                raise ValueError(
                    f"Well '{self.name}': no bracket in [{q_min},{q_max}] STB/D. "
                    f"F(q_min)={fa:.1f}, F(q_max)={fb:.1f}")

        for _ in range(80):
            m = 0.5 * (q_min + q_max)
            fm = F(m)
            if abs(fm) < tol:
                break
            if fa * fm <= 0:
                q_max, fb = m, fm
            else:
                q_min, fa = m, fm

        q_star  = 0.5 * (q_min + q_max)
        pwf_star = self.pwf_from_q(q_star)
        return q_star, pwf_star


# ===========================================================================
# Flowline Pipe (surface / subsea)
# ===========================================================================

class FlowlinePipe:
    """
    Surface/subsea flowline segment.
    Uses Beggs-Brill for |theta| < 45° and H&B for |theta| >= 45°.

    Elevation profile is specified as (x_ft, z_ft) pairs relative to inlet.
    The pressure traverse marches along the pipe accounting for elevation.

    Parameters
    ----------
    name    : identifier
    u, v    : upstream / downstream node names
    fluid   : BlackOilFluid
    profile : list of (horiz_dist_ft, elevation_ft) from inlet
    ID_in   : inner diameter [in]
    eps_in  : roughness [in]
    """

    def __init__(self, name: str, u: str, v: str,
                 fluid: BlackOilFluid,
                 profile: List[Tuple[float, float]],
                 ID_in: float,
                 eps_in: float = 0.001):
        self.name   = name
        self.u      = u
        self.v      = v
        self.fluid  = fluid
        self.ID_in  = ID_in
        self.eps_in = eps_in

        # Build segments from elevation profile
        self.pipe_segments: List[dict] = []
        for k in range(len(profile) - 1):
            x1, z1 = profile[k]
            x2, z2 = profile[k+1]
            dx = x2 - x1
            dz = z2 - z1
            L  = math.sqrt(dx**2 + dz**2)
            theta_deg = math.degrees(math.atan2(dz, dx)) if L > 0 else 0.0
            self.pipe_segments.append({
                'L_ft':     L,
                'theta_deg': theta_deg,
                'dz_ft':    dz,
            })

        self.total_length_ft = sum(s['L_ft'] for s in self.pipe_segments)
        self.net_elevation_ft = profile[-1][1] - profile[0][1]

    def dp_psi(self, q_stbd: float, p_in_psia: float) -> Tuple[float, List[dict]]:
        """
        Pressure traverse from inlet to outlet.
        Returns (total_dp_psi, profile_list).
        dp > 0 means pressure drops from u to v.
        """
        p_curr = p_in_psia
        profile = [{'L_cum_ft': 0.0, 'pressure_psi': p_curr}]
        L_cum  = 0.0

        for seg in self.pipe_segments:
            corr = get_correlation(self.fluid, self.ID_in,
                                   seg['theta_deg'], self.eps_in)
            p_curr = max(p_curr, 14.7)
            try:
                dpdl = corr.dpdl_psi_ft(q_stbd, p_curr)
            except (OverflowError, ZeroDivisionError, ValueError):
                dpdl = 0.0
            dp   = dpdl * seg['L_ft']
            p_curr -= dp
            p_curr = max(p_curr, 14.7)
            L_cum  += seg['L_ft']
            profile.append({'L_cum_ft': L_cum, 'pressure_psi': p_curr})

        total_dp = p_in_psia - p_curr
        return total_dp, profile


# ===========================================================================
# Choke
# ===========================================================================

class Choke:
    """
    Fixed or variable positive choke.

    Model (subcritical, incompressible liquid, Gilbert-type):
        q = C * d^a * (dp)^b
    Rearranged to give dp from q:
        dp = (q / (C * d^a))^(1/b)

    Default Gilbert (1954) constants for liquid:
        C = 10.9,  a = 1.89,  b = 0.546   (q in bbl/d, d in 64ths inch, dp in psi)

    Alternatively: direct Cv-based dp if cv_gpm_sqrtpsi is provided.
    """
    def __init__(self, name: str, u: str, v: str,
                 bean_64ths: float = 16.0,
                 C_gilbert: float = 10.9,
                 a_gilbert: float = 1.89,
                 b_gilbert: float = 0.546,
                 cv_gpm_sqrtpsi: Optional[float] = None):
        self.name   = name
        self.u      = u
        self.v      = v
        self.bean   = bean_64ths
        self.C      = C_gilbert
        self.a      = a_gilbert
        self.b      = b_gilbert
        self.cv     = cv_gpm_sqrtpsi

    def dp_psi(self, q_stbd: float, p_in_psia: float = None) -> float:
        if self.cv is not None:
            # Cv model: q [gpm] = Cv * sqrt(dp)
            q_gpm = q_stbd * 0.29167   # STB/D -> gpm (approximate)
            return (q_gpm / self.cv)**2

        # Gilbert model
        if q_stbd <= 0:
            return 0.0
        dp = (q_stbd / (self.C * self.bean**self.a))**(1.0 / self.b)
        return max(dp, 0.0)


# ===========================================================================
# Manifold (zero-dp junction node)
# ===========================================================================

class Manifold:
    """
    Junction where multiple flowlines meet.
    No pressure drop – just a named node.
    Acts as a ManifoldNode in the network.
    """
    def __init__(self, name: str):
        self.name = name
        self.node = ManifoldNode(name)
        self.inlet_names:  List[str] = []
        self.outlet_names: List[str] = []

    def register_inlet(self, comp_name: str):
        self.inlet_names.append(comp_name)

    def register_outlet(self, comp_name: str):
        self.outlet_names.append(comp_name)

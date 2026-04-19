"""
multiphase_correlations.py
==========================
Pressure-gradient correlations for multiphase pipe flow.

Correlations implemented
------------------------
1. Hagedorn & Brown (1965)  – vertical upward flow  (theta >= 45°)
2. Beggs & Brill  (1973/1991 revised) – all inclinations   (theta < 45°)

Swap angle : 45° (per assignment specification)

Both return dP/dL in psi/ft (positive = pressure drop in direction of flow).

All quantities in consistent field-unit / SI-mixed set:
  q   : in-situ volumetric flow  [ft³/s]
  v   : velocity                  [ft/s]
  rho : density                   [lbm/ft³]
  mu  : viscosity                 [lbm/(ft·s)]   (converted from cp internally)
  g   : 32.174 ft/s²
  gc  : 32.174 lbm·ft/(lbf·s²)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from fluid_correlations import BlackOilFluid

G    = 32.174   # ft/s²
GC   = 32.174   # lbm·ft/(lbf·s²)
PSF_TO_PSI = 1.0 / 144.0

SWAP_ANGLE_DEG = 45.0   # degrees; use HB above, BB below


def cp_to_lbmfts(mu_cp: float) -> float:
    return mu_cp * 6.7197e-4


# ---------------------------------------------------------------------------
# Helper: superficial velocities & no-slip holdup
# ---------------------------------------------------------------------------

def _superficial_velocities(q_l_ft3s: float, q_g_ft3s: float, A_ft2: float):
    """Return (vsl, vsg, vm, lambda_l) – no gas in pure liquid case."""
    vsl = q_l_ft3s / A_ft2
    vsg = q_g_ft3s / A_ft2
    vm  = vsl + vsg
    lam = vsl / max(vm, 1e-12)
    return vsl, vsg, vm, lam


# ===========================================================================
# 1. Hagedorn & Brown (H&B) — vertical multiphase
# ===========================================================================

class HagedornBrown:
    """
    Hagedorn & Brown (1965) vertical multiphase pressure gradient.

    Notes
    -----
    - Uses the four H&B correlating groups to find liquid holdup H_L.
    - Applies Griffith-Wallis bubble-flow correction (common practice).
    - Returns total dP/dL (friction + hydrostatic) in psi/ft.
    - Assumes no gas for single-phase liquid (HL = 1).

    Parameters
    ----------
    fluid      : BlackOilFluid  instance
    ID_in      : pipe inner diameter [in]
    """

    def __init__(self, fluid: BlackOilFluid, ID_in: float):
        self.fluid = fluid
        self.d     = ID_in / 12.0          # ft
        self.A     = math.pi * self.d**2 / 4.0  # ft²

    def _gas_density(self, p_psia: float) -> float:
        """Gas density using Hall-Yarborough z-factor."""
        return self.fluid.gas_density_lbmft3(p_psia)

    def _gas_viscosity_cp(self, p_psia: float) -> float:
        """Gas viscosity using fluid's Lee-Kesler with H-Y z [cp]."""
        return self.fluid.gas_viscosity_cp(p_psia)

    def _liquid_holdup(self, vsl: float, vsg: float, vm: float,
                       p_psia: float,
                       rho_l: float, rho_g: float,
                       mu_l_lbmfts: float, sigma_l: float = 30.0) -> float:
        """
        H&B holdup via four correlating groups (H&B 1965).

        Valid range: NGv < ~20 (slug/churn flow at moderate gas rates).
        Above NGv = 20: Griffith-Wallis (1961) slug holdup is used instead,
        which gracefully handles the churn-to-mist transition and avoids
        the chart extrapolation that causes HL → 1.0 at high gas velocities.

        Griffith-Wallis:
            HL = 1 - (1 - λ_l) * sqrt(vsg / vm)   [bounded to (λ_l, 1.0)]
        """
        lam_l = vsl / max(vm, 1e-12)

        if vsg < 1e-6:
            return 1.0

        d  = self.d
        mu_l_cp = mu_l_lbmfts / 6.7197e-4

        # H&B correlating groups
        NLv = vsl * (rho_l / (GC * sigma_l))**0.25
        NGv = vsg * (rho_l / (GC * sigma_l))**0.25
        Nd  = d   * (rho_l * GC / sigma_l)**0.5
        NL  = mu_l_cp * (GC / (rho_l * sigma_l**3))**0.25

        # ── Griffith-Wallis fallback for high NGv (churn/mist) ─────────
        # H&B chart was built for NGv < ~20.  Above that, the chart
        # curves converge and holdup decreases with increasing vsg.
        # Griffith-Wallis captures this correctly.
        NGV_MAX = 20.0
        if NGv > NGV_MAX:
            HL_gw = 1.0 - (1.0 - lam_l) * math.sqrt(vsg / max(vm, 1e-6))
            return max(lam_l, min(1.0, HL_gw))

        # ── H&B chart (valid range NGv ≤ 20) ───────────────────────────
        log_NL = max(-3.0, min(0.0, math.log10(max(NL, 1e-4))))
        CNL = 10.0 ** (-2.69851 + 0.15840*log_NL
                       - 0.55099*log_NL**2 + 0.54784*log_NL**3)

        p_ref = 14.7
        G1 = NLv * (p_psia**0.1) * CNL / (max(NGv, 0.01)**0.575 * (p_ref**0.1) * Nd)
        log_G1 = max(-3.0, min(1.0, math.log10(max(G1, 1e-6))))
        HL_psi_ratio = 10.0 ** (1.0 - 0.43942*log_G1 + 0.14672*log_G1**2)
        HL_psi_ratio = min(max(HL_psi_ratio, 0.01), 1.0)

        G2     = (NGv * (p_psia / p_ref)**0.1) / Nd
        log_G2 = math.log10(max(G2, 1e-6))
        psi    = 1.0 + 0.3 * math.tanh(2.0 * log_G2 + 2.0)
        psi    = max(1.0, min(psi, 1.8))

        HL = HL_psi_ratio * psi
        # Blend smoothly to Griffith-Wallis as NGv → NGV_MAX
        blend  = (NGv / NGV_MAX)**2          # 0→1 as NGv→NGV_MAX
        HL_gw  = 1.0 - (1.0 - lam_l) * math.sqrt(vsg / max(vm, 1e-6))
        HL_gw  = max(lam_l, min(1.0, HL_gw))
        HL     = (1.0 - blend) * HL + blend * HL_gw

        return min(max(HL, lam_l), 1.0)

    def dpdl_psi_ft(self, q_total_stbd: float, p_psia: float) -> float:
        """
        Total dP/dL [psi/ft] for vertical upward flow.

        Flow regime handling
        --------------------
        Bubble/slug  (vsg < v_crit) : Griffith-Wallis + H&B holdup
        Mist/annular (vsg >= v_crit): Wallis annular mist model
            - HL = λ_l  (no-slip, liquid dispersed as droplets in gas core)
            - friction based on gas phase with liquid entrainment correction

        Turner critical velocity separates slug from annular/mist:
            v_crit = 5.02 * [σ(ρ_l - ρ_g) / ρ_g²]^0.25   [ft/s]
        """
        fluid = self.fluid

        q_l   = fluid.liquid_rate_ft3s(q_total_stbd, p_psia)
        rho_l = fluid.mixture_density_lbmft3(p_psia)
        mu_l  = cp_to_lbmfts(fluid.mixture_viscosity_cp(p_psia))

        Rs_sc    = fluid.solution_gor(p_psia)
        free_gor = max(fluid.gor_scf_stb - Rs_sc, 0.0)
        q_o_stbd = q_total_stbd * (1.0 - fluid.wc)
        Bg_ft3_scf = fluid.gas_fvf_ft3_scf(p_psia)
        q_g  = q_o_stbd * free_gor * Bg_ft3_scf / 86400.0 * 5.614583

        rho_g = self._gas_density(p_psia)
        mu_g  = cp_to_lbmfts(self._gas_viscosity_cp(p_psia))

        vsl, vsg, vm, lam_l = _superficial_velocities(q_l, q_g, self.A)

        # Single-phase liquid (no free gas)
        if vsg < 1e-4:
            Re_sl = rho_l * vsl * self.d / max(mu_l, 1e-20)
            f_sl  = 64.0/max(Re_sl,1) if Re_sl < 2100 else 0.3164/max(Re_sl,1)**0.25
            dpdl_fric = f_sl * rho_l * vsl**2 / (2.0 * GC * self.d)
            dpdl_hyd  = rho_l * G / GC
            return (dpdl_fric + dpdl_hyd) * PSF_TO_PSI

        # Turner critical gas velocity for mist flow transition [ft/s]
        sigma_l = 30.0   # surface tension, dynes/cm ≈ lbm/s²·ft * 0.00685
        # Turner (1969): v_crit = 5.02 * [sigma*(rho_l-rho_g)/rho_g²]^0.25
        # (coefficients tuned for field units: sigma in dynes/cm, rho in lbm/ft³)
        sigma_si = sigma_l * 6.852e-3        # dynes/cm → lbm/s²
        drho     = max(rho_l - rho_g, 1.0)
        v_turner = 5.02 * (sigma_si * drho / max(rho_g, 0.01)**2)**0.25

        # ── MIST / ANNULAR FLOW ───────────────────────────────────────────
        if vsg >= v_turner:
            # Liquid holdup = no-slip (droplets entrained in gas)
            HL    = lam_l
            HL    = max(HL, 1e-6)

            # Mixture density
            rho_m = rho_l * HL + rho_g * (1.0 - HL)

            # Friction: gas-dominated, Moody on gas Reynolds
            # Wallis (1969) annular friction with entrainment factor
            Re_g  = rho_g * vsg * self.d / max(mu_g, 1e-20)
            if Re_g < 2100:
                f_g = 64.0 / max(Re_g, 1.0)
            else:
                f_g = 0.3164 / max(Re_g, 1.0)**0.25

            # Wallis entrainment correction: f_tp = f_g * (1 + 75*HL)
            f_tp = f_g * (1.0 + 75.0 * HL)

            dpdl_fric = f_tp * rho_g * vsg**2 / (2.0 * GC * self.d)
            dpdl_hyd  = rho_m * G / GC

            return max((dpdl_fric + dpdl_hyd) * PSF_TO_PSI, 0.0)

        # ── BUBBLE / SLUG FLOW (H&B) ──────────────────────────────────────
        LB = 1.071 - 0.2218 * vm**2 / self.d
        LB = max(LB, 0.25)

        if lam_l >= LB:
            HL = 1.0
        else:
            HL = self._liquid_holdup(vsl, vsg, vm, p_psia, rho_l, rho_g, mu_l)
            # Clamp: holdup cannot be less than no-slip holdup
            HL = max(HL, lam_l)

        rho_m = rho_l * HL + rho_g * (1.0 - HL)
        mu_m  = mu_l * lam_l + mu_g * (1.0 - lam_l)
        v_m   = vm

        Re = rho_m * v_m * self.d / max(mu_m, 1e-20)
        f_m = 64.0 / max(Re, 1.0) if Re < 2100 else 0.3164 / max(Re, 1.0)**0.25

        dpdl_fric = f_m * rho_m * v_m**2 / (2.0 * GC * self.d)
        dpdl_hyd  = rho_m * G / GC

        return max((dpdl_fric + dpdl_hyd) * PSF_TO_PSI, 0.0)

# ===========================================================================
# 2. Beggs & Brill (1973 / Payne correction 1979)
# ===========================================================================

class BeggsBrill:
    """
    Beggs & Brill (1973) multiphase flow correlation, all inclinations.
    Payne et al. (1979) holdup corrections applied.

    Parameters
    ----------
    fluid   : BlackOilFluid
    ID_in   : pipe inner diameter [in]
    theta   : pipe inclination from horizontal [deg], positive = uphill
    eps_in  : absolute roughness [in]
    """

    # Flow regime boundaries (from Table in Beggs & Brill 1973)
    _L1 = staticmethod(lambda lam: 316.0  * max(lam, 1e-9)**0.302)
    _L2 = staticmethod(lambda lam: 0.0009252 * max(lam, 1e-9)**(-2.4684))
    _L3 = staticmethod(lambda lam: 0.1   * max(lam, 1e-9)**(-1.4516))
    _L4 = staticmethod(lambda lam: 0.5   * max(lam, 1e-9)**(-6.738))

    def __init__(self, fluid: BlackOilFluid, ID_in: float,
                 theta_deg: float = 0.0, eps_in: float = 0.001):
        self.fluid = fluid
        self.d     = ID_in / 12.0
        self.A     = math.pi * self.d**2 / 4.0
        self.theta = math.radians(theta_deg)
        self.sin_t = math.sin(self.theta)
        self.eps   = eps_in / 12.0

    # -- gas properties: delegate to fluid (uses Hall-Yarborough z) ---------
    def _gas_density(self, p_psia: float) -> float:
        return self.fluid.gas_density_lbmft3(p_psia)

    def _gas_viscosity_cp(self, p_psia: float) -> float:
        return self.fluid.gas_viscosity_cp(p_psia)

    # -- holdup at horizontal (θ=0) -----------------------------------------
    def _EL_horizontal(self, lam: float, NFr: float) -> float:
        """
        Beggs-Brill horizontal holdup EL(0).
        Uses flow-regime-based coefficients.
        """
        L1 = self._L1(lam)
        L2 = self._L2(lam)
        L3 = self._L3(lam)
        L4 = self._L4(lam)

        # Determine flow regime
        if lam < 0.01 and NFr < L1:
            regime = 'segregated'
        elif lam >= 0.01 and NFr < L2:
            regime = 'segregated'
        elif lam >= 0.01 and L2 <= NFr <= L3:
            regime = 'transition'
        elif (0.01 <= lam < 0.4) and (L3 < NFr <= L1):
            regime = 'intermittent'
        elif lam >= 0.4 and (L3 < NFr <= L4):
            regime = 'intermittent'
        else:
            regime = 'distributed'

        # Coefficient table (Beggs & Brill 1973, Table 2)
        coeff = {
            'segregated':   (0.980,  0.4846,  0.0868),
            'intermittent': (0.845,  0.5351,  0.0173),
            'distributed':  (1.065,  0.5824,  0.0609),
        }

        if regime == 'transition':
            # Interpolate between segregated and intermittent
            A_t = (L3 - NFr) / max(L3 - L2, 1e-12)
            B_t = 1.0 - A_t
            EL_seg = self._EL_regime(lam, NFr, *coeff['segregated'])
            EL_int = self._EL_regime(lam, NFr, *coeff['intermittent'])
            return A_t * EL_seg + B_t * EL_int
        else:
            return self._EL_regime(lam, NFr, *coeff[regime])

    @staticmethod
    def _EL_regime(lam: float, NFr: float, a: float, b: float, c: float) -> float:
        EL = a * lam**b / NFr**c
        return min(max(EL, lam), 1.0)

    # -- inclination correction (Payne 1979) --------------------------------
    def _inclination_factor(self, lam: float, NFr: float,
                            NLv: float, EL0: float) -> float:
        """
        C factor for inclination.  Payne et al. (1979) correction.
        """
        theta = self.theta
        if abs(math.degrees(theta)) < 0.1:
            return 1.0

        if theta > 0:   # uphill
            d1, d2, d3, d4 = 0.011, -3.768, 3.539, -1.614
        else:           # downhill
            d1, d2, d3, d4 = 4.70, -0.3692, 0.1244, -0.5056

        C = max(0.0, (1.0 - lam) * math.log(
            max(abs(d1 * max(lam, 1e-9)**d2 * max(NFr, 1e-9)**d3 * max(NLv, 1e-9)**d4), 1e-10)))
        psi_incl = 1.0 + C * (math.sin(1.8 * theta) - math.sin(1.8 * theta)**3 / 3.0)
        return max(psi_incl, 0.0)

    def dpdl_psi_ft(self, q_total_stbd: float, p_psia: float) -> float:
        """
        Total dP/dL [psi/ft] for multiphase flow at given inclination.
        """
        fluid = self.fluid

        q_l   = fluid.liquid_rate_ft3s(q_total_stbd, p_psia)
        rho_l = fluid.mixture_density_lbmft3(p_psia)
        mu_l  = cp_to_lbmfts(fluid.mixture_viscosity_cp(p_psia))

        # Gas
        Rs_sc    = fluid.solution_gor(p_psia)
        free_gor = max(fluid.gor_scf_stb - Rs_sc, 0.0)
        q_o_stbd = q_total_stbd * (1.0 - fluid.wc)
        # Use Hall-Yarborough z-factor (from fluid) instead of fixed z=0.85
        Bg_ft3_scf = fluid.gas_fvf_ft3_scf(p_psia)
        q_g = q_o_stbd * free_gor * Bg_ft3_scf / 86400.0 * 5.614583

        rho_g = self._gas_density(p_psia)
        mu_g  = cp_to_lbmfts(self._gas_viscosity_cp(p_psia))

        vsl, vsg, vm, lam_l = _superficial_velocities(q_l, q_g, self.A)

        # Froude number
        NFr  = vm**2 / (G * self.d)
        # Liquid velocity number
        sigma_l = 30.0  # dynes/cm
        NLv = vsl * (rho_l / (G * sigma_l))**0.25

        # Horizontal holdup
        EL0 = self._EL_horizontal(lam_l, NFr)

        # Inclination correction
        beta = self._inclination_factor(lam_l, NFr, NLv, EL0)
        EL   = EL0 * beta
        EL   = min(max(EL, lam_l), 1.0)

        # Payne correction for upward flow (multiply EL by 0.924 for uphill)
        if self.theta > 0:
            EL = min(EL * 0.924, 1.0)
        elif self.theta < 0:
            EL = min(EL * 0.685, 1.0)

        # Mixture density
        rho_m  = rho_l * EL + rho_g * (1.0 - EL)
        # No-slip mixture density (for friction)
        rho_ns = rho_l * lam_l + rho_g * (1.0 - lam_l)

        # Mixture viscosity (no-slip, for Re) – log-weighted, safe
        mu_m  = math.exp(lam_l * math.log(max(mu_l, 1e-20))
                         + (1.0 - lam_l) * math.log(max(mu_g, 1e-20)))

        # Reynolds & friction
        Re = rho_ns * vm * self.d / max(mu_m, 1e-20)
        eps_rel = self.eps / self.d

        if Re < 2100:
            f_ns = 64.0 / max(Re, 1.0)
        else:
            # Colebrook-White
            def colebrook(f_):
                return -2.0 * math.log10(eps_rel / 3.7 + 2.51 / (Re * math.sqrt(f_))) - 1.0 / math.sqrt(f_)
            # Initial Swamee-Jain
            f_ns = (0.25 / (math.log10(eps_rel / 3.7 + 5.74 / Re**0.9))**2)
            for _ in range(5):
                f_ns = (1.0 / (-2.0 * math.log10(eps_rel / 3.7 + 2.51 / (Re * math.sqrt(f_ns)))))**2

        # B&B friction factor ratio (y-function)
        y = lam_l / max(EL, 1e-6)**2
        if 1.0 < y < 1.2:
            s = math.log(2.2 * y - 1.2)
        else:
            ln_y = math.log(max(y, 1e-6))
            s = ln_y / (-0.0523 + 3.182 * ln_y - 0.8725 * ln_y**2 + 0.01853 * ln_y**4)
        f_ratio = math.exp(s)
        f_tp    = f_ns * min(f_ratio, 2.5)

        # Pressure gradient components
        dpdl_fric = f_tp * rho_ns * vm**2 / (2.0 * GC * self.d)
        dpdl_hyd  = rho_m * G * self.sin_t / GC
        dpdl_acc  = 0.0   # acceleration term (negligible for liquid-dominated)

        total_psf = dpdl_fric + dpdl_hyd + dpdl_acc
        return total_psf * PSF_TO_PSI


# ===========================================================================
# 3. Dispatcher: pick correlation by inclination
# ===========================================================================

def get_correlation(fluid: BlackOilFluid, ID_in: float, theta_deg: float,
                    eps_in: float = 0.001):
    """
    Return the appropriate correlation object based on inclination angle.
    |theta| >= SWAP_ANGLE_DEG  -> HagedornBrown (treated as near-vertical)
    |theta| <  SWAP_ANGLE_DEG  -> BeggsBrill
    """
    if abs(theta_deg) >= SWAP_ANGLE_DEG:
        return HagedornBrown(fluid, ID_in)
    else:
        return BeggsBrill(fluid, ID_in, theta_deg, eps_in)

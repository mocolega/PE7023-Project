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

def _superficial_velocities(q_l_ft3s: float, q_g_ft3s: float, area_ft2: float):
    """
    Returns:
        vsl, vsg, vm, lambda_l
    """
    vsl = q_l_ft3s / max(area_ft2, 1e-20)
    vsg = q_g_ft3s / max(area_ft2, 1e-20)
    vm = vsl + vsg
    lam_l = vsl / max(vm, 1e-20)
    return vsl, vsg, vm, lam_l


# ===========================================================================
# 1. Hagedorn & Brown (H&B) — vertical multiphase
# ===========================================================================

class HagedornBrown:
    """
    Modified Hagedorn & Brown vertical multiphase pressure-gradient model.

    Structure
    ---------
    1) Pure liquid: single-phase liquid friction + hydrostatic
    2) Bubble flow: Griffith & Wallis modification
    3) Mist flow: Duns & Ros style mist fallback
    4) Otherwise: Hagedorn & Brown pseudo-holdup core
    5) Friction from Moody/Colebrook using a two-phase Reynolds number
    6) Optional acceleration correction

    Notes
    -----
    - Intended for vertical upward flow only.
    - Uses a smooth approximation to the H&B holdup chart.
    - This is a practical engineering implementation of
      "Hagedorn & Brown + standard modifications."
    """

    def __init__(self, fluid: BlackOilFluid, ID_in: float, eps_in: float = 0.001):
        self.fluid = fluid
        self.d = ID_in / 12.0                      # ft
        self.A = math.pi * self.d**2 / 4.0        # ft^2
        self.eps = eps_in / 12.0                  # ft

    # ------------------------------------------------------------------
    # Fluid accessors
    # ------------------------------------------------------------------

    def _gas_density(self, p_psia: float) -> float:
        return self.fluid.gas_density_lbmft3(p_psia)

    def _gas_viscosity_cp(self, p_psia: float) -> float:
        return self.fluid.gas_viscosity_cp(p_psia)

    # ------------------------------------------------------------------
    # Friction factor
    # ------------------------------------------------------------------

    @staticmethod
    def _colebrook_white(Re: float, eps_rel: float) -> float:
        """
        Moody/Colebrook friction factor.
        """
        if Re < 2100.0:
            return 64.0 / max(Re, 1.0)

        # Swamee-Jain initial guess
        f = 0.25 / (math.log10(eps_rel / 3.7 + 5.74 / Re**0.9) ** 2)

        for _ in range(12):
            rhs = -2.0 * math.log10(eps_rel / 3.7 + 2.51 / (Re * math.sqrt(f)))
            f_new = 1.0 / (rhs * rhs)
            if abs(f_new - f) < 1e-10:
                return f_new
            f = f_new

        return f

    # ------------------------------------------------------------------
    # Standard modifications
    # ------------------------------------------------------------------

    @staticmethod
    def _griffith_wallis_holdup(vsl: float, vsg: float, vm: float) -> float:
        """
        Simple Griffith & Wallis style slug/bubble holdup expression.
        Bounded so HL >= lambda_l.
        """
        lam_l = vsl / max(vm, 1e-20)
        HL = 1.0 - (1.0 - lam_l) * math.sqrt(vsg / max(vm, 1e-20))
        return min(max(HL, lam_l), 1.0)

    # ------------------------------------------------------------------
    # Hagedorn & Brown core pseudo-holdup
    # ------------------------------------------------------------------

    def _hb_holdup_core(
        self,
        vsl: float,
        vsg: float,
        vm: float,
        p_psia: float,
        rho_l: float,
        mu_l_cp: float,
        sigma_l: float = 30.0,
    ) -> float:
        """
        Smooth H&B pseudo-holdup approximation based on the classic
        four correlating groups.

        Returns H_L bounded between lambda_l and 1.
        """
        lam_l = vsl / max(vm, 1e-20)

        if vsg < 1e-12:
            return 1.0

        # H&B dimensionless groups in field-units style form
        NLv = vsl * (rho_l / max(GC * sigma_l, 1e-20))**0.25
        NGv = vsg * (rho_l / max(GC * sigma_l, 1e-20))**0.25
        Nd  = self.d * (rho_l * GC / max(sigma_l, 1e-20))**0.5
        NL  = mu_l_cp * (GC / max(rho_l * sigma_l**3, 1e-20))**0.25

        # Viscosity correction factor
        log_NL = math.log10(max(NL, 1e-6))
        log_NL = min(max(log_NL, -4.0), 1.0)

        CNL = 10.0 ** (
            -2.69851
            + 0.15840 * log_NL
            - 0.55099 * log_NL**2
            + 0.54784 * log_NL**3
        )

        # Pressure correction normalized to atmospheric reference
        p_ref = 14.7

        # Group-1 style quantity
        G1 = (
            NLv
            * (p_psia / p_ref)**0.1
            * CNL
            / max(NGv**0.575 * Nd, 1e-20)
        )

        # Smooth approximation of chart for HL/psi
        log_G1 = math.log10(max(G1, 1e-8))
        log_G1 = min(max(log_G1, -5.0), 2.0)

        log_HL_over_psi = (
            -0.001864 * log_G1**3
            - 0.025194 * log_G1**2
            + 0.113419 * log_G1
            - 0.138397
        )

        HL_over_psi = min(max(10.0**log_HL_over_psi, 0.01), 1.0)

        # Group-2 style psi factor
        G2 = NGv * (p_psia / p_ref)**0.1 / max(Nd, 1e-20)
        log_G2 = math.log10(max(G2, 1e-8))

        psi = 1.0 + 0.3 * math.tanh(2.0 * log_G2 + 2.0)
        psi = min(max(psi, 1.0), 1.8)

        HL = HL_over_psi * psi
        HL = min(max(HL, lam_l), 1.0)

        return HL

    # ------------------------------------------------------------------
    # Flow regime selection for standard modifications
    # ------------------------------------------------------------------

    def _liquid_holdup(
        self,
        vsl: float,
        vsg: float,
        vm: float,
        p_psia: float,
        rho_l: float,
        rho_g: float,
        mu_l_cp: float,
        sigma_l: float = 30.0,
    ) -> tuple[float, str]:
        """
        Returns:
            HL, regime

        Regime logic used here:
        - bubble: Griffith & Wallis correction
        - mist: Duns & Ros style fallback (HL = lambda_l)
        - else: H&B core
        """
        lam_l = vsl / max(vm, 1e-20)

        if vsg < 1e-10:
            return 1.0, "liquid"

        # Common velocity numbers for regime logic
        NGv = vsg * (rho_l / max(G * sigma_l, 1e-20))**0.25
        Ns  = vm  * (rho_l / max(G * sigma_l, 1e-20))**0.25

        # Griffith & Wallis bubble criterion
        is_bubble = (vsg < 0.25 * vm) and (Ns < 4.0)
        if is_bubble:
            return 1.0, "bubble"

        # Duns & Ros style mist threshold
        NGV_MIST = 60.0
        if NGv > NGV_MIST:
            return max(lam_l, 1e-6), "mist"

        # Standard H&B core elsewhere
        HL = self._hb_holdup_core(
            vsl=vsl,
            vsg=vsg,
            vm=vm,
            p_psia=p_psia,
            rho_l=rho_l,
            mu_l_cp=mu_l_cp,
            sigma_l=sigma_l,
        )
        return HL, "hb"

    # ------------------------------------------------------------------
    # Main pressure gradient
    # ------------------------------------------------------------------

    def dpdl_psi_ft(self, q_total_stbd: float, p_psia: float, include_acceleration: bool = True) -> float:
        """
        Total pressure gradient [psi/ft] for vertical upward flow.
        """
        fluid = self.fluid
        sigma_l = 30.0

        # -------------------------
        # Liquid
        # -------------------------
        q_l = fluid.liquid_rate_ft3s(q_total_stbd, p_psia)
        rho_l = fluid.mixture_density_lbmft3(p_psia)
        mu_l_cp = fluid.mixture_viscosity_cp(p_psia)
        mu_l = cp_to_lbmfts(mu_l_cp)

        # -------------------------
        # Gas
        # -------------------------
        Rs_sc = fluid.solution_gor(p_psia)
        free_gor = max(fluid.gor_scf_stb - Rs_sc, 0.0)
        q_o_stbd = q_total_stbd * (1.0 - fluid.wc)
        Bg_ft3_scf = fluid.gas_fvf_ft3_scf(p_psia)

        # scf/day * ft3/scf / day->s = ft3/s
        q_g = q_o_stbd * free_gor * Bg_ft3_scf / 86400.0

        rho_g = self._gas_density(p_psia)
        mu_g_cp = self._gas_viscosity_cp(p_psia)
        mu_g = cp_to_lbmfts(mu_g_cp)

        # -------------------------
        # Velocities
        # -------------------------
        vsl, vsg, vm, lam_l = _superficial_velocities(q_l, q_g, self.A)

        # -------------------------
        # Pure liquid
        # -------------------------
        if vsg < 1e-10:
            Re = rho_l * vsl * self.d / max(mu_l, 1e-20)
            f = self._colebrook_white(Re, self.eps / self.d)

            dpdl_fric = f * rho_l * vsl**2 / (2.0 * GC * self.d)
            dpdl_elev = rho_l * G / GC
            total = dpdl_fric + dpdl_elev
            return total * PSF_TO_PSI

        # -------------------------
        # Holdup
        # -------------------------
        HL, regime = self._liquid_holdup(
            vsl=vsl,
            vsg=vsg,
            vm=vm,
            p_psia=p_psia,
            rho_l=rho_l,
            rho_g=rho_g,
            mu_l_cp=mu_l_cp,
            sigma_l=sigma_l,
        )

        # Bubble modification
        if regime == "bubble":
            HL = 1.0

        # Mist fallback
        elif regime == "mist":
            HL = max(lam_l, 1e-6)
        else:
            HL = min(max(HL, lam_l), 1.0)

        # -------------------------
        # Mixture properties
        # -------------------------
        rho_m = rho_l * HL + rho_g * (1.0 - HL)

        # H&B friction uses a two-phase Reynolds-number style treatment.
        # Practical implementation: no-slip mixture viscosity for Re.
        mu_tp = math.exp(
            lam_l * math.log(max(mu_l, 1e-30))
            + (1.0 - lam_l) * math.log(max(mu_g, 1e-30))
        )

        Re_tp = rho_m * vm * self.d / max(mu_tp, 1e-20)
        f_tp = self._colebrook_white(Re_tp, self.eps / self.d)

        # Mist-flow adjustment: use gas-dominated friction in mist
        if regime == "mist":
            Re_g = rho_g * max(vsg, 1e-12) * self.d / max(mu_g, 1e-20)
            f_g = self._colebrook_white(Re_g, self.eps / self.d)
            f_tp = f_g

        # -------------------------
        # Pressure-gradient terms
        # -------------------------
        dpdl_fric = f_tp * rho_m * vm**2 / (2.0 * GC * self.d)
        dpdl_elev = rho_m * G / GC

        if include_acceleration:
            # Practical kinetic-energy correction
            Ek = rho_m * vm * vsg / (GC * max(p_psia * 144.0, 1e-20))
            Ek = min(max(Ek, 0.0), 0.95)
            total = (dpdl_fric + dpdl_elev) / (1.0 - Ek)
        else:
            Ek = 0.0
            total = dpdl_fric + dpdl_elev

        return total * PSF_TO_PSI

    # ------------------------------------------------------------------
    # Debug details
    # ------------------------------------------------------------------

    def details(self, q_total_stbd: float, p_psia: float, include_acceleration: bool = True) -> dict:
        fluid = self.fluid
        sigma_l = 30.0

        q_l = fluid.liquid_rate_ft3s(q_total_stbd, p_psia)
        rho_l = fluid.mixture_density_lbmft3(p_psia)
        mu_l_cp = fluid.mixture_viscosity_cp(p_psia)
        mu_l = cp_to_lbmfts(mu_l_cp)

        Rs_sc = fluid.solution_gor(p_psia)
        free_gor = max(fluid.gor_scf_stb - Rs_sc, 0.0)
        q_o_stbd = q_total_stbd * (1.0 - fluid.wc)
        Bg_ft3_scf = fluid.gas_fvf_ft3_scf(p_psia)
        q_g = q_o_stbd * free_gor * Bg_ft3_scf / 86400.0

        rho_g = self._gas_density(p_psia)
        mu_g_cp = self._gas_viscosity_cp(p_psia)
        mu_g = cp_to_lbmfts(mu_g_cp)

        vsl, vsg, vm, lam_l = _superficial_velocities(q_l, q_g, self.A)
        HL, regime = self._liquid_holdup(vsl, vsg, vm, p_psia, rho_l, rho_g, mu_l_cp, sigma_l)

        if regime == "bubble":
            HL = 1.0
        elif regime == "mist":
            HL = max(lam_l, 1e-6)

        HL = min(max(HL, lam_l), 1.0)

        rho_m = rho_l * HL + rho_g * (1.0 - HL)
        mu_tp = math.exp(
            lam_l * math.log(max(mu_l, 1e-30))
            + (1.0 - lam_l) * math.log(max(mu_g, 1e-30))
        )

        Re_tp = rho_m * vm * self.d / max(mu_tp, 1e-20)
        f_tp = self._colebrook_white(Re_tp, self.eps / self.d)

        if regime == "mist":
            Re_g = rho_g * max(vsg, 1e-12) * self.d / max(mu_g, 1e-20)
            f_tp = self._colebrook_white(Re_g, self.eps / self.d)

        dpdl_fric = f_tp * rho_m * vm**2 / (2.0 * GC * self.d)
        dpdl_elev = rho_m * G / GC

        if include_acceleration:
            Ek = rho_m * vm * vsg / (GC * max(p_psia * 144.0, 1e-20))
            Ek = min(max(Ek, 0.0), 0.95)
            total = (dpdl_fric + dpdl_elev) / (1.0 - Ek)
        else:
            Ek = 0.0
            total = dpdl_fric + dpdl_elev

        NGv = vsg * (rho_l / max(G * sigma_l, 1e-20))**0.25
        Ns  = vm  * (rho_l / max(G * sigma_l, 1e-20))**0.25

        return {
            "q_l_ft3s": q_l,
            "q_g_ft3s": q_g,
            "vsl_fts": vsl,
            "vsg_fts": vsg,
            "vm_fts": vm,
            "lambda_l": lam_l,
            "rho_l": rho_l,
            "rho_g": rho_g,
            "rho_m": rho_m,
            "mu_l_cp": mu_l_cp,
            "mu_g_cp": mu_g_cp,
            "HL": HL,
            "regime": regime,
            "NGv": NGv,
            "Ns": Ns,
            "Re_tp": Re_tp,
            "f_tp": f_tp,
            "Ek": Ek,
            "dpdl_fric_psf_per_ft": dpdl_fric,
            "dpdl_elev_psf_per_ft": dpdl_elev,
            "dpdl_total_psi_per_ft": total * PSF_TO_PSI,
        }
# ===========================================================================
# 2. Beggs & Brill (1973 / Payne correction 1979)
# ===========================================================================

# ===========================================================================
# 2. Beggs & Brill (1973) with Payne et al. rough-pipe / holdup corrections
# ===========================================================================

class BeggsBrill:
    """
    Beggs & Brill (1973) multiphase flow correlation for all inclinations.
    Payne et al. (1979) holdup corrections applied.

    Parameters
    ----------
    fluid   : BlackOilFluid
    ID_in   : pipe inner diameter [in]
    theta_deg : pipe inclination from horizontal [deg]
                positive = uphill, negative = downhill
    eps_in  : absolute roughness [in]
    """

    # Horizontal flow regime boundaries
    _L1 = staticmethod(lambda lam: 316.0      * max(lam, 1e-12)**0.302)
    _L2 = staticmethod(lambda lam: 0.0009252  * max(lam, 1e-12)**(-2.4684))
    _L3 = staticmethod(lambda lam: 0.1        * max(lam, 1e-12)**(-1.4516))
    _L4 = staticmethod(lambda lam: 0.5        * max(lam, 1e-12)**(-6.738))

    # Horizontal liquid holdup coefficients: EL(0) = a * lam^b / NFr^c
    _HOLdup_COEFF = {
        "segregated":   (0.980, 0.4846, 0.0868),
        "intermittent": (0.845, 0.5351, 0.0173),
        "distributed":  (1.065, 0.5824, 0.0609),
    }

    # Inclination correction coefficients:
    # beta = (1-lam) * ln(d * lam^e * NLv^f * NFr^g)
    # Note: distributed uses beta = 0  -> psi = 1
    _INCL_COEFF_UPHILL = {
        "segregated":   (0.0110, -3.7680,  3.5390, -1.6140),
        "intermittent": (2.9600,  0.3050, -0.4473,  0.0978),
        "distributed":  (0.0,     0.0,     0.0,     0.0),
    }

    _INCL_COEFF_DOWNHILL = {
        "segregated":   (4.7000, -0.3692,  0.1244, -0.5056),
        "intermittent": (4.7000, -0.3692,  0.1244, -0.5056),
        "distributed":  (0.0,     0.0,     0.0,     0.0),
    }

    def __init__(
        self,
        fluid,
        ID_in: float,
        theta_deg: float = 0.0,
        eps_in: float = 0.001,
    ):
        self.fluid = fluid
        self.d = ID_in / 12.0
        self.A = math.pi * self.d**2 / 4.0
        self.theta_deg = theta_deg
        self.theta = math.radians(theta_deg)
        self.sin_t = math.sin(self.theta)
        self.eps = eps_in / 12.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gas_density(self, p_psia: float) -> float:
        return self.fluid.gas_density_lbmft3(p_psia)

    def _gas_viscosity_cp(self, p_psia: float) -> float:
        return self.fluid.gas_viscosity_cp(p_psia)

    @staticmethod
    def _safe_pow(x: float, p: float, floor: float = 1e-12) -> float:
        return max(x, floor) ** p

    @staticmethod
    def _colebrook_white(Re: float, eps_rel: float) -> float:
        if Re < 2100.0:
            return 64.0 / max(Re, 1.0)

        # Swamee-Jain initial guess
        f = 0.25 / (math.log10(eps_rel / 3.7 + 5.74 / Re**0.9) ** 2)

        for _ in range(12):
            rhs = -2.0 * math.log10(eps_rel / 3.7 + 2.51 / (Re * math.sqrt(f)))
            f_new = 1.0 / (rhs * rhs)
            if abs(f_new - f) < 1e-10:
                return f_new
            f = f_new

        return f

    @staticmethod
    def _friction_ratio_s(y: float) -> float:
        """
        Beggs & Brill friction-factor ratio function.
        """
        y = max(y, 1e-12)

        if 1.0 < y < 1.2:
            return math.log(2.2 * y - 1.2)

        ln_y = math.log(y)
        denom = (
            -0.0523
            + 3.182 * ln_y
            - 0.8725 * ln_y**2
            + 0.01853 * ln_y**4
        )

        # safeguard
        if abs(denom) < 1e-12:
            return 0.0

        return ln_y / denom

    def _flow_regime(self, lam: float, NFr: float) -> str:
        """
        Determine horizontal Beggs-Brill flow regime.
        """
        L1 = self._L1(lam)
        L2 = self._L2(lam)
        L3 = self._L3(lam)
        L4 = self._L4(lam)

        if (lam < 0.01 and NFr < L1) or (lam >= 0.01 and NFr < L2):
            return "segregated"

        if lam >= 0.01 and L2 <= NFr <= L3:
            return "transition"

        if ((0.01 <= lam < 0.4) and (L3 < NFr <= L1)) or ((lam >= 0.4) and (L3 < NFr <= L4)):
            return "intermittent"

        return "distributed"

    @staticmethod
    def _EL0_regime(lam: float, NFr: float, a: float, b: float, c: float) -> float:
        EL0 = a * max(lam, 1e-12)**b / max(NFr, 1e-12)**c
        return min(max(EL0, lam), 1.0)

    def _EL0(self, lam: float, NFr: float, regime: str) -> float:
        if regime == "transition":
            raise ValueError("Transition EL0 should be handled by interpolation.")
        a, b, c = self._HOLdup_COEFF[regime]
        return self._EL0_regime(lam, NFr, a, b, c)

    def _inclination_multiplier(self, lam: float, NFr: float, NLv: float, regime: str) -> float:
        """
        psi = 1 + beta * (sin(1.8θ) - (1/3)sin^3(1.8θ))
        beta = (1-lam) * ln(d * lam^e * NLv^f * NFr^g)
        """
        if abs(self.theta_deg) < 1e-10:
            return 1.0

        if regime == "distributed":
            return 1.0

        coeffs = self._INCL_COEFF_UPHILL if self.theta > 0.0 else self._INCL_COEFF_DOWNHILL
        d, e, f, g = coeffs[regime]

        arg = (
            d
            * self._safe_pow(lam, e)
            * self._safe_pow(NLv, f)
            * self._safe_pow(NFr, g)
        )

        beta = (1.0 - lam) * math.log(max(arg, 1e-12))
        trig = math.sin(1.8 * self.theta) - (math.sin(1.8 * self.theta) ** 3) / 3.0
        psi = 1.0 + beta * trig

        return max(psi, 0.0)

    def _payne_correction(self, EL: float) -> float:
        """
        Payne et al. correction.
        """
        if self.theta > 0.0:
            EL *= 0.924
        elif self.theta < 0.0:
            EL *= 0.685
        return EL

    def _liquid_holdup(self, lam: float, NFr: float, NLv: float) -> tuple[float, str]:
        """
        Returns inclined liquid holdup EL and regime.
        """
        regime = self._flow_regime(lam, NFr)

        if regime == "transition":
            L2 = self._L2(lam)
            L3 = self._L3(lam)

            A = (L3 - NFr) / max(L3 - L2, 1e-12)
            B = 1.0 - A

            EL0_seg = self._EL0(lam, NFr, "segregated")
            EL0_int = self._EL0(lam, NFr, "intermittent")

            psi_seg = self._inclination_multiplier(lam, NFr, NLv, "segregated")
            psi_int = self._inclination_multiplier(lam, NFr, NLv, "intermittent")

            EL_seg = min(max(self._payne_correction(EL0_seg * psi_seg), lam), 1.0)
            EL_int = min(max(self._payne_correction(EL0_int * psi_int), lam), 1.0)

            EL = A * EL_seg + B * EL_int
            EL = min(max(EL, lam), 1.0)
            return EL, regime

        EL0 = self._EL0(lam, NFr, regime)
        psi = self._inclination_multiplier(lam, NFr, NLv, regime)
        EL = EL0 * psi
        EL = self._payne_correction(EL)
        EL = min(max(EL, lam), 1.0)
        return EL, regime

    # ------------------------------------------------------------------
    # Main calculation
    # ------------------------------------------------------------------

    def dpdl_psi_ft(self, q_total_stbd: float, p_psia: float, include_acceleration: bool = True) -> float:
        """
        Total pressure gradient [psi/ft].

        Parameters
        ----------
        q_total_stbd : total surface liquid rate [STB/D]
        p_psia       : flowing pressure [psia]
        include_acceleration : whether to include Beggs-Brill acceleration correction

        Returns
        -------
        dP/dL [psi/ft]
        """
        fluid = self.fluid

        # -------------------------
        # Liquid properties / rate
        # -------------------------
        q_l = fluid.liquid_rate_ft3s(q_total_stbd, p_psia)   # ft3/s
        rho_l = fluid.mixture_density_lbmft3(p_psia)         # lbm/ft3
        mu_l = cp_to_lbmfts(fluid.mixture_viscosity_cp(p_psia))

        # -------------------------
        # Gas properties / rate
        # -------------------------
        Rs_sc = fluid.solution_gor(p_psia)
        free_gor = max(fluid.gor_scf_stb - Rs_sc, 0.0)

        q_o_stbd = q_total_stbd * (1.0 - fluid.wc)
        Bg_ft3_scf = fluid.gas_fvf_ft3_scf(p_psia)

        # scf/day * ft3/scf / 86400 = ft3/s
        q_g = q_o_stbd * free_gor * Bg_ft3_scf / 86400.0

        rho_g = self._gas_density(p_psia)
        mu_g = cp_to_lbmfts(self._gas_viscosity_cp(p_psia))

        # -------------------------
        # Superficial velocities
        # -------------------------
        vsl, vsg, vm, lam_l = _superficial_velocities(q_l, q_g, self.A)

        # Beggs-Brill groups
        NFr = vm**2 / (G * self.d)

        # Liquid velocity number
        # This assumes sigma_l is in dynes/cm and field-units style use in your codebase.
        sigma_l = 30.0
        NLv = vsl * (rho_l / max(G * sigma_l, 1e-12))**0.25

        # -------------------------
        # Liquid holdup
        # -------------------------
        EL, regime = self._liquid_holdup(lam_l, NFr, NLv)

        # -------------------------
        # Mixture properties
        # -------------------------
        rho_m = rho_l * EL + rho_g * (1.0 - EL)                  # slip mixture density
        rho_ns = rho_l * lam_l + rho_g * (1.0 - lam_l)          # no-slip density

        # no-slip viscosity for Reynolds number
        mu_ns = math.exp(
            lam_l * math.log(max(mu_l, 1e-30))
            + (1.0 - lam_l) * math.log(max(mu_g, 1e-30))
        )

        # -------------------------
        # Friction factor
        # -------------------------
        Re = rho_ns * vm * self.d / max(mu_ns, 1e-30)
        eps_rel = self.eps / self.d
        f_ns = self._colebrook_white(Re, eps_rel)

        y = lam_l / max(EL, 1e-12)**2
        s = self._friction_ratio_s(y)
        f_tp = f_ns * math.exp(s)

        # -------------------------
        # Pressure gradient terms
        # -------------------------
        dpdl_fric = f_tp * rho_ns * vm**2 / (2.0 * GC * self.d)   # lbf/ft3
        dpdl_elev = rho_m * G * self.sin_t / GC                   # lbf/ft3

        if include_acceleration:
            Ek = rho_m * vm * vsg / (GC * max(p_psia * 144.0, 1e-12))
            Ek = min(max(Ek, 0.0), 0.95)
            total_psf_per_ft = (dpdl_fric + dpdl_elev) / (1.0 - Ek)
        else:
            total_psf_per_ft = dpdl_fric + dpdl_elev

        return total_psf_per_ft * PSF_TO_PSI

    # ------------------------------------------------------------------
    # Useful debug hook
    # ------------------------------------------------------------------

    def details(self, q_total_stbd: float, p_psia: float, include_acceleration: bool = True) -> dict:
        """
        Returns intermediate variables for debugging / validation.
        """
        fluid = self.fluid

        q_l = fluid.liquid_rate_ft3s(q_total_stbd, p_psia)
        rho_l = fluid.mixture_density_lbmft3(p_psia)
        mu_l = cp_to_lbmfts(fluid.mixture_viscosity_cp(p_psia))

        Rs_sc = fluid.solution_gor(p_psia)
        free_gor = max(fluid.gor_scf_stb - Rs_sc, 0.0)
        q_o_stbd = q_total_stbd * (1.0 - fluid.wc)
        Bg_ft3_scf = fluid.gas_fvf_ft3_scf(p_psia)
        q_g = q_o_stbd * free_gor * Bg_ft3_scf / 86400.0

        rho_g = self._gas_density(p_psia)
        mu_g = cp_to_lbmfts(self._gas_viscosity_cp(p_psia))

        vsl, vsg, vm, lam_l = _superficial_velocities(q_l, q_g, self.A)

        NFr = vm**2 / (G * self.d)
        sigma_l = 30.0
        NLv = vsl * (rho_l / max(G * sigma_l, 1e-12))**0.25

        EL, regime = self._liquid_holdup(lam_l, NFr, NLv)

        rho_m = rho_l * EL + rho_g * (1.0 - EL)
        rho_ns = rho_l * lam_l + rho_g * (1.0 - lam_l)

        mu_ns = math.exp(
            lam_l * math.log(max(mu_l, 1e-30))
            + (1.0 - lam_l) * math.log(max(mu_g, 1e-30))
        )

        Re = rho_ns * vm * self.d / max(mu_ns, 1e-30)
        eps_rel = self.eps / self.d
        f_ns = self._colebrook_white(Re, eps_rel)

        y = lam_l / max(EL, 1e-12)**2
        s = self._friction_ratio_s(y)
        f_tp = f_ns * math.exp(s)

        dpdl_fric = f_tp * rho_ns * vm**2 / (2.0 * GC * self.d)
        dpdl_elev = rho_m * G * self.sin_t / GC

        if include_acceleration:
            Ek = rho_m * vm * vsg / (GC * max(p_psia * 144.0, 1e-12))
            Ek = min(max(Ek, 0.0), 0.95)
            total = (dpdl_fric + dpdl_elev) / (1.0 - Ek)
        else:
            Ek = 0.0
            total = dpdl_fric + dpdl_elev

        return {
            "q_l_ft3s": q_l,
            "q_g_ft3s": q_g,
            "vsl_fts": vsl,
            "vsg_fts": vsg,
            "vm_fts": vm,
            "lambda_l": lam_l,
            "NFr": NFr,
            "NLv": NLv,
            "regime": regime,
            "EL": EL,
            "rho_l": rho_l,
            "rho_g": rho_g,
            "rho_m": rho_m,
            "rho_ns": rho_ns,
            "mu_l_lbmfts": mu_l,
            "mu_g_lbmfts": mu_g,
            "mu_ns_lbmfts": mu_ns,
            "Re": Re,
            "f_ns": f_ns,
            "y": y,
            "s": s,
            "f_tp": f_tp,
            "Ek": Ek,
            "dpdl_fric_psf_per_ft": dpdl_fric,
            "dpdl_elev_psf_per_ft": dpdl_elev,
            "dpdl_total_psi_per_ft": total * PSF_TO_PSI,
        }


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

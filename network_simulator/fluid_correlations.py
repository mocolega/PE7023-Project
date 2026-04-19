"""
fluid_correlations.py
=====================
Black-oil PVT anchored to measured values.

Known measured values
---------------------
  API        = 45       → sg_oil = 0.8017
  µ_o @ Pb   = 0.902 cp (measured at reservoir = Pb)
  γ_g        = 0.6636
  GOR        = 800 scf/STB
  Pb         = 3800 psia  (measured)
  Bo @ Pb    = 1.2 bbl/STB (measured)

Gas compressibility
-------------------
  Hall-Yarborough z-factor correlation replaces the fixed z=0.85.
  This matters because Pr = Pb, so any drawdown produces free gas
  and the gas volume (Bg) directly sets the in-situ gas velocity
  which drives the multiphase friction calculation.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BlackOilFluid:
    """
    Fluid model anchored to measured PVT values.

    Parameters
    ----------
    api           : API gravity
    mu_o_res_cp   : measured oil viscosity at p_res_psia  [cp]
    sg_gas        : gas specific gravity (air = 1)
    gor_scf_stb   : producing GOR  [scf/STB]
    pb_psia       : measured bubble-point pressure  [psia]
    bo_pb         : measured Bo at bubble point  [bbl/STB]
    wc            : water cut fraction (0–1)
    T_F           : constant system temperature  [°F]
    p_res_psia    : reservoir pressure (= Pb for this field)
    p_sep_psia    : separator pressure
    """
    api:          float = 45.0
    mu_o_res_cp:  float = 0.902
    sg_gas:       float = 0.6636
    gor_scf_stb:  float = 800.0
    pb_psia:      float = 3800.0
    bo_pb:        float = 1.2
    wc:           float = 0.05
    T_F:          float = 150.0
    p_res_psia:   float = 3800.0
    p_sep_psia:   float = 1202.54

    sg_oil:           float = field(init=False)
    rho_water_lbmft3: float = field(init=False)

    def __post_init__(self):
        self.sg_oil           = 141.5 / (self.api + 131.5)
        self.rho_water_lbmft3 = self._water_density_lbmft3()
        self._mu_anchor_p     = max(self.p_res_psia, self.pb_psia)

    # ================================================================
    # Gas compressibility – Hall-Yarborough (replaces fixed z=0.85)
    # ================================================================

    def z_factor(self, p_psia: float) -> float:
        """
        Hall-Yarborough (1974) z-factor.
        Valid for 1.05 < Tpr < 3.0, Ppr < 15.
        Falls back to ideal gas (z=1) at very low pressures.
        """
        if p_psia < 14.7:
            return 1.0

        T_R = self.T_F + 459.67
        sg  = self.sg_gas

        # Kay's mixing rule pseudo-criticals (Standing 1977)
        Tpc = 168.0 + 325.0 * sg - 12.5 * sg**2   # °R
        Ppc = 677.0 + 15.0  * sg - 37.5 * sg**2   # psia

        Tpr = T_R / Tpc
        Ppr = p_psia / Ppc

        if Tpr < 1.05:
            Tpr = 1.05   # avoid singularity

        t = 1.0 / Tpr
        A = -0.06125 * t * math.exp(-1.2 * (1.0 - t)**2)

        # Newton-Raphson on implicit Hall-Yarborough equation
        y = 0.0125 * Ppr * t * math.exp(-1.2 * (1.0 - t)**2)
        y = max(y, 1e-4)

        for _ in range(50):
            ey = math.exp(min((2.18 + 2.82*t) * math.log(max(y, 1e-12)), 500))
            F  = ( A * Ppr
                  + (y + y**2 + y**3 - y**4) / (1.0 - y)**3
                  - (14.76*t - 9.76*t**2 + 4.58*t**3) * y**2
                  + (90.7*t - 242.2*t**2 + 42.4*t**3) * ey )
            dF = ( (1.0 + 4.0*y + 4.0*y**2 - 4.0*y**3 + y**4) / (1.0-y)**4
                  - (29.52*t - 19.52*t**2 + 9.16*t**3) * y
                  + (90.7*t - 242.2*t**2 + 42.4*t**3)
                    * (2.18 + 2.82*t) * ey )
            if abs(dF) < 1e-20:
                break
            dy = -F / dF
            y += dy
            if abs(dy) < 1e-8:
                break

        z = -A * Ppr / max(y, 1e-12)
        return max(min(z, 2.0), 0.05)   # physical bounds

    # ================================================================
    # Bubble point – KNOWN
    # ================================================================

    def bubble_point_psia(self) -> float:
        return self.pb_psia

    # ================================================================
    # Solution GOR – Standing shape anchored to GOR at Pb
    # ================================================================

    def solution_gor(self, p_psia: float) -> float:
        if p_psia >= self.pb_psia:
            return self.gor_scf_stb
        T   = self.T_F
        sg  = self.sg_gas
        api = self.api
        x   = 0.0125 * api - 0.00091 * T

        def _rs_raw(p):
            base = max(p / 18.2 + 1.4, 0.0) / (10.0 ** x)
            return sg * max(base, 0.0) ** (1.0 / 0.83)

        rs_at_pb = _rs_raw(self.pb_psia)
        if rs_at_pb <= 0:
            return self.gor_scf_stb * max(p_psia / self.pb_psia, 0.0)
        return min(self.gor_scf_stb / rs_at_pb * _rs_raw(p_psia),
                   self.gor_scf_stb)

    # ================================================================
    # Oil FVF – anchored to measured Bo(Pb) = 1.2 bbl/STB
    # ================================================================

    def oil_fvf(self, p_psia: float) -> float:
        pb   = self.pb_psia
        Rs   = self.solution_gor(p_psia)
        Rs_b = self.gor_scf_stb
        sg   = self.sg_gas
        api  = self.api
        T    = self.T_F
        sg_100 = sg * (1.0 + 5.912e-5 * api * self.p_sep_psia
                       * math.log10(max(self.p_sep_psia, 1.0) / 114.7))
        C1, C2, C3 = 4.670e-4, 1.100e-5, 1.337e-9

        def _vb(r):
            return (1.0 + C1*r + C2*(T-60)*(api/sg_100)
                    + C3*r*(T-60)*(api/sg_100))

        if p_psia <= pb:
            ratio = self.bo_pb / max(_vb(Rs_b), 1e-6)
            return max(_vb(Rs) * ratio, 1.0)
        else:
            co = self._oil_compressibility(pb, Rs_b)
            return self.bo_pb * math.exp(-co * (p_psia - pb))

    def _oil_compressibility(self, p_psia: float, Rs: float) -> float:
        co = (-1433 + 5*Rs + 17.2*self.T_F
              - 1180*self.sg_gas + 12.61*self.api) / (1e5 * p_psia)
        return max(co, 1e-8)

    # ================================================================
    # Oil viscosity – anchored to measured µ_o = 0.902 cp at Pb
    # ================================================================

    def oil_viscosity_cp(self, p_psia: float) -> float:
        pb      = self.pb_psia
        T       = self.T_F
        api     = self.api
        mu_anch = self.mu_o_res_cp
        p_anch  = self._mu_anchor_p

        x_b   = 10.0 ** (3.0324 - 0.02023 * api)
        mu_od = max(10.0 ** (x_b * T**(-1.163)) - 1.0, 0.01)

        def _br(rs):
            a = 10.715 * (rs + 100.0)**(-0.515)
            b = 5.44   * (rs + 150.0)**(-0.338)
            return max(a * mu_od**b, 0.01)

        if p_anch >= pb:
            m_a   = 2.6 * p_anch**1.187 * math.exp(-11.513 - 8.98e-5 * p_anch)
            mu_pb = mu_anch / max((p_anch / pb)**m_a, 1e-10)
        else:
            mu_pb = mu_anch * _br(self.gor_scf_stb) / max(
                _br(self.solution_gor(p_anch)), 1e-10)
        mu_pb = max(mu_pb, 0.01)

        if p_psia <= pb:
            br_pb = _br(self.gor_scf_stb)
            br_p  = _br(self.solution_gor(p_psia))
            return max(mu_pb * br_p / max(br_pb, 1e-10), 0.01)
        else:
            m = 2.6 * p_psia**1.187 * math.exp(-11.513 - 8.98e-5 * p_psia)
            return max(mu_pb * (p_psia / pb)**m, 0.01)

    # ================================================================
    # Gas properties – using Hall-Yarborough z
    # ================================================================

    def gas_fvf_ft3_scf(self, p_psia: float) -> float:
        """Gas formation volume factor [ft³/scf] at p, T."""
        T_R = self.T_F + 459.67
        z   = self.z_factor(p_psia)
        # Bg = z * T_R / p  *  (14.7 / 520)   [ft³/scf]
        return z * T_R / max(p_psia, 0.1) * (14.7 / 520.0)

    def gas_density_lbmft3(self, p_psia: float) -> float:
        """In-situ gas density [lbm/ft³]."""
        T_R = self.T_F + 459.67
        Mg  = 28.97 * self.sg_gas
        z   = self.z_factor(p_psia)
        return p_psia * Mg / (10.73 * T_R * z)

    def gas_viscosity_cp(self, p_psia: float) -> float:
        """Lee-Kesler gas viscosity [cp]."""
        T_R   = self.T_F + 459.67
        Mg    = 28.97 * self.sg_gas
        rho_g = self.gas_density_lbmft3(p_psia)
        K     = (9.4 + 0.02*Mg) * T_R**1.5 / (209 + 19*Mg + T_R)
        X     = 3.5 + 986.0/T_R + 0.01*Mg
        Y     = 2.4 - 0.2*X
        return max(1e-4 * K * math.exp(X * max(rho_g/62.4, 0.0)**Y), 0.01)

    # ================================================================
    # Oil density
    # ================================================================

    def oil_density_lbmft3(self, p_psia: float) -> float:
        Rs      = self.solution_gor(p_psia)
        Bo      = self.oil_fvf(p_psia)
        rho_sto = self.sg_oil * 62.4
        return (rho_sto + 0.0136 * Rs * self.sg_gas) / Bo

    # ================================================================
    # Water PVT
    # ================================================================

    def _water_density_lbmft3(self) -> float:
        return 62.4 - 0.0027 * (self.T_F - 60.0)

    def water_viscosity_cp(self) -> float:
        return max(math.exp(-11.0 + 1838.0 / (self.T_F + 459.67)), 0.1)

    def water_fvf(self, p_psia: float) -> float:
        return 1.0

    # ================================================================
    # Liquid mixture (oil + water)
    # ================================================================

    def mixture_density_lbmft3(self, p_psia: float) -> float:
        wc    = self.wc
        Bo    = self.oil_fvf(p_psia)
        Bw    = self.water_fvf(p_psia)
        rho_o = self.oil_density_lbmft3(p_psia)
        rho_w = self.rho_water_lbmft3
        q_o   = (1.0 - wc) * Bo
        q_w   = wc * Bw
        tot   = q_o + q_w
        return (q_o * rho_o + q_w * rho_w) / tot

    def mixture_viscosity_cp(self, p_psia: float) -> float:
        wc    = self.wc
        Bo    = self.oil_fvf(p_psia)
        Bw    = self.water_fvf(p_psia)
        mu_o  = self.oil_viscosity_cp(p_psia)
        mu_w  = self.water_viscosity_cp()
        q_o   = (1.0 - wc) * Bo
        q_w   = wc * Bw
        tot   = q_o + q_w
        f_o   = q_o / tot
        f_w   = q_w / tot
        return math.exp(f_o * math.log(max(mu_o, 1e-6))
                        + f_w * math.log(max(mu_w, 1e-6)))

    def liquid_rate_ft3s(self, q_stbd_total: float, p_psia: float) -> float:
        """Surface liquid [STB/D] → in-situ volumetric [ft³/s]."""
        wc  = self.wc
        Bo  = self.oil_fvf(p_psia)
        Bw  = self.water_fvf(p_psia)
        q_insitu_bpd = q_stbd_total * ((1.0 - wc)*Bo + wc*Bw)
        return q_insitu_bpd * 5.614583 / 86400.0


# ================================================================
# Self-test – verify anchors and z-factor
# ================================================================
if __name__ == "__main__":
    f = BlackOilFluid()

    print("=" * 65)
    print("ANCHOR VERIFICATION")
    print("=" * 65)
    print(f"  Bo(Pb)     = {f.oil_fvf(f.pb_psia):.6f}  [should be 1.200000]")
    print(f"  Rs(Pb)     = {f.solution_gor(f.pb_psia):.2f}  [should be 800.00]")
    print(f"  µo(p_res)  = {f.oil_viscosity_cp(f.p_res_psia):.4f}  [should be ~0.9020]")

    print("\n" + "=" * 65)
    print("Z-FACTOR  (Hall-Yarborough vs old fixed z=0.85)")
    print("=" * 65)
    print(f"  {'p (psia)':>10}  {'z (H-Y)':>10}  {'z_old':>8}  {'Bg_HY (ft³/scf)':>17}  {'Bg_old':>10}")
    print(f"  {'─'*58}")
    for p in [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3800]:
        z_hy  = f.z_factor(p)
        z_old = 0.85
        T_R   = f.T_F + 459.67
        Bg_hy  = z_hy  * T_R / p * (14.7/520)
        Bg_old = z_old * T_R / p * (14.7/520)
        print(f"  {p:>10}  {z_hy:>10.4f}  {z_old:>8.2f}"
              f"  {Bg_hy:>17.5f}  {Bg_old:>10.5f}")

    print("\n" + "=" * 65)
    print("FULL PVT TABLE")
    print("=" * 65)
    print(f"  {'p':>6}  {'Rs':>7}  {'Bo':>8}  {'µo':>7}  "
          f"{'z':>7}  {'ρmix':>8}  {'µmix':>8}")
    print(f"  {'─'*58}")
    for p in [500, 1000, 1500, 2000, 2500, 3000, 3800]:
        print(f"  {p:>6}  {f.solution_gor(p):>7.1f}  {f.oil_fvf(p):>8.4f}"
              f"  {f.oil_viscosity_cp(p):>7.4f}  {f.z_factor(p):>7.4f}"
              f"  {f.mixture_density_lbmft3(p):>8.3f}"
              f"  {f.mixture_viscosity_cp(p):>8.4f}")

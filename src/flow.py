r"""
Reduced-order airflow and thermal model for the Excalibur pack.

The chain is:

  1. **System resistance.**  Zukauskas staggered tube-bank pressure drop,

     $$\Delta p_{\text{bank}} = N_L\,\chi\,f\,\frac{\rho V_{\max}^2}{2}$$

     plus duct and grille losses, giving $\Delta p_{\text{sys}}(Q)$.

  2. **Operating point.**  Solve $\Delta p_{\text{fan}}(Q) = \Delta p_{\text{sys}}(Q)$
     for the flow $Q$ the fans actually deliver against that resistance.  This is
     the step the design review left open: it *assumed* a face velocity
     (free delivery x 0.25 fudge x 0.85 duct efficiency = 1.3684 m/s) rather
     than intersecting the curves.

  3. **Heat transfer.**  Zukauskas correlation at the real flow,

     $$\mathrm{Nu} = C_1 C_2 \mathrm{Re}_{\max}^m \mathrm{Pr}^{0.36}
       \left(\frac{\mathrm{Pr}}{\mathrm{Pr}_s}\right)^{1/4},
       \qquad h = \frac{\mathrm{Nu}\,k}{D}$$

  4. **Row march.**  Air warms as it crosses the 17 rows,

     $$\dot m c_p \,\Delta T_{\text{air}} = \sum \dot Q_{\text{cell}},
       \qquad T_{\text{cell}} = T_{\text{air,local}} +
       \frac{\dot Q_{\text{cell}}}{h A_{\text{cell}}}$$

     with air properties re-evaluated at each row's local temperature.

Correlations follow Incropera & DeWitt, *Fundamentals of Heat and Mass
Transfer* 7e, Sec. 7.6 -- the same source the review's MATLAB study cites.

Geometry is MEASURED from the CAD (see `params.py`), which is the substantive
difference from the earlier `flow_model/`: that work assumed an equilateral
bank ($S_L = S_T$), and the CAD says $S_L/S_T = 1.23$.
"""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from params import (
    CFM_PER_M3MIN, KELVIN, MM_H2O_TO_PA,
    DELTA_FAN, SAN_ACE_FAN,
    Fan, OperatingConditions, PackElectrical, PackGeometry,
)


# =============================================================================
# Air properties
# =============================================================================

@dataclass(frozen=True)
class AirProperties:
    """Dry air at ~1 atm, evaluated at a given temperature [K].

    Ideal gas for density, Sutherland for viscosity, and polynomial fits for
    conductivity and specific heat over the 250-450 K range this model uses.
    """

    temperature: float

    @property
    def density(self) -> float:                 # kg/m^3
        return 101325.0 / (287.05 * self.temperature)

    @property
    def dynamic_viscosity(self) -> float:       # Pa.s  (Sutherland)
        mu0, T0, S = 1.716e-5, 273.15, 110.4
        T = self.temperature
        return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)

    @property
    def kinematic_viscosity(self) -> float:     # m^2/s
        return self.dynamic_viscosity / self.density

    @property
    def conductivity(self) -> float:            # W/m.K
        T = self.temperature
        return 1.5207e-11 * T**3 - 4.8574e-8 * T**2 + 1.0184e-4 * T - 3.9333e-4

    @property
    def specific_heat(self) -> float:           # J/kg.K
        T = self.temperature
        return 1002.5 + 2.75e-4 * (T - 200.0) ** 2

    @property
    def prandtl(self) -> float:
        return self.dynamic_viscosity * self.specific_heat / self.conductivity


# =============================================================================
# Zukauskas correlations (Incropera 7e, Sec. 7.6)
# =============================================================================

def row_correction(n_rows: int) -> float:
    """$C_2$, the correction for a bank shallower than 20 rows (Table 7.8, staggered)."""
    if n_rows >= 20:
        return 1.0
    rows = np.array([1, 2, 3, 4, 5, 7, 10, 13, 16, 20])
    c2 = np.array([0.64, 0.76, 0.84, 0.89, 0.92, 0.95, 0.97, 0.98, 0.99, 1.00])
    return float(np.interp(n_rows, rows, c2))


def zukauskas_nusselt(re_max: float, pr: float, pr_s: float,
                      n_rows: int, s_t: float, s_l: float) -> float:
    r"""Average Nusselt number for a staggered bank (Table 7.7).

    $$\mathrm{Nu} = C_1 C_2 \mathrm{Re}_{\max}^m \mathrm{Pr}^{0.36}
      (\mathrm{Pr}/\mathrm{Pr}_s)^{1/4}$$

    For $10^3 < \mathrm{Re} < 2\times10^5$ the staggered constant depends on the
    pitch ratio: $C_1 = 0.35 (S_T/S_L)^{1/5}$ when $S_T/S_L < 2$.  The review
    took $S_T/S_L = 1$; the CAD gives 0.813, which is what is used here.
    """
    ratio = s_t / s_l
    if re_max < 1e2:
        c1, m = 1.04, 0.40
    elif re_max < 1e3:
        c1, m = 0.71, 0.50
    elif re_max < 2e5:
        c1, m = (0.35 * ratio**0.2, 0.60) if ratio < 2.0 else (0.40, 0.60)
    else:
        c1, m = 0.031 * ratio**0.2, 0.80

    return (c1 * row_correction(n_rows) * re_max**m
            * pr**0.36 * (pr / pr_s) ** 0.25)


def zukauskas_friction(re_max: float, p_t: float, p_l: float) -> tuple[float, float]:
    r"""Friction factor $f$ and correction $\chi$ for a staggered bank (Fig. 7.14).

    $P_T = S_T/D$ and $P_L = S_L/D$.  This pack sits at $P_T = 1.114$, which is
    tighter than any published curve -- Fig. 7.14's tightest staggered line is
    $P_T = 1.25$ -- so $f$ is obtained by fitting that line and scaling on the
    free-area ratio, which is what actually sets the dynamic head through the
    minimum section.  **This extrapolation is the single largest uncertainty in
    the pressure drop.**  It is conservative: it over-predicts $\Delta p$ and so
    under-predicts the flow.

    $\chi$ corrects for a non-equilateral bank.  It is 1 when $S_T = S_L$; for
    $P_T/P_L < 1$ (this pack: 0.813) it rises above 1.  The earlier model set
    $\chi = 1$ on the strength of the equilateral assumption, which the CAD
    does not support.
    """
    re = max(re_max, 10.0)

    # Fit to the P_T = 1.25 staggered line of Fig. 7.14.
    f_125 = 0.45 + 42.0 * re**-0.72

    free_125 = (1.25 - 1.0) / 1.25
    free_act = max(p_t - 1.0, 1e-3) / p_t
    f = f_125 * (free_125 / free_act) ** 0.5

    # chi from the Fig. 7.14 inset, parameterised on (P_T - 1)/(P_L - 1).
    # Equilateral -> 1; a bank stretched along the flow -> mildly above 1.
    ratio = (p_t - 1.0) / max(p_l - 1.0, 1e-6)
    chi = float(np.clip(1.0 + 0.30 * (1.0 - ratio), 0.8, 1.6))

    return f, chi


# =============================================================================
# Result record
# =============================================================================

@dataclass
class FlowResult:
    """Everything the solver produces at one operating point."""

    volumetric_flow: float          # m^3/s, total
    inlet_velocity: float           # m/s, at the bank face
    v_max: float                    # m/s, in the minimum section
    mass_flow: float                # kg/s
    reynolds_max: float
    nusselt: float
    h_conv: float                   # W/m^2.K
    friction_factor: float
    chi: float
    pressure_drop_bank: float       # Pa
    pressure_drop_duct: float       # Pa
    pressure_drop_total: float      # Pa
    fan_pressure: float             # Pa
    inlet_temperature: float        # K
    outlet_temperature: float       # K
    cell_surface_temperature: float # K, the hottest row
    air_temperature_rise: float     # K
    cell_temperature_rise: float    # K
    row_air_temperatures: np.ndarray = field(default_factory=lambda: np.array([]))
    row_cell_temperatures: np.ndarray = field(default_factory=lambda: np.array([]))
    row_h: np.ndarray = field(default_factory=lambda: np.array([]))
    row_reynolds: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_balance_error: float = 0.0   # relative, should be ~0

    @property
    def flow_cfm(self) -> float:
        return self.volumetric_flow * 60.0 * CFM_PER_M3MIN

    @property
    def flow_m3min(self) -> float:
        return self.volumetric_flow * 60.0


# =============================================================================
# The model
# =============================================================================

class PackFlowModel:
    def __init__(self, geometry: PackGeometry | None = None,
                 fan: Fan = DELTA_FAN,
                 electrical: PackElectrical | None = None,
                 conditions: OperatingConditions | None = None,
                 heat_load: float | None = None) -> None:
        self.geo = geometry or PackGeometry.from_layout()
        self.fan = fan
        self.elec = electrical or PackElectrical()
        self.cond = conditions or OperatingConditions()
        self._heat_override = heat_load

    @property
    def heat_load(self) -> float:
        """Total pack dissipation [W]."""
        return (self._heat_override if self._heat_override is not None
                else self.elec.heat_load)

    # -- pressure side ------------------------------------------------------

    def system_pressure_drop(self, volumetric_flow: float,
                             air: AirProperties) -> tuple[float, float]:
        """(bank, duct+grille) pressure drop [Pa] for a total flow [m^3/s]."""
        g = self.geo
        v_in = volumetric_flow / g.duct_area
        v_max = g.v_max(v_in)
        re_max = v_max * g.cell_diameter / air.kinematic_viscosity

        f, chi = zukauskas_friction(re_max, g.pitch_ratio_T, g.pitch_ratio_L)
        dp_bank = g.n_rows * chi * f * (air.density * v_max**2 / 2.0)

        # Duct and grille losses are referenced to the fan face velocity, the
        # fastest air outside the bank.
        v_fan = volumetric_flow / (self.fan.active_area * self.fan.count)
        k = self.cond.duct_loss_coefficient + self.cond.grille_loss_coefficient
        dp_duct = k * air.density * v_fan**2 / 2.0

        return dp_bank, dp_duct

    def find_operating_point(self, iterations: int = 4) -> float:
        r"""Solve $\Delta p_{\text{fan}}(Q) = \Delta p_{\text{sys}}(Q)$ -> $Q$ [m^3/s].

        The system curve depends on air density, which depends on how much the
        air has warmed, which depends on the flow -- so the intersection and the
        thermal march are coupled.  The loop below iterates the two to
        consistency: solve for Q at an assumed mean air temperature, march the
        rows to find the real one, re-solve.  It converges in two or three
        passes, and without it the reported fan and system pressures disagree
        by ~0.5% because they are evaluated at different air states.
        """
        T_mean = self.cond.inlet_temperature + 5.0
        q = None

        for _ in range(iterations):
            air = AirProperties(T_mean)

            def residual(qq: float) -> float:
                dp_bank, dp_duct = self.system_pressure_drop(qq, air)
                return self.fan.pressure(qq) - (dp_bank + dp_duct)

            lo, hi = 1e-6, self.fan.free_delivery * 1.5
            if residual(lo) < 0:        # system already harder than shut-off
                return lo
            q_new = brentq(residual, lo, hi, xtol=1e-10)

            if q is not None and abs(q_new - q) < 1e-9:
                q = q_new
                break
            q = q_new

            # march the rows at this flow to get the real mean air temperature
            mdot = q * AirProperties(self.cond.inlet_temperature).density
            cp = AirProperties(T_mean).specific_heat
            dT = self.heat_load / (mdot * cp)
            T_mean = self.cond.inlet_temperature + 0.5 * dT

        return q

    # -- thermal side -------------------------------------------------------

    def solve(self, volumetric_flow: float | None = None) -> FlowResult:
        """Run the full chain, marching the air through the rows."""
        if volumetric_flow is None:
            volumetric_flow = self.find_operating_point()

        g, cond = self.geo, self.cond
        T_in = cond.inlet_temperature
        Q_total = self.heat_load

        v_in = volumetric_flow / g.duct_area
        v_max = g.v_max(v_in)
        air_in = AirProperties(T_in)
        mass_flow = volumetric_flow * air_in.density

        # Heat is split evenly over the rows.  (In reality the cells nearest
        # the busbars run hotter; the review makes the same simplification.)
        q_per_row = Q_total / g.n_rows
        # lateral area of one row of cells: N_T tubes of the full stacked length
        area_row = math.pi * g.cell_diameter * g.tube_length * g.n_parallel

        T_air = T_in
        rows_air, rows_cell, rows_h, rows_re = [], [], [], []
        f = chi = 0.0

        for _ in range(g.n_rows):
            air = AirProperties(T_air)
            re_max = v_max * g.cell_diameter / air.kinematic_viscosity

            # Pr_s is evaluated at the wall, which depends on h, which depends
            # on Pr_s.  Air's Pr varies weakly, so three passes converge.
            T_s = T_air + 10.0
            nu = h = 0.0
            for _ in range(3):
                pr_s = AirProperties(T_s).prandtl
                nu = zukauskas_nusselt(re_max, air.prandtl, pr_s, g.n_rows,
                                       g.transverse_pitch, g.longitudinal_pitch)
                h = nu * air.conductivity / g.cell_diameter
                T_s = T_air + q_per_row / (h * area_row)

            rows_cell.append(T_s)
            rows_h.append(h)
            rows_re.append(re_max)

            # the air leaves this row warmer
            T_air += q_per_row / (mass_flow * air.specific_heat)
            rows_air.append(T_air)

        T_out = T_air
        T_cell_max = max(rows_cell)

        # pressure drop at the mean air state
        air_mean = AirProperties(0.5 * (T_in + T_out))
        dp_bank, dp_duct = self.system_pressure_drop(volumetric_flow, air_mean)
        f, chi = zukauskas_friction(
            v_max * g.cell_diameter / air_mean.kinematic_viscosity,
            g.pitch_ratio_T, g.pitch_ratio_L)

        # -- sanity check: does the enthalpy rise account for the heat? -------
        cp_mean = AirProperties(0.5 * (T_in + T_out)).specific_heat
        q_absorbed = mass_flow * cp_mean * (T_out - T_in)
        err = abs(q_absorbed - Q_total) / Q_total if Q_total else 0.0

        return FlowResult(
            volumetric_flow=volumetric_flow,
            inlet_velocity=v_in,
            v_max=v_max,
            mass_flow=mass_flow,
            reynolds_max=float(np.mean(rows_re)),
            nusselt=nu,
            h_conv=float(np.mean(rows_h)),
            friction_factor=f,
            chi=chi,
            pressure_drop_bank=dp_bank,
            pressure_drop_duct=dp_duct,
            pressure_drop_total=dp_bank + dp_duct,
            fan_pressure=self.fan.pressure(volumetric_flow),
            inlet_temperature=T_in,
            outlet_temperature=T_out,
            cell_surface_temperature=T_cell_max,
            air_temperature_rise=T_out - T_in,
            cell_temperature_rise=T_cell_max - T_in,
            row_air_temperatures=np.array(rows_air),
            row_cell_temperatures=np.array(rows_cell),
            row_h=np.array(rows_h),
            row_reynolds=np.array(rows_re),
            energy_balance_error=err,
        )

    # -- curves -------------------------------------------------------------

    def fan_curve(self, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Fan static pressure vs total flow.  Returns (m^3/s, Pa)."""
        q = np.linspace(0.0, self.fan.free_delivery, n)
        return q, np.array([self.fan.pressure(qi) for qi in q])

    def system_curve(self, n: int = 200,
                     q_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """System resistance vs flow.  Returns (m^3/s, Pa)."""
        air = AirProperties(self.cond.inlet_temperature + 5.0)
        hi = q_max if q_max is not None else self.fan.free_delivery * 1.25
        q = np.linspace(1e-6, hi, n)
        dp = np.array([sum(self.system_pressure_drop(qi, air)) for qi in q])
        return q, dp

    # -- per-cell field -----------------------------------------------------

    def cell_temperatures(self, result: FlowResult,
                          order: list[tuple[int, int, int]] | None = None
                          ) -> np.ndarray:
        """Temperature for every cell, in `out/cell_positions.csv` order.

        The 1-D model resolves temperature by ROW only, so every cell in a row
        shares its row's value.  Returning a full 408-long array anyway is what
        lets the render colour cells directly, and leaves room for a 2-D model
        later without changing the interface.
        """
        if order is None:
            import csv as _csv
            path = pathlib.Path(__file__).resolve().parent.parent / "out" / "cell_positions.csv"
            with path.open() as fh:
                order = [(int(r["row"]), int(r["col"]), int(r["bank"]))
                         for r in _csv.DictReader(fh)]
        rt = result.row_cell_temperatures
        return np.array([rt[min(r, len(rt) - 1)] - KELVIN for r, _, _ in order])


# =============================================================================
# Validation against the review's own MATLAB study
# =============================================================================

def validate_against_review(model: PackFlowModel | None = None) -> dict:
    """Reproduce the review's fixed-velocity cases and compare.

    The review (slides 18 and 105) reports cell-surface temperature RISE at
    several imposed face velocities.  Running this model at the same imposed
    velocities -- not at its own operating point -- is an independent check
    that the correlations are wired up correctly, since the two derivations
    share only the textbook.

    Note slide 18 uses 318.84 W and slide 105 uses 295.69 W, so each case is
    compared against the heat load its own slide states.
    """
    cases = [
        # (velocity m/s, review rise C, heat W, slide)
        (1.86, 4.696, 318.84, "p18 staggered"),
        (1.00, 8.597, 318.84, "p18 staggered"),
        (0.50, 17.080, 318.84, "p18 staggered"),
        (0.34, 25.088, 318.84, "p18 staggered"),
        (1.3684, 5.85, 295.69, "p105 staggered"),
    ]
    out = []
    for v, ref_rise, q, src in cases:
        m = PackFlowModel(heat_load=q)
        r = m.solve(v * m.geo.duct_area)
        out.append({
            "velocity_ms": v,
            "source": src,
            "heat_w": q,
            "review_rise_c": ref_rise,
            "model_rise_c": r.cell_temperature_rise,
            "delta_c": r.cell_temperature_rise - ref_rise,
            "ratio": r.cell_temperature_rise / ref_rise,
        })
    return {"cases": out}


def summarize(result: FlowResult, model: PackFlowModel) -> str:
    g, K = model.geo, KELVIN
    limit = model.cond.cell_temperature_limit
    target = model.cond.cell_temperature_target
    lines = [
        f"Fan set ................ {model.fan.count} x {model.fan.name} "
        f"({model.fan.voltage:.0f} V, {model.fan.total_power:.0f} W total)",
        f"Cells .................. {g.n_cells} "
        f"({g.n_rows} rows x {g.n_parallel}P x {g.n_banks} banks, staggered)",
        f"Bank face .............. {g.duct_width*1000:.1f} x {g.duct_height*1000:.1f} mm"
        f"  = {g.duct_area*1e4:.1f} cm^2",
        f"S_T / S_L / D .......... {g.transverse_pitch*1000:.2f} / "
        f"{g.longitudinal_pitch*1000:.2f} / {g.cell_diameter*1000:.2f} mm",
        f"A_min / A_frontal ...... {g.min_free_area_ratio:.4f}",
        "",
        f"OPERATING POINT",
        f"  flow ................. {result.flow_m3min:.3f} m^3/min  "
        f"({result.flow_cfm:.1f} CFM)",
        f"  face velocity ........ {result.inlet_velocity:.3f} m/s",
        f"  V_max in bank ........ {result.v_max:.3f} m/s",
        f"  Re_max ............... {result.reynolds_max:,.0f}",
        f"  Nu / h ............... {result.nusselt:.1f} / {result.h_conv:.1f} W/m^2.K",
        f"  f / chi .............. {result.friction_factor:.3f} / {result.chi:.3f}",
        "",
        f"PRESSURE",
        f"  bank ................. {result.pressure_drop_bank:8.1f} Pa"
        f"  ({result.pressure_drop_bank/MM_H2O_TO_PA:6.1f} mmH2O)",
        f"  duct + grille ........ {result.pressure_drop_duct:8.1f} Pa"
        f"  ({result.pressure_drop_duct/MM_H2O_TO_PA:6.1f} mmH2O)",
        f"  total ................ {result.pressure_drop_total:8.1f} Pa"
        f"  ({result.pressure_drop_total/MM_H2O_TO_PA:6.1f} mmH2O)",
        f"  fan delivers ......... {result.fan_pressure:8.1f} Pa"
        f"   [{100*result.pressure_drop_bank/result.pressure_drop_total:.0f}% "
        f"of the drop is the bank]",
        "",
        f"THERMAL  (heat load {model.heat_load:.2f} W)",
        f"  inlet air ............ {result.inlet_temperature-K:5.1f} C",
        f"  outlet air ........... {result.outlet_temperature-K:5.1f} C"
        f"   (rise {result.air_temperature_rise:.2f} C)",
        f"  hottest cell ......... {result.cell_surface_temperature-K:5.1f} C"
        f"   (rise {result.cell_temperature_rise:.2f} C)",
        f"  design goal 45 C ..... "
        f"{'MET' if result.cell_surface_temperature <= target else 'MISSED'}"
        f"   (margin {(target-result.cell_surface_temperature):+.1f} C)",
        f"  limit 60 C ........... "
        f"{'PASS' if result.cell_surface_temperature <= limit else 'FAIL'}"
        f"   (margin {(limit-result.cell_surface_temperature):+.1f} C)",
        "",
        f"CHECKS",
        f"  energy balance ....... {result.energy_balance_error*100:.3f}% error",
        f"  Re range ............. {result.row_reynolds.min():,.0f} - "
        f"{result.row_reynolds.max():,.0f}"
        f"   [Zukauskas valid 1e3-2e5: "
        f"{'OK' if 1e3 <= result.row_reynolds.min() and result.row_reynolds.max() <= 2e5 else 'OUT OF RANGE'}]",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    model = PackFlowModel()
    res = model.solve()
    print(summarize(res, model))
    print()
    print("VALIDATION against the review's MATLAB study")
    print(f"{'v [m/s]':>9} {'source':>16} {'review':>9} {'model':>9} {'delta':>8} {'ratio':>7}")
    for c in validate_against_review()["cases"]:
        print(f"{c['velocity_ms']:9.4f} {c['source']:>16} "
              f"{c['review_rise_c']:8.2f}C {c['model_rise_c']:8.2f}C "
              f"{c['delta_c']:+7.2f}C {c['ratio']:7.3f}")



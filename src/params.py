"""
Every physical parameter, in one place, with units and a source.

Nothing here is invented.  Each value is either

  * MEASURED  -- taken from `BoxAssembly.step` by `src/geometry.py` (Task 1), or
  * REVIEW    -- stated in `Excalibur Battery Design Review.pdf`, slide noted, or
  * DATASHEET -- from the component datasheet reproduced in the review, or
  * ASSUMED   -- an engineering assumption that the review does not pin down.
                 These are the ones to challenge; each says why it was chosen.

Slide numbers are PDF page numbers of `Excalibur Battery Design Review.pdf`.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
LAYOUT_JSON = ROOT / "out" / "pack_layout.json"

# unit conversions
MM_H2O_TO_PA = 9.80665
IN_H2O_TO_PA = 249.0889
CFM_PER_M3MIN = 35.3147
KELVIN = 273.15


# =============================================================================
# Geometry -- MEASURED from the STEP (Task 1)
# =============================================================================

@dataclass(frozen=True)
class PackGeometry:
    """The cell bank, as measured from `BoxAssembly.step`.

    Defaults are the measured values; `from_layout()` re-reads them from
    `out/pack_layout.json` so the model always tracks the CAD.

    Flow direction (REVIEW, slide 100): the review treats the bank as
    `N = 204` tubes with `N_T = 12` per row and a tube length of 130 mm, i.e.
    the two 65 mm banks stacked end to end are ONE tube.  204 / 12 = 17 rows.
    So the air crosses the 17-row stack, 12 cells wide, and each tube is the
    full 130 mm.  That is the z axis of the CAD.
    """

    cell_diameter: float = 0.0184          # m   MEASURED (cylindrical face, not bbox)
    cell_height: float = 0.065             # m   MEASURED
    n_parallel: int = 12                   # N_T, cells across the flow front  MEASURED
    n_rows: int = 17                       # N_L, rows along the flow          MEASURED
    n_banks: int = 2                       # stacked along the cell axis       MEASURED

    transverse_pitch: float = 0.0205       # S_T [m]  MEASURED (20.50 mm)
    longitudinal_pitch: float = 0.0252     # S_L [m]  MEASURED (25.20 mm)

    # The duct/plenum feeding the bank.  ASSUMED to be the bank's own frontal
    # area: the fan/duct unit was suppressed before the STEP export, so there is
    # no ducting in the CAD to measure.  The review's own sizing script used
    # 0.24725 x 0.121 m = 0.0299 m^2, which is the same construction (12 cells
    # of pitch by two banks), so this reproduces their number from measured
    # pitches instead of back-solved ones.
    duct_height_override: float | None = None

    @classmethod
    def from_layout(cls, path: pathlib.Path = LAYOUT_JSON) -> "PackGeometry":
        """Build from the CAD-derived `out/pack_layout.json`."""
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        return cls(
            cell_diameter=d["cell_diameter_mm"] / 1000.0,
            cell_height=d["cell_length_mm"] / 1000.0,
            n_parallel=int(d["n_cols"]),
            n_rows=int(d["n_rows"]),
            n_banks=int(d["n_banks"]),
            transverse_pitch=d["transverse_pitch_S_T_mm"] / 1000.0,
            longitudinal_pitch=d["longitudinal_pitch_S_L_mm"] / 1000.0,
        )

    # -- derived ------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return self.n_parallel * self.n_rows * self.n_banks

    @property
    def tube_length(self) -> float:
        """Effective cylinder length [m]: the two banks stacked (REVIEW p100)."""
        return self.cell_height * self.n_banks

    @property
    def diagonal_pitch(self) -> float:
        r"""$S_D = \sqrt{S_L^2 + (S_T/2)^2}$ for a staggered bank."""
        return math.hypot(self.longitudinal_pitch, self.transverse_pitch / 2.0)

    @property
    def duct_width(self) -> float:
        """Flow-front width [m]: 12 cells on the transverse pitch."""
        return self.n_parallel * self.transverse_pitch

    @property
    def duct_height(self) -> float:
        """Flow-front height [m]: the stacked tube length."""
        if self.duct_height_override is not None:
            return self.duct_height_override
        return self.tube_length

    @property
    def duct_area(self) -> float:
        """Frontal (face) area of the bank [m^2]."""
        return self.duct_width * self.duct_height

    @property
    def pitch_ratio_T(self) -> float:
        return self.transverse_pitch / self.cell_diameter

    @property
    def pitch_ratio_L(self) -> float:
        return self.longitudinal_pitch / self.cell_diameter

    @property
    def min_free_area_ratio(self) -> float:
        r"""$A_{\min}/A_{\text{frontal}}$ -- the contraction that sets $V_{\max}$.

        For a staggered bank the minimum flow area is at the transverse plane
        ($S_T - D$ per pitch) unless the diagonal plane is tighter, in which
        case it is $2(S_D - D)$ per pitch.

        With the MEASURED pitches the transverse plane wins by a wide margin
        (2.10 mm against 17.61 mm), so the bank is unambiguously
        transverse-limited -- which the review's equilateral assumption could
        not have established.
        """
        S_T, S_D, D = self.transverse_pitch, self.diagonal_pitch, self.cell_diameter
        return min(S_T - D, 2.0 * (S_D - D)) / S_T

    def v_max(self, v_inlet: float) -> float:
        """Maximum velocity in the bank [m/s], by continuity through A_min."""
        return v_inlet / self.min_free_area_ratio

    @property
    def total_surface_area(self) -> float:
        """Convective area of all cells [m^2] (lateral cylinder area)."""
        return (math.pi * self.cell_diameter * self.tube_length
                * self.n_parallel * self.n_rows)

    def describe(self) -> str:
        return (
            f"{self.n_cells} cells = {self.n_rows} rows x {self.n_parallel} across "
            f"x {self.n_banks} banks (staggered)\n"
            f"  D = {self.cell_diameter*1000:.2f} mm, tube length "
            f"{self.tube_length*1000:.0f} mm\n"
            f"  S_T = {self.transverse_pitch*1000:.2f} mm (S_T/D = {self.pitch_ratio_T:.3f}), "
            f"S_L = {self.longitudinal_pitch*1000:.2f} mm (S_L/D = {self.pitch_ratio_L:.3f})\n"
            f"  S_D = {self.diagonal_pitch*1000:.2f} mm, "
            f"min gap = {(self.transverse_pitch-self.cell_diameter)*1000:.2f} mm\n"
            f"  face = {self.duct_width*1000:.1f} x {self.duct_height*1000:.1f} mm "
            f"= {self.duct_area*1e4:.1f} cm^2, A_min/A = {self.min_free_area_ratio:.4f}"
        )


# =============================================================================
# Electrical / heat load -- REVIEW
# =============================================================================

@dataclass(frozen=True)
class PackElectrical:
    """34S12P pack.  REVIEW slides 17 and 99.

    The review gives two worst cases, from two different power budgets:

      slide  99: 5200 W draw -> 61.18 A -> 295.69 W   (LV bus 100 W)
      slide  17: 5400 W draw -> 63.53 A -> 318.84 W   (LV bus 300 W, + horn)

    The later slide (99) is the one its own flow analysis uses, so 295.69 W is
    the default; 318.84 W is kept as the more conservative variant.
    """

    n_series: int = 34                     # REVIEW p99
    n_parallel: int = 12                   # REVIEW p99
    cell_resistance: float = 0.028         # ohm/cell, REVIEW p99 (LG MJ1)
    pack_voltage: float = 85.0             # V nominal, REVIEW p99
    power_draw: float = 5200.0             # W, REVIEW p99

    @property
    def pack_resistance(self) -> float:
        r"""$R = R_{\text{cell}} \cdot N_s / N_p$ [ohm]."""
        return self.cell_resistance * self.n_series / self.n_parallel

    @property
    def pack_current(self) -> float:
        """Pack current [A] at the design draw."""
        return self.power_draw / self.pack_voltage

    @property
    def cell_current(self) -> float:
        r"""Per-cell current [A]: $I_{\text{cell}} = I / N_p$."""
        return self.pack_current / self.n_parallel

    @property
    def heat_load(self) -> float:
        r"""Total pack $I^2R$ loss [W]."""
        return self.pack_current**2 * self.pack_resistance

    @property
    def heat_per_cell(self) -> float:
        r"""Per-cell dissipation [W]: $I_{\text{cell}}^2 R_{\text{cell}}$."""
        return self.cell_current**2 * self.cell_resistance


# The variant from slide 17 (5400 W / 318.84 W).
PACK_5400W = PackElectrical(power_draw=5400.0)


# =============================================================================
# Operating conditions -- REVIEW slides 15 / 98, plus ASSUMED loss coefficients
# =============================================================================

@dataclass(frozen=True)
class OperatingConditions:
    inlet_temperature: float = 40.0 + KELVIN    # K   REVIEW p15/p98 worst case
    cell_temperature_target: float = 45.0 + KELVIN   # K  REVIEW p15 design goal
    cell_temperature_limit: float = 60.0 + KELVIN    # K  REVIEW p100 (T_s limit)

    # ASSUMED.  The review does not give duct or grille loss coefficients.
    # These are conventional K-factors for a short ducted path plus a finger
    # guard.  They are small next to the bank (see the results), so the answer
    # is not sensitive to them -- but they are the least defensible numbers
    # here and the first thing to replace with a measurement.
    duct_loss_coefficient: float = 3.5
    grille_loss_coefficient: float = 1.4


# =============================================================================
# Fans -- DATASHEET, digitized from the P-Q curve on REVIEW slide 104
# =============================================================================

# Delta GFB0812ES-E, 12 V curve.  Read off the datasheet plot reproduced on
# slide 104: shut-off 150 mmH2O, free delivery ~3.9 m^3/min, with the
# characteristic flat shoulder to ~2 m^3/min then a steep fall.
DELTA_GFB0812ES_E_CURVE = np.array([
    # flow [m^3/min], static pressure [mmH2O]
    [0.00, 150.0],
    [0.50, 149.5],
    [1.00, 148.0],
    [1.30, 145.0],
    [1.60, 140.0],
    [2.00, 132.0],
    [2.40, 121.0],
    [2.80, 108.0],
    [3.00,  99.0],
    [3.20,  88.0],
    [3.40,  74.0],
    [3.60,  57.0],
    [3.75,  40.0],
    [3.85,  22.0],
    [3.92,   0.0],
])

# San Ace 80 (9GA0812P1G001-class), the fan the written spec calls for.
# REVIEW slide 107 gives only the two endpoints: 159 CFM and 4.6 inH2O.
# The curve SHAPE between them is ASSUMED (a typical high-static-pressure
# 80 mm profile), so this fan is for comparison only, not for sizing.
SAN_ACE_80_CURVE = np.array([
    [0.00, 116.8],      # 4.6 inH2O
    [0.60, 115.0],
    [1.20, 111.0],
    [1.80, 104.0],
    [2.40,  94.0],
    [3.00,  81.0],
    [3.50,  66.0],
    [3.90,  50.0],
    [4.20,  32.0],
    [4.40,  14.0],
    [4.50,   0.0],      # 159 CFM
])


@dataclass
class Fan:
    """A fan, or a set of identical fans in parallel."""

    name: str
    curve: np.ndarray                  # (flow m^3/min per fan, static pressure mmH2O)
    count: int = 1
    rated_power: float = 0.0           # W, each
    voltage: float = 12.0
    frame_size: float = 0.080          # m   REVIEW p107: 80 mm, 3 across the pack
    hub_diameter: float = 0.032        # m   ASSUMED: typical hub fraction
    source: str = ""

    def pressure(self, volumetric_flow: float) -> float:
        """Static pressure rise [Pa] for a TOTAL flow [m^3/s] through `count` fans.

        Fans in parallel share the pressure rise and split the flow, so the
        single-fan curve is evaluated at `flow / count`.
        """
        per_fan = (volumetric_flow / self.count) * 60.0     # m^3/min
        q, dp = self.curve[:, 0], self.curve[:, 1]
        if per_fan <= q[0]:
            return float(dp[0]) * MM_H2O_TO_PA
        if per_fan >= q[-1]:
            # Past free delivery the fan cannot sustain flow.  Extrapolate
            # steeply negative so a root finder stays bracketed.
            slope = (dp[-1] - dp[-2]) / (q[-1] - q[-2])
            return float(dp[-1] + slope * (per_fan - q[-1])) * MM_H2O_TO_PA
        return float(np.interp(per_fan, q, dp)) * MM_H2O_TO_PA

    @property
    def free_delivery(self) -> float:
        """Total flow at zero static pressure [m^3/s]."""
        return self.curve[-1, 0] / 60.0 * self.count

    @property
    def shutoff_pressure(self) -> float:
        """Static pressure at zero flow [Pa]."""
        return float(self.curve[0, 1]) * MM_H2O_TO_PA

    @property
    def active_area(self) -> float:
        """Annular swept area per fan [m^2] (frame minus hub)."""
        return math.pi * ((self.frame_size / 2) ** 2 - (self.hub_diameter / 2) ** 2)

    @property
    def total_power(self) -> float:
        return self.rated_power * self.count


DELTA_FAN = Fan(
    name="Delta GFB0812ES-E",
    curve=DELTA_GFB0812ES_E_CURVE,
    count=3,                      # REVIEW p107: "fits 3 across pack"
    rated_power=25.2,             # DATASHEET
    voltage=12.0,
    source="P-Q curve digitized from REVIEW slide 104 (12 V); "
           "this is the fan actually in the CAD",
)

SAN_ACE_FAN = Fan(
    name="San Ace 80",
    curve=SAN_ACE_80_CURVE,
    count=3,
    rated_power=64.0,             # REVIEW p107
    voltage=24.0,
    source="endpoints from REVIEW slide 107 (159 CFM, 4.6 inH2O); "
           "curve shape ASSUMED -- comparison only",
)


# =============================================================================
# Provenance table, printed by the report so nothing is silently assumed
# =============================================================================

PROVENANCE = [
    ("cell diameter D", "18.40 mm", "MEASURED", "cylindrical face in the STEP"),
    ("cell length", "65.00 mm", "MEASURED", "STEP bounding box"),
    ("rows x cols x banks", "17 x 12 x 2", "MEASURED", "STEP cell lattice"),
    ("S_T", "20.50 mm", "MEASURED", "STEP, within-row pitch"),
    ("S_L", "25.20 mm", "MEASURED", "STEP, row-to-row pitch"),
    ("stagger", "S_T/2 = 10.25 mm", "MEASURED", "STEP; REVIEW p14 confirms the intent"),
    ("flow direction", "across the 17 rows", "REVIEW", "p100: N=204, N_T=12, L=130 mm"),
    ("pack config", "34S12P", "REVIEW", "p99"),
    ("cell resistance", "28 mohm", "REVIEW", "p99 (LG MJ1)"),
    ("heat load", "295.69 W", "REVIEW", "p99, at 5200 W draw / 61.18 A"),
    ("heat load (alt)", "318.84 W", "REVIEW", "p17, at 5400 W draw / 63.53 A"),
    ("inlet air", "40 C", "REVIEW", "p15/p98 worst case"),
    ("cell limit", "60 C", "REVIEW", "p100 (T_s); p15 design goal is 45 C"),
    ("fan", "3 x Delta GFB0812ES-E", "DATASHEET", "P-Q curve, REVIEW p104"),
    ("fan shut-off", "150 mmH2O", "DATASHEET", "REVIEW p104, 12 V curve"),
    ("duct + grille K", "3.5 + 1.4", "ASSUMED", "not given in the review"),
    ("duct face area", "246 x 130 mm", "ASSUMED", "= bank frontal area; ducting "
                                                  "was suppressed before export"),
]


def provenance_table() -> str:
    w = max(len(r[0]) for r in PROVENANCE)
    lines = [f"{'parameter'.ljust(w)}  {'value':>18}  {'source':9}  note",
             "-" * (w + 60)]
    for name, val, src, note in PROVENANCE:
        lines.append(f"{name.ljust(w)}  {val:>18}  {src:9}  {note}")
    return "\n".join(lines)


if __name__ == "__main__":
    g = PackGeometry.from_layout()
    e = PackElectrical()
    print(g.describe())
    print()
    print(f"pack: {e.n_series}S{e.n_parallel}P, R = {e.pack_resistance*1000:.1f} mohm")
    print(f"  I = {e.pack_current:.2f} A, I_cell = {e.cell_current:.2f} A")
    print(f"  heat = {e.heat_load:.2f} W total, "
          f"{e.heat_per_cell*1000:.1f} mW/cell")
    print(f"  cell surface area (total) = {g.total_surface_area*1e4:.0f} cm^2")
    print()
    print(provenance_table())


# =============================================================================
# Cell thermal properties -- needed only by the TRANSIENT model
# =============================================================================
# The steady model never needed these: it solves an algebraic balance, so the
# cell's heat capacity never enters.  The transient model integrates
# dT/dt, so it does.
#
# Honesty note on tags: the LG INR18650 MJ1 datasheet publishes mass and
# electrical data but NOT thermal conductivity or specific heat.  Those two are
# LITERATURE ranges for cylindrical Li-ion cells, not datasheet values, and are
# tagged as such rather than laundered into a false DATASHEET claim.

@dataclass(frozen=True)
class CellThermal:
    """Per-cell thermal properties for the lumped-node transient model."""

    mass: float = 0.048                 # kg      DATASHEET   MJ1 nominal 48 g
    specific_heat: float = 1050.0       # J/kg.K  LITERATURE  1000-1100 typical
    k_radial: float = 0.5               # W/m.K   LITERATURE  0.2-1.0, anisotropic
    k_axial: float = 25.0               # W/m.K   LITERATURE  20-30 along the roll

    # Cell-to-cell conduction through tape, holder and tab. This is the weakest
    # number in the set and the first thing worth measuring.
    contact_conductance: float = 0.02   # W/K     ASSUMED

    # Manufacturing scatter, 1-sigma, as a fraction.
    sigma_R: float = 0.05               # -       ASSUMED  5% on R_int
    sigma_hA: float = 0.12              # -       ASSUMED  12% on the convective
                                        #                  coupling (bypass,
                                        #                  tape coverage, stack-up)
    sigma_G: float = 0.15               # -       ASSUMED  15% on contact G

    @property
    def heat_capacity(self) -> float:
        """C = m c_p  [J/K] -- the thermal mass of one cell."""
        return self.mass * self.specific_heat

    def biot(self, h: float, diameter: float = 0.0184) -> float:
        r"""$\mathrm{Bi} = h L_c / k$ with $L_c = D/4$ for a cylinder.

        Bi > 0.1 means the lumped assumption is formally violated and the node
        temperature is a VOLUME MEAN, not a surface temperature.  At the model's
        own h ~ 137 W/m^2.K this lands at 0.6-3.1, so the violation is real and
        must be reported rather than ignored.
        """
        return h * (diameter / 4.0) / self.k_radial

    def radial_rise(self, q_cell: float, length: float = 0.065) -> float:
        r"""Centre-to-surface rise of an internally generating cylinder,
        $\Delta T = \dot q / (4\pi k L)$ [K].  Added to the lumped mean when
        comparing against the 60 C SURFACE limit."""
        return q_cell / (4.0 * math.pi * self.k_radial * length)


CELL_THERMAL = CellThermal()


# =============================================================================
# Non-cell heat sources -- contactors and boards
# =============================================================================
# MEASURED placement (from Main Battery.step): the 4 Gigavacs sit at
# z = -583..-526 (at the green rear plate) and the two PCBs at y = 196..198
# (on the shelf).  Both are OUTSIDE the cell air path, so this heat raises the
# EXHAUST temperature and must NOT be added to the cell heat load -- doing that
# would inflate every cell temperature by ~11%.
#
# None of these values appear in the design review.  All ASSUMED.

@dataclass(frozen=True)
class ElectronicsHeat:
    """Heat from the contactors and boards, as a downstream plenum source."""

    n_contactors: int = 4
    contactor_coil_w: float = 4.0        # W each  ASSUMED economiser hold power
    contactor_contact_ohm: float = 100e-6  # ohm    ASSUMED 500 A-class contactor
    bms_pcb_w: float = 5.0               # W       ASSUMED
    hv_pdb_w: float = 10.0               # W       ASSUMED

    def total(self, pack_current: float) -> float:
        """Total non-cell dissipation [W] at a given pack current."""
        contact = self.n_contactors * pack_current**2 * self.contactor_contact_ohm
        coil = self.n_contactors * self.contactor_coil_w
        return contact + coil + self.bms_pcb_w + self.hv_pdb_w


ELECTRONICS = ElectronicsHeat()


PROVENANCE += [
    ("cell mass", "48 g", "DATASHEET", "LG INR18650 MJ1 nominal"),
    ("cell c_p", "1050 J/kg.K", "LITERATURE", "1000-1100 typical for 18650 Li-ion"),
    ("cell k_radial", "0.5 W/m.K", "LITERATURE", "0.2-1.0; sets the Biot number"),
    ("cell k_axial", "25 W/m.K", "LITERATURE", "20-30 along the jelly roll"),
    ("cell-cell conductance", "0.02 W/K", "ASSUMED", "tape + holder + tab; weakest value here"),
    ("R_int scatter", "5% 1-sigma", "ASSUMED", "manufacturing spread"),
    ("hA scatter", "12% 1-sigma", "ASSUMED", "bypass, tape coverage, stack-up"),
    ("contactor coil", "4 W each x4", "ASSUMED", "economiser hold; not in the review"),
    ("contactor contact R", "100 uohm", "ASSUMED", "500 A-class sealed contactor"),
    ("BMS + HV PDB", "5 + 10 W", "ASSUMED", "not in the review"),
    ("blue cutout (Front Plate)", "251.3 x 121.0 mm", "MEASURED", "Battery Box.step B-rep"),
    ("green cutout (Rear Plate)", "244.0 x 124.0 mm", "MEASURED", "Battery Box.step B-rep"),
    ("green cutout porosity", "0.886", "MEASURED", "purple Rear Panel truss"),
]

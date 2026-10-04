r"""
The two candidate airflow directions through the pack, and a comparison of them.

The CAD does not settle which way the air crosses the cell bank: the fans and
ducts were suppressed before the STEP export, so `CLAUDE.md` asks that the
airflow path be confirmed before the flow model is trusted.  Rather than pick
one, this module builds **both** and lets you toggle:

    FROM_12   air enters the 12-wide face and crosses the 17 rows (flow || z)
    FROM_17   air enters the 17-wide face and crosses the 12 columns (flow || x)

Both geometries are MEASURED from `BoxAssembly.step` via `out/cell_positions.csv`
-- neither is assumed.  `verify_against_cad()` re-derives both from the 408 cell
centres and checks them.

The second one is not obvious, and getting it wrong is the easy mistake here.
Viewed along $x$, the staggered lattice does **not** present 12 planes of 34
cells at the 25.2 mm row pitch.  It resolves into **24 planes at 10.25 mm**,
each holding 8 or 9 cells at **50.4 mm** pitch.  In staggered-tube-bank terms
that is

$$S_T = 50.4\ \text{mm},\qquad S_L = 2\times 10.25 = 20.5\ \text{mm},$$

with a half-row offset of 25.2 mm and $N_L = 12$ full rows.  So the transverse
pitch in that direction is 50.4 mm, not 25.2 mm, and the bank is far more open
than the other direction -- the minimum-area plane even switches from
transverse to diagonal.

Physics is NOT reimplemented here.  `flow.PackFlowModel` already takes an
arbitrary `PackGeometry` and its correlations are already parameterised on
$S_T$ and $S_L$, so each direction is just a different geometry handed to the
same solver.

Run:
    python src/direction.py                 # both, side by side
    python src/direction.py --only from_17  # one in detail
    python src/direction.py --export        # out/flow_from_12.json, flow_from_17.json
    python src/direction.py --verify        # re-derive both from the CAD
"""

from __future__ import annotations

import argparse
import csv
import enum
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from params import KELVIN, MM_H2O_TO_PA, DELTA_FAN, SAN_ACE_FAN, PackGeometry
import flow as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"

# Measured pitches, from out/pack_layout.json  (see verify_against_cad).
PITCH_X_MM = 20.5      # cell spacing within a z-plane
PITCH_Z_MM = 25.2      # z-plane spacing
STAGGER_MM = 10.25     # = PITCH_X / 2
N_ROWS_Z = 17
N_COLS_X = 12


class Direction(enum.Enum):
    """Which face the air enters."""

    FROM_12 = "from_12"   # 12-wide face, 17 rows deep, flow || z
    FROM_17 = "from_17"   # 17-wide face, 12 rows deep, flow || x

    @property
    def label(self) -> str:
        return {
            "from_12": "FROM THE 12  --  air enters the 12-wide face, "
                       "crosses 17 rows  (flow || z)",
            "from_17": "FROM THE 17  --  air enters the 17-wide face, "
                       "crosses 12 rows  (flow || x)",
        }[self.value]

    @property
    def short(self) -> str:
        return {"from_12": "from the 12", "from_17": "from the 17"}[self.value]

    @property
    def coord_key(self) -> str:
        """Which cell coordinate increases along the flow."""
        return {"from_12": "z_mm", "from_17": "x_mm"}[self.value]


def geometry_for(direction: Direction) -> PackGeometry:
    r"""The staggered-bank geometry seen by the air in one direction.

    A staggered bank is counted by its OFFSET PLANES: the "rows" the air crosses
    are the successive planes of tubes, each offset half a transverse pitch from
    the last.  Applying that consistently to the same 408-cell lattice gives

    | | FROM_12 (flow \|\| z) | FROM_17 (flow \|\| x) |
    |---|---|---|
    | $S_T$ across the flow | 20.5 mm | 50.4 mm |
    | $S_L$ plane to plane  | 25.2 mm | 10.25 mm |
    | $N_L$ planes          | 17      | 24      |
    | $N_T$ per plane, per bank | 12  | 8.5 (9/8 alternating) |
    | flow front            | 246.0 mm | 428.4 mm |

    Both recover 408 cells and the same 1.533 m^2 of wetted area, as they must --
    it is one lattice viewed two ways.  $S_D$ comes out at 27.20 mm either way,
    for the same reason.

    `n_parallel` is the GEOMETRIC count across the flow front, NOT the electrical
    12P; the two coincide only for FROM_12.  Conflating them is precisely what
    breaks when the direction is swapped, so the electrical configuration is left
    to `PackElectrical` and is untouched here.
    """
    if direction is Direction.FROM_12:
        return PackGeometry(
            n_parallel=N_COLS_X,                        # 12 across the front
            n_rows=N_ROWS_Z,                            # 17 planes deep
            transverse_pitch=PITCH_X_MM / 1000.0,       # 20.5 mm
            longitudinal_pitch=PITCH_Z_MM / 1000.0,     # 25.2 mm
        )

    return PackGeometry(
        n_parallel=N_ROWS_Z / 2.0,                      # 8.5 across (9/8 alternating)
        n_rows=2 * N_COLS_X,                            # 24 offset planes deep
        transverse_pitch=2 * PITCH_Z_MM / 1000.0,       # 50.4 mm
        longitudinal_pitch=STAGGER_MM / 1000.0,         # 10.25 mm plane pitch
    )


def solve(direction: Direction, fan=DELTA_FAN) -> F.FlowResult:
    """Run the existing solver on one direction's geometry."""
    return F.PackFlowModel(geometry=geometry_for(direction), fan=fan).solve()


# =============================================================================
# Per-cell field
# =============================================================================

def read_cells() -> list[dict]:
    """Read out/cell_positions.csv, rebuilding it from the STEP if it is absent.

    The inventory step owns that file, so it may legitimately not exist yet on a
    fresh checkout.  Rebuilding is cheap once the tessellation cache is warm.
    """
    path = OUT_DIR / "cell_positions.csv"
    if not path.exists():
        import subprocess
        print(f"  {path.name} missing -- running src/inventory.py to rebuild it")
        subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name("inventory.py"))],
                       check=True)
    with path.open() as fh:
        return list(csv.DictReader(fh))


def cell_temperatures(direction: Direction, result: F.FlowResult,
                      cells: list[dict] | None = None) -> np.ndarray:
    """Per-cell surface temperature [C] in cell_positions.csv order.

    The 1-D march resolves temperature by ROW, and which coordinate counts as
    the row index depends on the direction -- this is why the solver's own
    `cell_temperatures` cannot be reused for FROM_17: it indexes on the CSV's
    `row` column, which is the z-row, correct only for FROM_12.

    Cells are binned into $N_L$ equal bands along the flow coordinate, air
    entering at the low-coordinate end.
    """
    if cells is None:
        cells = read_cells()

    coord = np.array([float(c[direction.coord_key]) for c in cells])
    rt = np.asarray(result.row_cell_temperatures) - KELVIN
    n = len(rt)

    lo, hi = coord.min(), coord.max()
    idx = np.clip(((coord - lo) / (hi - lo + 1e-12) * n).astype(int), 0, n - 1)
    return rt[idx]


# =============================================================================
# Reporting
# =============================================================================

def describe(direction: Direction, r: F.FlowResult) -> str:
    g = geometry_for(direction)
    L, A = [], None
    out = []
    A = out.append
    A(f"=== {direction.label} ===")
    A("")
    A("  Bank as the air sees it   (measured from BoxAssembly.step)")
    A(f"    S_T across the flow ..... {g.transverse_pitch*1e3:8.2f} mm    S_T/D = {g.pitch_ratio_T:.3f}")
    A(f"    S_L plane to plane ...... {g.longitudinal_pitch*1e3:8.2f} mm    S_L/D = {g.pitch_ratio_L:.3f}")
    A(f"    S_D diagonal ............ {g.diagonal_pitch*1e3:8.2f} mm")
    A(f"    transverse gap .......... {(g.transverse_pitch-g.cell_diameter)*1e3:8.2f} mm")
    A(f"    diagonal gap ............ {2*(g.diagonal_pitch-g.cell_diameter)*1e3:8.2f} mm")
    binding = ("diagonal" if 2*(g.diagonal_pitch-g.cell_diameter)
               < (g.transverse_pitch-g.cell_diameter) else "transverse")
    A(f"    minimum-area plane ...... {binding:>8}")
    A(f"    A_min / A_face .......... {g.min_free_area_ratio:8.4f}")
    A(f"    planes the air crosses .. {g.n_rows:8d}")
    A(f"    cells per plane, /bank .. {g.n_parallel:8.1f}")
    A(f"    flow front .............. {g.duct_width*1e3:.1f} x {g.duct_height*1e3:.1f} mm"
      f"  = {g.duct_area*1e4:.1f} cm^2")
    A(f"    cells (check) ........... {g.n_cells:8.0f}")
    A(f"    wetted area (check) ..... {g.total_surface_area:8.4f} m^2")
    A("")
    A("  Operating point")
    A(f"    flow .................... {r.flow_cfm:8.1f} CFM   ({r.flow_m3min:.2f} m^3/min)")
    A(f"    face velocity ........... {r.inlet_velocity:8.2f} m/s")
    A(f"    V_max in the bank ....... {r.v_max:8.2f} m/s    (x{r.v_max/r.inlet_velocity:.2f})")
    A(f"    Re_max .................. {r.reynolds_max:8.0f}")
    A(f"    Nu / h .................. {r.nusselt:8.1f} / {r.h_conv:.1f} W/m^2.K")
    A(f"    f / chi ................. {r.friction_factor:8.3f} / {r.chi:.3f}")
    A("")
    A("  Pressure")
    A(f"    bank .................... {r.pressure_drop_bank:8.1f} Pa  "
      f"({r.pressure_drop_bank/MM_H2O_TO_PA:.1f} mmH2O)")
    A(f"    duct + grille ........... {r.pressure_drop_duct:8.1f} Pa  "
      f"({r.pressure_drop_duct/MM_H2O_TO_PA:.1f} mmH2O)")
    A(f"    total ................... {r.pressure_drop_total:8.1f} Pa  "
      f"({r.pressure_drop_total/MM_H2O_TO_PA:.1f} mmH2O)")
    share = 100 * r.pressure_drop_bank / r.pressure_drop_total
    A(f"    fan delivers ............ {r.fan_pressure:8.1f} Pa   [bank is {share:.0f}% of the drop]")
    A("")
    A("  Thermal")
    A(f"    inlet air ............... {r.inlet_temperature-KELVIN:8.1f} C")
    A(f"    exhaust air ............. {r.outlet_temperature-KELVIN:8.1f} C   "
      f"(rise {r.air_temperature_rise:.2f} C)")
    tc = r.cell_surface_temperature - KELVIN
    A(f"    hottest cell surface .... {tc:8.1f} C")
    A(f"    vs 45 C design goal ..... {'MET  ' if tc <= 45 else 'MISSED'}   "
      f"(margin {45-tc:+.1f} C)")
    A(f"    vs 60 C limit ........... {'PASS ' if tc <= 60 else 'FAIL '}   "
      f"(margin {60-tc:+.1f} C)")
    A("")
    A("  Checks")
    A(f"    energy balance error .... {r.energy_balance_error*100:8.3f} %")
    ok = "OK" if 1e3 < r.reynolds_max < 2e5 else "OUT OF RANGE"
    A(f"    Re in Zukauskas range ... {ok:>8}   (1e3 - 2e5)")
    return "\n".join(out)


def compare(ra: F.FlowResult, rb: F.FlowResult) -> str:
    ga = geometry_for(Direction.FROM_12)
    gb = geometry_for(Direction.FROM_17)
    tca = ra.cell_surface_temperature - KELVIN
    tcb = rb.cell_surface_temperature - KELVIN

    rows = [
        ("GEOMETRY", "", "", ""),
        ("  planes crossed  N_L", f"{ga.n_rows}", f"{gb.n_rows}", ""),
        ("  S_T across      [mm]", f"{ga.transverse_pitch*1e3:.2f}", f"{gb.transverse_pitch*1e3:.2f}", ""),
        ("  S_L along       [mm]", f"{ga.longitudinal_pitch*1e3:.2f}", f"{gb.longitudinal_pitch*1e3:.2f}", ""),
        ("  min gap         [mm]", f"{min(ga.transverse_pitch-ga.cell_diameter, 2*(ga.diagonal_pitch-ga.cell_diameter))*1e3:.2f}",
                                    f"{min(gb.transverse_pitch-gb.cell_diameter, 2*(gb.diagonal_pitch-gb.cell_diameter))*1e3:.2f}", ""),
        ("  A_min/A_face", f"{ga.min_free_area_ratio:.4f}", f"{gb.min_free_area_ratio:.4f}", f"{gb.min_free_area_ratio/ga.min_free_area_ratio:.2f}x"),
        ("  flow front      [mm]", f"{ga.duct_width*1e3:.0f}", f"{gb.duct_width*1e3:.0f}", ""),
        ("  face area      [cm^2]", f"{ga.duct_area*1e4:.0f}", f"{gb.duct_area*1e4:.0f}", f"{gb.duct_area/ga.duct_area:.2f}x"),
        ("FLOW", "", "", ""),
        ("  flow            [CFM]", f"{ra.flow_cfm:.1f}", f"{rb.flow_cfm:.1f}", f"{rb.flow_cfm/ra.flow_cfm:.2f}x"),
        ("  face velocity   [m/s]", f"{ra.inlet_velocity:.2f}", f"{rb.inlet_velocity:.2f}", ""),
        ("  V_max           [m/s]", f"{ra.v_max:.2f}", f"{rb.v_max:.2f}", f"{rb.v_max/ra.v_max:.2f}x"),
        ("  Re_max", f"{ra.reynolds_max:.0f}", f"{rb.reynolds_max:.0f}", ""),
        ("PRESSURE", "", "", ""),
        ("  bank             [Pa]", f"{ra.pressure_drop_bank:.0f}", f"{rb.pressure_drop_bank:.0f}", f"{rb.pressure_drop_bank/ra.pressure_drop_bank:.3f}x"),
        ("  total            [Pa]", f"{ra.pressure_drop_total:.0f}", f"{rb.pressure_drop_total:.0f}", f"{rb.pressure_drop_total/ra.pressure_drop_total:.3f}x"),
        ("THERMAL", "", "", ""),
        ("  h           [W/m^2.K]", f"{ra.h_conv:.1f}", f"{rb.h_conv:.1f}", f"{rb.h_conv/ra.h_conv:.2f}x"),
        ("  air rise          [C]", f"{ra.air_temperature_rise:.2f}", f"{rb.air_temperature_rise:.2f}", ""),
        ("  exhaust air       [C]", f"{ra.outlet_temperature-KELVIN:.1f}", f"{rb.outlet_temperature-KELVIN:.1f}", ""),
        ("  hottest cell      [C]", f"{tca:.1f}", f"{tcb:.1f}", f"{tcb-tca:+.1f} C"),
        ("  margin to 45 C    [C]", f"{45-tca:+.1f}", f"{45-tcb:+.1f}", ""),
        ("  margin to 60 C    [C]", f"{60-tca:+.1f}", f"{60-tcb:+.1f}", ""),
    ]
    w = max(len(r[0]) for r in rows)
    out = [f"{'':{w}}   {'from the 12':>13}   {'from the 17':>13}   ratio",
           f"{'':{w}}   {'12 wide, 17 dp':>13}   {'17 wide, 12 dp':>13}",
           "-" * (w + 50)]
    for name, va, vb, rr in rows:
        if va == "" and vb == "":
            out.append("")
            out.append(name)
        else:
            out.append(f"{name:{w}}   {va:>13}   {vb:>13}   {rr}")
    return "\n".join(out)


# =============================================================================
# CAD verification -- re-derive both geometries from the 408 cell centres
# =============================================================================

def verify_against_cad() -> dict:
    """Re-derive both directions' pitches from out/cell_positions.csv.

    Neither geometry is asserted: both are measured here and compared against
    what `geometry_for` builds, so a re-exported STEP that moved the pack would
    show up as a mismatch rather than passing silently.
    """
    cells = read_cells()
    b0 = [c for c in cells if c["bank"] == "0"]
    xs = np.array([float(c["x_mm"]) for c in b0])
    zs = np.array([float(c["z_mm"]) for c in b0])

    uz = np.unique(np.round(zs, 2))
    ux = np.unique(np.round(xs, 2))
    in_z = np.sort(np.unique(np.round(xs[np.isclose(zs, uz[0], atol=0.1)], 2)))
    in_x = np.sort(np.unique(np.round(zs[np.isclose(xs, ux[0], atol=0.1)], 2)))

    meas = {
        "n_cells": len(cells),
        "n_z_planes": int(len(uz)),
        "z_plane_pitch_mm": round(float(np.diff(uz).mean()), 3),
        "in_plane_pitch_along_z_mm": round(float(np.diff(in_z).mean()), 3),
        "n_x_planes": int(len(ux)),
        "x_plane_pitch_mm": round(float(np.diff(ux).mean()), 3),
        "in_plane_pitch_along_x_mm": round(float(np.diff(in_x).mean()), 3),
        "cells_per_x_plane_bank0": sorted({
            int(np.sum(np.isclose(xs, v, atol=0.1))) for v in ux}),
    }

    checks = []
    g12, g17 = geometry_for(Direction.FROM_12), geometry_for(Direction.FROM_17)
    checks.append(("FROM_12 S_T", g12.transverse_pitch*1e3, meas["in_plane_pitch_along_z_mm"]))
    checks.append(("FROM_12 S_L", g12.longitudinal_pitch*1e3, meas["z_plane_pitch_mm"]))
    checks.append(("FROM_12 N_L", float(g12.n_rows), float(meas["n_z_planes"])))
    checks.append(("FROM_17 S_T", g17.transverse_pitch*1e3, meas["in_plane_pitch_along_x_mm"]))
    checks.append(("FROM_17 S_L", g17.longitudinal_pitch*1e3, meas["x_plane_pitch_mm"]))
    checks.append(("FROM_17 N_L", float(g17.n_rows), float(meas["n_x_planes"])))
    checks.append(("cells FROM_12", float(g12.n_cells), float(meas["n_cells"])))
    checks.append(("cells FROM_17", float(g17.n_cells), float(meas["n_cells"])))
    checks.append(("wetted area m^2", g12.total_surface_area, g17.total_surface_area))

    meas["checks"] = [
        {"what": w, "model": round(m, 4), "cad": round(c, 4),
         "ok": bool(abs(m - c) < max(0.05, 1e-3 * abs(c)))}
        for w, m, c in checks
    ]
    meas["all_ok"] = all(c["ok"] for c in meas["checks"])
    return meas


# =============================================================================
# Export
# =============================================================================

def export(direction: Direction, r: F.FlowResult,
           cells: list[dict] | None = None) -> pathlib.Path:
    """Write out/flow_<direction>.json.

    `cell_temperatures_c` is exactly what `src/render.py --temps` reads: 408
    values in cell_positions.csv order.
    """
    OUT_DIR.mkdir(exist_ok=True)
    g = geometry_for(direction)
    temps = cell_temperatures(direction, r, cells)
    payload = {
        "direction": direction.value,
        "direction_label": direction.label,
        "geometry": {
            "S_T_mm": g.transverse_pitch * 1e3,
            "S_L_mm": g.longitudinal_pitch * 1e3,
            "S_D_mm": g.diagonal_pitch * 1e3,
            "n_transverse_per_bank": g.n_parallel,
            "n_planes": g.n_rows,
            "flow_front_mm": g.duct_width * 1e3,
            "face_area_cm2": g.duct_area * 1e4,
            "min_free_area_ratio": g.min_free_area_ratio,
            "n_cells": g.n_cells,
            "wetted_area_m2": g.total_surface_area,
        },
        "operating_point": {
            "flow_cfm": r.flow_cfm,
            "flow_m3s": r.volumetric_flow,
            "face_velocity_ms": r.inlet_velocity,
            "v_max_ms": r.v_max,
            "reynolds_max": r.reynolds_max,
            "nusselt": r.nusselt,
            "h_conv_w_m2k": r.h_conv,
            "friction_f": r.friction_factor,
            "chi": r.chi,
            "dp_bank_pa": r.pressure_drop_bank,
            "dp_duct_pa": r.pressure_drop_duct,
            "dp_total_pa": r.pressure_drop_total,
        },
        "temperatures": {
            "inlet_c": r.inlet_temperature - KELVIN,
            "outlet_c": r.outlet_temperature - KELVIN,
            "air_rise_c": r.air_temperature_rise,
            "cell_max_c": r.cell_surface_temperature - KELVIN,
            "margin_to_45_c": 45.0 - (r.cell_surface_temperature - KELVIN),
            "margin_to_60_c": 60.0 - (r.cell_surface_temperature - KELVIN),
            "plane_air_c": [float(t) - KELVIN for t in r.row_air_temperatures],
            "plane_cell_c": [float(t) - KELVIN for t in r.row_cell_temperatures],
        },
        "checks": {
            "energy_balance_error": r.energy_balance_error,
            "re_in_range": bool(1e3 < r.reynolds_max < 2e5),
        },
        "cell_temperatures_c": [float(t) for t in temps],
    }
    path = OUT_DIR / f"flow_{direction.value}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Airflow either way through the pack: 'from the 12' or 'from the 17'.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["from_12", "from_17"], default=None,
                    help="just one direction, in detail (default: both)")
    ap.add_argument("--fan", choices=["delta", "sanace"], default="delta",
                    help="delta = the fan in the CAD (default); sanace = the spec fan")
    ap.add_argument("--export", action="store_true",
                    help="write out/flow_from_12.json and out/flow_from_17.json")
    ap.add_argument("--verify", action="store_true",
                    help="re-derive both geometries from the CAD and check them")
    args = ap.parse_args()

    fan = DELTA_FAN if args.fan == "delta" else SAN_ACE_FAN

    if args.verify:
        v = verify_against_cad()
        print("CAD VERIFICATION  (out/cell_positions.csv, bank 0)")
        print(f"  cells {v['n_cells']},  z-planes {v['n_z_planes']} @ "
              f"{v['z_plane_pitch_mm']} mm,  x-planes {v['n_x_planes']} @ "
              f"{v['x_plane_pitch_mm']} mm")
        print(f"  cells per x-plane (bank 0): {v['cells_per_x_plane_bank0']}"
              "   [9/8 alternating -> 8.5 mean]")
        print()
        for c in v["checks"]:
            mark = "ok " if c["ok"] else "BAD"
            print(f"    [{mark}] {c['what']:18} model {c['model']:>10}   cad {c['cad']:>10}")
        print(f"\n  all checks: {'PASS' if v['all_ok'] else 'FAIL'}\n")

    dirs = ([Direction(args.only)] if args.only
            else [Direction.FROM_12, Direction.FROM_17])
    results = [(d, solve(d, fan)) for d in dirs]

    print(f"Fan: {fan.count} x {fan.name}    "
          f"inlet air {results[0][1].inlet_temperature-KELVIN:.0f} C\n")
    for d, r in results:
        print(describe(d, r))
        print()

    if len(results) == 2:
        print("=" * 72)
        print("SIDE BY SIDE")
        print("=" * 72)
        print(compare(results[0][1], results[1][1]))
        print()
        a, b = results[0][1], results[1][1]
        ta = a.cell_surface_temperature - KELVIN
        tb = b.cell_surface_temperature - KELVIN
        better = "from the 17" if tb < ta else "from the 12"
        print(f"  -> '{better}' runs cooler, by {abs(tb-ta):.1f} C.")
        print("     The 17-wide face is the open direction: S_T is 50.4 mm rather")
        print("     than 20.5 mm, so the minimum section is far larger and the bank")
        print(f"     resists {a.pressure_drop_bank/b.pressure_drop_bank:.0f}x less, "
              f"passing {b.flow_cfm/a.flow_cfm:.1f}x the air.")
        print()
        print("  NOTE: which face the fans actually blow through is NOT settled by")
        print("  the CAD -- the fans and ducts were suppressed before the STEP")
        print("  export.  Confirm against the design review before relying on either.")

    if args.export:
        cells = read_cells()
        print()
        for d, r in results:
            print(f"  wrote {export(d, r, cells)}")


if __name__ == "__main__":
    main()

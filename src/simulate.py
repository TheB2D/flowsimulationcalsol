"""
Run the flow + thermal model and write every result artifact.

    python src/simulate.py

Writes into `out/`:
  * `results.json`            -- operating point, row march, sweeps, validation
  * `cell_temperatures.csv`   -- 408 cells with their computed temperature
  * `cell_temperatures.json`  -- the same, as the array `render.py --temps` wants
  * the figures from `plots.py`

Then colour the geometry by the real field:

    python src/render.py --temperature --temps out/cell_temperatures.json
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import plots
from flow import PackFlowModel, summarize, validate_against_review
from params import (
    CFM_PER_M3MIN, KELVIN, MM_H2O_TO_PA,
    DELTA_FAN, SAN_ACE_FAN, OperatingConditions, PACK_5400W,
    PackElectrical, provenance_table,
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def cell_table(model: PackFlowModel, result) -> list[dict]:
    """Join the computed row temperatures onto the CAD cell positions."""
    path = OUT / "cell_positions.csv"
    if not path.exists():
        raise SystemExit("run src/inventory.py first -- out/cell_positions.csv is missing")

    with path.open() as fh:
        cells = list(csv.DictReader(fh))

    air = result.row_air_temperatures - KELVIN
    cell_T = result.row_cell_temperatures - KELVIN

    out = []
    for c in cells:
        r = int(c["row"])
        out.append({
            "cell_id": int(c["cell_id"]),
            "row": r, "col": int(c["col"]), "bank": int(c["bank"]),
            "x_mm": float(c["x_mm"]), "y_mm": float(c["y_mm"]), "z_mm": float(c["z_mm"]),
            "air_c": float(air[min(r, len(air) - 1)]),
            "cell_c": float(cell_T[min(r, len(cell_T) - 1)]),
        })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = PackFlowModel()
    res = model.solve()

    print(summarize(res, model))
    print()

    # ---- per-cell field ---------------------------------------------------
    cells = cell_table(model, res)
    with (OUT / "cell_temperatures.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0].keys()))
        w.writeheader()
        for c in cells:
            w.writerow(c)
    (OUT / "cell_temperatures.json").write_text(json.dumps(
        {"cell_temperatures_c": [c["cell_c"] for c in cells],
         "order": "out/cell_positions.csv (row, col, bank)"}, indent=1))

    # ---- sweeps -----------------------------------------------------------
    vel_sweep = []
    for v in np.linspace(0.15, 3.0, 60):
        r = model.solve(float(v) * model.geo.duct_area)
        vel_sweep.append({
            "velocity_ms": float(v),
            "flow_cfm": r.flow_cfm,
            "dp_total_pa": r.pressure_drop_total,
            "re_max": r.reynolds_max,
            "h_w_m2k": r.h_conv,
            "air_out_c": r.outlet_temperature - KELVIN,
            "cell_max_c": r.cell_surface_temperature - KELVIN,
        })

    heat_sweep = []
    for q in np.linspace(100, 700, 40):
        r = PackFlowModel(heat_load=float(q)).solve(res.volumetric_flow)
        heat_sweep.append({
            "heat_w": float(q),
            "cell_max_c": r.cell_surface_temperature - KELVIN,
            "air_out_c": r.outlet_temperature - KELVIN,
        })

    ambient_sweep = []
    for a in np.linspace(15, 50, 36):
        m = PackFlowModel(conditions=OperatingConditions(
            inlet_temperature=float(a) + KELVIN))
        r = m.solve()
        ambient_sweep.append({
            "inlet_c": float(a),
            "cell_max_c": r.cell_surface_temperature - KELVIN,
            "flow_cfm": r.flow_cfm,
        })

    # ---- variants ---------------------------------------------------------
    import dataclasses
    variants = []
    for label, m in [
        ("design (3 x Delta, 295.7 W, 40 C)", model),
        ("higher load 318.8 W (review p17)",
         PackFlowModel(electrical=PACK_5400W)),
        ("San Ace 80 x3 (spec'd fan, assumed curve)",
         PackFlowModel(fan=SAN_ACE_FAN)),
        ("2 fans (one failed)",
         PackFlowModel(fan=dataclasses.replace(DELTA_FAN, count=2))),
        ("1 fan (two failed)",
         PackFlowModel(fan=dataclasses.replace(DELTA_FAN, count=1))),
        ("50 C ambient",
         PackFlowModel(conditions=OperatingConditions(
             inlet_temperature=50.0 + KELVIN))),
    ]:
        r = m.solve()
        variants.append({
            "case": label,
            "flow_cfm": r.flow_cfm,
            "velocity_ms": r.inlet_velocity,
            "dp_total_pa": r.pressure_drop_total,
            "air_out_c": r.outlet_temperature - KELVIN,
            "cell_max_c": r.cell_surface_temperature - KELVIN,
            "margin_to_limit_c": (m.cond.cell_temperature_limit
                                  - r.cell_surface_temperature),
            "pass": bool(r.cell_surface_temperature <= m.cond.cell_temperature_limit),
        })

    g = model.geo
    data = {
        "meta": {
            "geometry_source": "BoxAssembly.step, measured (Task 1)",
            "physics_source": "Incropera & DeWitt 7e Sec. 7.6 (Zukauskas)",
            "review": "Excalibur Battery Design Review.pdf",
            "provenance": provenance_table(),
        },
        "geometry": {
            "n_cells": g.n_cells, "n_rows": g.n_rows,
            "n_parallel": g.n_parallel, "n_banks": g.n_banks,
            "cell_diameter_mm": g.cell_diameter * 1000,
            "tube_length_mm": g.tube_length * 1000,
            "S_T_mm": g.transverse_pitch * 1000,
            "S_L_mm": g.longitudinal_pitch * 1000,
            "S_D_mm": g.diagonal_pitch * 1000,
            "S_T_over_D": g.pitch_ratio_T, "S_L_over_D": g.pitch_ratio_L,
            "face_area_cm2": g.duct_area * 1e4,
            "min_free_area_ratio": g.min_free_area_ratio,
            "total_cell_surface_cm2": g.total_surface_area * 1e4,
        },
        "operating_point": {
            "flow_cfm": res.flow_cfm, "flow_m3min": res.flow_m3min,
            "face_velocity_ms": res.inlet_velocity, "v_max_ms": res.v_max,
            "mass_flow_kgs": res.mass_flow,
            "reynolds_max": res.reynolds_max, "nusselt": res.nusselt,
            "h_w_m2k": res.h_conv,
            "friction_factor": res.friction_factor, "chi": res.chi,
            "dp_bank_pa": res.pressure_drop_bank,
            "dp_duct_pa": res.pressure_drop_duct,
            "dp_total_pa": res.pressure_drop_total,
            "dp_total_mmH2O": res.pressure_drop_total / MM_H2O_TO_PA,
            "bank_share_of_dp": res.pressure_drop_bank / res.pressure_drop_total,
            "heat_w": model.heat_load,
            "inlet_c": res.inlet_temperature - KELVIN,
            "outlet_c": res.outlet_temperature - KELVIN,
            "air_rise_c": res.air_temperature_rise,
            "cell_max_c": res.cell_surface_temperature - KELVIN,
            "cell_rise_c": res.cell_temperature_rise,
            "limit_c": model.cond.cell_temperature_limit - KELVIN,
            "target_c": model.cond.cell_temperature_target - KELVIN,
            "margin_to_limit_c": (model.cond.cell_temperature_limit
                                  - res.cell_surface_temperature),
            "pass": bool(res.cell_surface_temperature
                         <= model.cond.cell_temperature_limit),
            "energy_balance_error": res.energy_balance_error,
        },
        "rows": [
            {"row": i + 1,
             "air_c": float(a - KELVIN), "cell_c": float(c - KELVIN),
             "h_w_m2k": float(h), "re_max": float(re)}
            for i, (a, c, h, re) in enumerate(zip(
                res.row_air_temperatures, res.row_cell_temperatures,
                res.row_h, res.row_reynolds))
        ],
        "velocity_sweep": vel_sweep,
        "heat_sweep": heat_sweep,
        "ambient_sweep": ambient_sweep,
        "variants": variants,
        "validation": validate_against_review(),
    }
    (OUT / "results.json").write_text(json.dumps(data, indent=1, default=_jsonable))

    # ---- figures ----------------------------------------------------------
    plots.main()

    # ---- report -----------------------------------------------------------
    print("VARIANTS")
    print(f"  {'case':44s} {'CFM':>7} {'cell C':>8} {'margin':>8}  verdict")
    for v in variants:
        print(f"  {v['case']:44s} {v['flow_cfm']:7.1f} {v['cell_max_c']:8.1f} "
              f"{v['margin_to_limit_c']:+8.1f}  {'PASS' if v['pass'] else 'FAIL'}")
    print()
    print("VALIDATION vs the review's MATLAB study")
    print(f"  {'v [m/s]':>8} {'source':>16} {'review':>8} {'model':>8} {'ratio':>7}")
    for c in data["validation"]["cases"]:
        print(f"  {c['velocity_ms']:8.4f} {c['source']:>16} "
              f"{c['review_rise_c']:7.2f}C {c['model_rise_c']:7.2f}C "
              f"{c['ratio']:7.3f}")
    print()
    for f in ("results.json", "cell_temperatures.csv", "cell_temperatures.json"):
        print(f"wrote out/{f}  ({(OUT / f).stat().st_size/1024:.0f} KB)")
    print()
    print("colour the 3D geometry by this field with:")
    print("  .venv/bin/python src/render.py --temperature "
          "--temps out/cell_temperatures.json")


if __name__ == "__main__":
    main()

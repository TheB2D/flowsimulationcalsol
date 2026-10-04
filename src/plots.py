"""
Fan curve and heat-simulation figures.

    python src/plots.py            # writes every figure into out/

Figures:
  1. fan_curve.png        -- fan vs system curve, operating point marked
  2. thermal_march.png    -- air and cell temperature row by row
  3. velocity_sweep.png   -- cell temperature vs flow, with the operating point
  4. sensitivity.png      -- heat load, ambient, and fan-count sweeps
  5. summary.png          -- all four on one sheet
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flow import PackFlowModel, validate_against_review
from params import (
    CFM_PER_M3MIN, KELVIN, MM_H2O_TO_PA,
    DELTA_FAN, SAN_ACE_FAN, OperatingConditions, PackGeometry,
)

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"

# house style, matching the project's design-system palette
INK = "#18181B"
ACCENT = "#2563EB"
WARN = "#DC2626"
MUTED = "#64748B"
GRID = "#E4E4E7"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 160,
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
    "figure.facecolor": "white",
})


def _cfm(ax_q: np.ndarray) -> np.ndarray:
    """m^3/s -> CFM."""
    return ax_q * 60.0 * CFM_PER_M3MIN


# =============================================================================
# 1. Fan curve vs system curve
# =============================================================================

def fan_curve_figure(model: PackFlowModel, ax=None):
    r"""Fan $\Delta p(Q)$ against system $\Delta p(Q)$, with the intersection marked."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))

    res = model.solve()

    qf, pf = model.fan_curve(300)
    qs, ps = model.system_curve(300, q_max=model.fan.free_delivery * 1.15)

    ax.plot(_cfm(qf), pf / MM_H2O_TO_PA, color=ACCENT, lw=2.2,
            label=f"{model.fan.count} x {model.fan.name} (parallel)")
    ax.plot(_cfm(qs), ps / MM_H2O_TO_PA, color=WARN, lw=2.2,
            label="system resistance (bank + duct)")

    # the bank's share alone, to show what dominates
    air_q = [model.system_pressure_drop(q, __import__("flow").AirProperties(
        model.cond.inlet_temperature + 5.0))[0] for q in qs]
    ax.plot(_cfm(qs), np.array(air_q) / MM_H2O_TO_PA, color=WARN, lw=1.0,
            ls=":", label="  of which the cell bank")

    qop = res.volumetric_flow
    pop = res.pressure_drop_total / MM_H2O_TO_PA
    ax.plot([_cfm(np.array([qop]))[0]], [pop], "o", ms=9, color=INK, zorder=5)
    ax.annotate(
        f"operating point\n{res.flow_cfm:.1f} CFM @ {pop:.0f} mmH$_2$O\n"
        f"face velocity {res.inlet_velocity:.2f} m/s",
        xy=(res.flow_cfm, pop), xytext=(res.flow_cfm + 30, pop - 42),
        fontsize=8.5, color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))

    # the review's assumed point, for contrast
    v_review = 1.3684
    q_review = v_review * model.geo.duct_area
    qr_cfm = float(_cfm(np.array([q_review]))[0])
    ax.axvline(qr_cfm, color=MUTED, lw=1.0, ls="--")
    ax.annotate(f"review assumed\n1.3684 m/s ({qr_cfm:.0f} CFM)",
                xy=(qr_cfm, 30), xytext=(qr_cfm + 6, 30),
                fontsize=7.8, color=MUTED)

    ax.set_xlabel("volumetric flow  [CFM]")
    ax.set_ylabel(r"static pressure  [mmH$_2$O]")
    ax.set_title("Fan curve vs system resistance")
    # zoom on the region that matters -- the fan's free delivery is far to the
    # right of where this system actually sits
    ax.set_xlim(0, max(res.flow_cfm * 2.6, 160))
    ax.set_ylim(0, model.fan.shutoff_pressure / MM_H2O_TO_PA * 1.12)
    ax.legend(loc="upper right", fontsize=8.0)

    sec = ax.secondary_yaxis(
        "right", functions=(lambda v: v * MM_H2O_TO_PA, lambda v: v / MM_H2O_TO_PA))
    sec.set_ylabel("static pressure  [Pa]")

    if own:
        fig.tight_layout()
        fig.savefig(OUT / "fan_curve.png")
        plt.close(fig)
    return res


# =============================================================================
# 2. Row-by-row thermal march
# =============================================================================

def thermal_march_figure(model: PackFlowModel, ax=None):
    """Air and cell-surface temperature as the flow crosses the 17 rows."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))

    res = model.solve()
    rows = np.arange(1, model.geo.n_rows + 1)
    air = res.row_air_temperatures - KELVIN
    cell = res.row_cell_temperatures - KELVIN
    T_in = model.cond.inlet_temperature - KELVIN

    ax.plot(np.r_[0, rows], np.r_[T_in, air], "-o", ms=3.5, color=ACCENT,
            lw=1.8, label="air temperature")
    ax.plot(rows, cell, "-s", ms=3.5, color=WARN, lw=1.8,
            label="cell surface temperature")

    ax.axhline(model.cond.cell_temperature_limit - KELVIN, color=WARN,
               ls="--", lw=1.2)
    ax.text(0.4, model.cond.cell_temperature_limit - KELVIN + 0.4,
            "60 C limit", color=WARN, fontsize=8)
    ax.axhline(model.cond.cell_temperature_target - KELVIN, color=MUTED,
               ls=":", lw=1.2)
    ax.text(0.4, model.cond.cell_temperature_target - KELVIN + 0.4,
            "45 C design goal", color=MUTED, fontsize=8)

    ax.annotate(f"hottest cell {cell.max():.1f} C",
                xy=(rows[np.argmax(cell)], cell.max()),
                xytext=(rows[np.argmax(cell)] - 6.5, cell.max() + 2.2),
                fontsize=8.5, color=WARN,
                arrowprops=dict(arrowstyle="->", color=WARN, lw=1.0))

    ax.set_xlabel("row along the flow  (1 = inlet, 17 = exhaust)")
    ax.set_ylabel("temperature  [C]")
    ax.set_title(f"Temperature through the bank  ({model.heat_load:.0f} W, "
                 f"{res.flow_cfm:.0f} CFM)")
    ax.set_xlim(0, model.geo.n_rows + 0.5)
    lo = T_in - 1.0
    hi = max(model.cond.cell_temperature_limit - KELVIN + 2.5, cell.max() + 3)
    ax.set_ylim(lo, hi)
    ax.legend(loc="center right", fontsize=8.5)

    if own:
        fig.tight_layout()
        fig.savefig(OUT / "thermal_march.png")
        plt.close(fig)
    return res


# =============================================================================
# 3. Velocity sweep
# =============================================================================

def velocity_sweep_figure(model: PackFlowModel, ax=None):
    """Hottest-cell temperature against face velocity, operating point marked."""
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))

    res = model.solve()
    vs = np.linspace(0.15, 3.0, 70)
    cell, air = [], []
    for v in vs:
        r = model.solve(v * model.geo.duct_area)
        cell.append(r.cell_surface_temperature - KELVIN)
        air.append(r.outlet_temperature - KELVIN)

    ax.plot(vs, cell, color=WARN, lw=2.2, label="hottest cell surface")
    ax.plot(vs, air, color=ACCENT, lw=1.8, label="exhaust air")

    ax.axhline(model.cond.cell_temperature_limit - KELVIN, color=WARN,
               ls="--", lw=1.2)
    ax.text(2.35, model.cond.cell_temperature_limit - KELVIN + 1.2,
            "60 C limit", color=WARN, fontsize=8)
    ax.axhline(model.cond.cell_temperature_target - KELVIN, color=MUTED,
               ls=":", lw=1.2)
    ax.text(2.35, model.cond.cell_temperature_target - KELVIN + 1.2,
            "45 C goal", color=MUTED, fontsize=8)

    ax.plot([res.inlet_velocity], [res.cell_surface_temperature - KELVIN],
            "o", ms=9, color=INK, zorder=5)
    ax.annotate(f"operating point\n{res.inlet_velocity:.2f} m/s, "
                f"{res.cell_surface_temperature-KELVIN:.1f} C",
                xy=(res.inlet_velocity, res.cell_surface_temperature - KELVIN),
                xytext=(res.inlet_velocity + 0.45,
                        res.cell_surface_temperature - KELVIN + 9),
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))

    # the review's MATLAB points, as an independent check
    val = validate_against_review()["cases"]
    vx = [c["velocity_ms"] for c in val]
    vy = [c["review_rise_c"] + model.cond.inlet_temperature - KELVIN for c in val]
    ax.plot(vx, vy, "^", ms=7, color=MUTED, ls="none",
            label="review MATLAB study (slides 18, 105)")

    ax.set_xlabel("face velocity into the bank  [m/s]")
    ax.set_ylabel("temperature  [C]")
    ax.set_title("Cooling performance vs flow")
    ax.set_xlim(0, 3.0)
    ax.set_ylim(model.cond.inlet_temperature - KELVIN - 1, 95)
    ax.legend(loc="upper right", fontsize=8.2)

    if own:
        fig.tight_layout()
        fig.savefig(OUT / "velocity_sweep.png")
        plt.close(fig)


# =============================================================================
# 4. Sensitivity
# =============================================================================

def sensitivity_figure(model: PackFlowModel, axes=None):
    """How the result moves with heat load, ambient temperature and fan count.

    Every sweep is rebuilt from `model`'s OWN geometry, fan, electrical and
    conditions via `_variant`, so a non-default geometry is honoured.  The
    sweeps used to construct bare `PackFlowModel(...)` objects, which silently
    fell back to the default lattice and quietly ignored whatever geometry the
    caller passed in.
    """
    own = axes is None
    if own:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    def _variant(**kw) -> PackFlowModel:
        base = dict(geometry=model.geo, fan=model.fan,
                    electrical=model.elec, conditions=model.cond)
        base.update(kw)
        return PackFlowModel(**base)

    res = model.solve()
    ax1, ax2, ax3 = axes

    # -- heat load ----------------------------------------------------------
    loads = np.linspace(100, 700, 40)
    t = [_variant(heat_load=float(q)).solve(res.volumetric_flow)
         .cell_surface_temperature - KELVIN for q in loads]
    ax1.plot(loads, t, color=WARN, lw=2.2)
    ax1.axhline(model.cond.cell_temperature_limit - KELVIN, color=WARN, ls="--", lw=1.2)
    ax1.axvline(model.heat_load, color=INK, ls=":", lw=1.2)
    ax1.plot([model.heat_load], [res.cell_surface_temperature - KELVIN], "o",
             ms=8, color=INK)
    # where does it hit the limit?
    over = np.argmax(np.array(t) > model.cond.cell_temperature_limit - KELVIN)
    if over:
        ax1.annotate(f"60 C at {loads[over]:.0f} W\n"
                     f"= {loads[over]/model.heat_load:.1f}x design load",
                     xy=(loads[over], model.cond.cell_temperature_limit - KELVIN),
                     xytext=(loads[over] - 260, model.cond.cell_temperature_limit - KELVIN + 4),
                     fontsize=8.2, color=WARN,
                     arrowprops=dict(arrowstyle="->", color=WARN, lw=1.0))
    ax1.set_xlabel("pack heat dissipation  [W]")
    ax1.set_ylabel("hottest cell  [C]")
    ax1.set_title("vs heat load", fontsize=10)

    # -- ambient ------------------------------------------------------------
    amb = np.linspace(15, 50, 40)
    t2 = []
    for a in amb:
        m = _variant(conditions=OperatingConditions(
            inlet_temperature=float(a) + KELVIN))
        t2.append(m.solve().cell_surface_temperature - KELVIN)
    ax2.plot(amb, t2, color=WARN, lw=2.2)
    ax2.axhline(model.cond.cell_temperature_limit - KELVIN, color=WARN, ls="--", lw=1.2)
    ax2.axvline(model.cond.inlet_temperature - KELVIN, color=INK, ls=":", lw=1.2)
    ax2.plot([model.cond.inlet_temperature - KELVIN],
             [res.cell_surface_temperature - KELVIN], "o", ms=8, color=INK)
    ax2.annotate("worst case\n40 C", xy=(40, res.cell_surface_temperature - KELVIN),
                 xytext=(25, res.cell_surface_temperature - KELVIN + 5),
                 fontsize=8.2, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax2.set_xlabel("inlet air temperature  [C]")
    ax2.set_ylabel("hottest cell  [C]")
    ax2.set_title("vs ambient", fontsize=10)

    # -- fan count ----------------------------------------------------------
    counts = [1, 2, 3, 4, 5]
    import dataclasses
    flows, temps = [], []
    for n in counts:
        f = dataclasses.replace(model.fan, count=n)
        r = _variant(fan=f).solve()
        flows.append(r.flow_cfm)
        temps.append(r.cell_surface_temperature - KELVIN)
    ax3b = ax3.twinx()
    ax3.bar(counts, flows, width=0.55, color=ACCENT, alpha=0.30, zorder=1)
    ax3b.plot(counts, temps, "-s", color=WARN, lw=2.0, ms=6, zorder=3)
    ax3b.axhline(model.cond.cell_temperature_limit - KELVIN, color=WARN,
                 ls="--", lw=1.2, zorder=2)

    ax3.set_xlabel("number of fans in parallel")
    ax3.set_ylabel("flow  [CFM]", color=ACCENT)
    ax3.tick_params(axis="y", colors=ACCENT)
    ax3b.set_ylabel("hottest cell  [C]", color=WARN)
    ax3b.tick_params(axis="y", colors=WARN)
    ax3.set_xticks(counts)
    # give each axis its own honest range: the bars must not be read against
    # the temperature scale, and the 60 C limit must stay visible
    ax3.set_ylim(0, max(flows) * 1.25)
    ax3b.set_ylim(min(temps) - 2,
                  max(model.cond.cell_temperature_limit - KELVIN + 2,
                      max(temps) + 2))
    ax3.set_title("vs fan count  (3 = as designed)", fontsize=10)
    # the temperature line must sit in front of the bars
    ax3b.set_zorder(3); ax3b.patch.set_visible(False)
    ax3.set_zorder(1)
    ax3b.grid(False)
    ax3b.text(counts[-1], model.cond.cell_temperature_limit - KELVIN + 0.4,
              "60 C limit", color=WARN, fontsize=7.5, ha="right")
    ax3.annotate(f"{flows[2]:.0f} CFM", xy=(3, flows[2]), xytext=(3, flows[2] * 1.06),
                 fontsize=7.8, color=ACCENT, ha="center")

    if own:
        fig.tight_layout()
        fig.savefig(OUT / "sensitivity.png")
        plt.close(fig)


# =============================================================================
# Summary sheet
# =============================================================================

def summary_sheet(model: PackFlowModel):
    fig = plt.figure(figsize=(14.5, 9.6))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.30,
                          left=0.055, right=0.955, top=0.895, bottom=0.075)

    res = model.solve()

    ax1 = fig.add_subplot(gs[0, 0]); fan_curve_figure(model, ax1)
    ax2 = fig.add_subplot(gs[0, 1]); thermal_march_figure(model, ax2)
    ax3 = fig.add_subplot(gs[0, 2]); velocity_sweep_figure(model, ax3)
    axs = [fig.add_subplot(gs[1, i]) for i in range(3)]
    sensitivity_figure(model, axs)

    verdict = ("PASS" if res.cell_surface_temperature
               <= model.cond.cell_temperature_limit else "FAIL")
    goal = ("met" if res.cell_surface_temperature
            <= model.cond.cell_temperature_target else "missed")
    fig.suptitle(
        "CalSol Excalibur battery pack - airflow and thermal model     "
        f"{res.flow_cfm:.0f} CFM  |  hottest cell "
        f"{res.cell_surface_temperature-KELVIN:.1f} C  |  "
        f"60 C limit {verdict} ({model.cond.cell_temperature_limit-KELVIN-(res.cell_surface_temperature-KELVIN):+.1f} C)  |  "
        f"45 C goal {goal}",
        fontsize=12.5, fontweight="bold", y=0.966)
    fig.text(0.055, 0.935,
             "geometry measured from BoxAssembly.step  -  "
             f"{model.geo.n_cells} cells, {model.geo.n_rows} rows x "
             f"{model.geo.n_parallel} across x {model.geo.n_banks} banks, staggered, "
             f"S_T {model.geo.transverse_pitch*1000:.1f} / S_L "
             f"{model.geo.longitudinal_pitch*1000:.1f} / D "
             f"{model.geo.cell_diameter*1000:.1f} mm  -  "
             f"worst case {model.cond.inlet_temperature-KELVIN:.0f} C inlet, "
             f"{model.heat_load:.0f} W, fan-driven flow only",
             fontsize=8.6, color=MUTED)

    fig.savefig(OUT / "summary.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = PackFlowModel()

    fan_curve_figure(model)
    thermal_march_figure(model)
    velocity_sweep_figure(model)
    sensitivity_figure(model)
    summary_sheet(model)

    for f in ("fan_curve.png", "thermal_march.png", "velocity_sweep.png",
              "sensitivity.png", "summary.png"):
        p = OUT / f
        print(f"wrote out/{f}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()

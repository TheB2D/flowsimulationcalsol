"""
Figures for the LaTeX writeup.

Reuses `plots.py`'s house style -- same palette, same rcParams, same limit-line
and operating-point conventions -- so the report reads as one document with the
dashboards rather than two different projects.

    .venv/bin/python src/report_figs.py       # writes writeup/figs/*.pdf

PDF rather than PNG: these are vector plots and LaTeX embeds them at full
resolution with no resampling.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import plots as P
from plots import INK, ACCENT, WARN, MUTED, GRID
import direction as D
import network as N
import placement as PL
import transient as TRN
from flow import PackFlowModel
from params import KELVIN, DELTA_FAN, CFM_PER_M3MIN, MM_H2O_TO_PA

FIGS = pathlib.Path(__file__).resolve().parent.parent / "writeup" / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"savefig.dpi": 200, "figure.dpi": 140})


def _save(fig, name: str) -> None:
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote writeup/figs/{name}.pdf")


def fig_network(model) -> None:
    """Fan characteristics vs the system curve, one-sided and push-pull.

    The point of the figure is that series staging moves the INTERSECTION, not
    the system curve: the bank is unchanged, the available pressure is not.
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    one = N.one_sided(3, "front")
    pp = N.push_pull(3, 3)

    q = np.linspace(1e-5, 0.13, 300)
    from flow import AirProperties
    air = AirProperties(model.cond.inlet_temperature + 5.0)
    sysp = np.array([sum(model.system_pressure_drop(x, air)) for x in q])

    ax.plot(q * 60 * CFM_PER_M3MIN, sysp, color=WARN, lw=2.2,
            label="system resistance (bank + duct)")
    for top, col, ls, lab in ((one, ACCENT, "-", "3 fans, one face"),
                              (pp, "#7C3AED", "-", "3 push + 3 pull (series)")):
        pr = np.array([top.net_pressure(x) for x in q])
        ax.plot(q * 60 * CFM_PER_M3MIN, pr, color=col, lw=2.0, ls=ls, label=lab)
        r = N.solve_topology(top, model)
        ax.plot([r.flow_cfm], [r.pressure_drop_total], "o", ms=9,
                color=INK, zorder=5)
        ax.annotate(f"{r.flow_cfm:.0f} CFM\n{r.cell_surface_temperature-KELVIN:.1f} C",
                    xy=(r.flow_cfm, r.pressure_drop_total),
                    xytext=(22, -30 if top is one else 26),
                    textcoords="offset points", fontsize=8.5, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.1))

    ax.set_xlabel("volumetric flow  [CFM]")
    ax.set_ylabel("static pressure  [Pa]")
    ax.set_title("Series staging moves the operating point")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 3600)
    ax.legend(fontsize=8.5, loc="upper left")
    sec = ax.secondary_yaxis("right",
                             functions=(lambda p: p / MM_H2O_TO_PA,
                                        lambda m: m * MM_H2O_TO_PA))
    sec.set_ylabel("static pressure  [mmH$_2$O]")
    _save(fig, "network")


def fig_fan_count(model) -> None:
    """Why adding fans does not help on the cutout axis."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    counts = np.array([1, 2, 3, 4, 6, 8, 10, 12])

    for ax, d, title in ((a1, D.Direction.FROM_12, "flow $\\parallel z$  (the cutout axis)"),
                         (a2, D.Direction.FROM_17, "flow $\\parallel x$  (the open axis)")):
        cfm, tc = [], []
        for n in counts:
            r = D.solve(d, dataclasses.replace(DELTA_FAN, count=int(n)))
            cfm.append(r.flow_cfm)
            tc.append(r.cell_surface_temperature - KELVIN)
        ax.bar(counts, cfm, width=0.62, color=ACCENT, alpha=0.30, label="flow")
        ax.set_ylabel("flow  [CFM]", color=ACCENT)
        ax.tick_params(axis="y", colors=ACCENT)
        ax.set_xlabel("fans in parallel on one face")
        ax.set_title(title, fontsize=10)

        axb = ax.twinx()
        axb.plot(counts, tc, "-s", color=WARN, lw=2.0, ms=5)
        axb.set_ylabel("hottest cell  [C]", color=WARN)
        axb.tick_params(axis="y", colors=WARN)
        axb.axhline(45.0, color=MUTED, ls=":", lw=1.2)
        axb.text(counts[0], 45.15, "45 C goal", color=MUTED, fontsize=8)
        axb.set_zorder(3); axb.patch.set_visible(False); axb.grid(False)
        axb.annotate(f"{tc[2]-tc[-1]:.2f} C for {11*DELTA_FAN.rated_power:.0f} W more",
                     xy=(counts[2], tc[2]), xytext=(10, 22),
                     textcoords="offset points", fontsize=8.2, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    fig.suptitle("Parallel fans add capacity to a curve that has already collapsed",
                 fontsize=11.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "fan_count")


def fig_pareto(model) -> None:
    """The placement search: temperature against fan power."""
    cands = PL.search(model)
    front = PL.pareto(cands)
    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    ax.plot([c.power for c in cands], [c.t_cell for c in cands], "o", ms=6,
            color=MUTED, alpha=0.5, label="all 30 feasible arrangements")
    ax.plot([c.power for c in front], [c.t_cell for c in front], "-s",
            color=ACCENT, lw=2.2, ms=7, label="Pareto front")

    one = [c for c in cands if c.n_fans == 3
           and (not c.topology.inlets or not c.topology.outlets)]
    if one:
        b = one[0]
        ax.plot([b.power], [b.t_cell], "X", ms=12, color=WARN, zorder=6)
        ax.annotate(f"current design\n3 fans one side\n{b.t_cell:.2f} C, {b.power:.0f} W",
                    xy=(b.power, b.t_cell), xytext=(18, 12),
                    textcoords="offset points", fontsize=8.5, color=WARN,
                    arrowprops=dict(arrowstyle="->", color=WARN, lw=1.1))
    k = front[1]
    ax.annotate(f"1+1 push-pull\n{k.t_cell:.2f} C, {k.power:.0f} W\ncooler AND cheaper",
                xy=(k.power, k.t_cell), xytext=(28, 28),
                textcoords="offset points", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))

    ax.axhline(45.0, color=MUTED, ls=":", lw=1.2)
    ax.text(20, 45.08, "45 C design goal", color=MUTED, fontsize=8)
    ax.set_xlabel("total fan power  [W]")
    ax.set_ylabel("hottest cell surface  [C]")
    ax.set_title("Placement search over the measured cutouts")
    ax.legend(fontsize=8.5, loc="upper right")
    _save(fig, "pareto")


def fig_transient(model) -> None:
    """Transient response and the scatter distribution."""
    r = N.solve_topology(N.push_pull(3, 3), model)
    tr = TRN.build(model, r)
    tr.run(t_end=600.0, dt=0.5)
    t, tmax, tmean = tr.history()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    a1.plot(t, tmax, color=WARN, lw=2.2, label="hottest cell")
    a1.plot(t, tmean, color=ACCENT, lw=1.8, label="pack mean")
    a1.axhline(45.0, color=MUTED, ls=":", lw=1.2)
    a1.text(8, 45.1, "45 C design goal", color=MUTED, fontsize=8)
    tau = tr.tau
    a1.axvline(tau, color=INK, ls=":", lw=1.1)
    a1.annotate(rf"$\tau = {tau:.0f}$ s", xy=(tau, tmean[len(tmean)//6]),
                xytext=(26, -4), textcoords="offset points",
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    a1.set_xlabel("time  [s]")
    a1.set_ylabel("temperature  [C]")
    a1.set_title("Transient from a cold start", fontsize=10)
    a1.legend(fontsize=8.5, loc="lower right")

    dev = tr.deviation_c()
    a2.hist(dev, bins=30, color=ACCENT, alpha=0.75, edgecolor="white", lw=0.5)
    a2.axvline(0.0, color=INK, ls=":", lw=1.2)
    a2.set_xlabel("departure from row mean  [K]")
    a2.set_ylabel("cells")
    a2.set_title(rf"Cell-to-cell scatter  ($\sigma_R$ = {tr.th.sigma_R:.0%}, "
                 rf"spread {dev.std():.3f} K)", fontsize=10)
    a2.annotate("1--2\\% of the 4.2 K\nrow-to-row rise",
                xy=(dev.max() * 0.72, a2.get_ylim()[1] * 0.62),
                fontsize=8.5, color=MUTED)
    fig.tight_layout()
    _save(fig, "transient")


def fig_march(model) -> None:
    """Air and cell temperature along the flow, both topologies."""
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for top, col, lab in ((N.one_sided(3, "front"), MUTED, "3 fans, one face"),
                          (N.push_pull(3, 3), ACCENT, "3 push + 3 pull")):
        r = N.solve_topology(top, model)
        rows = np.arange(1, len(r.row_cell_temperatures) + 1)
        ax.plot(rows, np.asarray(r.row_cell_temperatures) - KELVIN, "-s",
                color=col, lw=2.0, ms=4, label=f"cell surface, {lab}")
        ax.plot(rows, np.asarray(r.row_air_temperatures) - KELVIN, "--o",
                color=col, lw=1.4, ms=3, alpha=0.65,
                label=f"air, {lab}")
    ax.axhline(60.0, color=WARN, ls="--", lw=1.2)
    ax.text(1, 60.4, "60 C limit", color=WARN, fontsize=8)
    ax.axhline(45.0, color=MUTED, ls=":", lw=1.2)
    ax.text(1, 45.3, "45 C design goal", color=MUTED, fontsize=8)
    ax.set_xlabel("row along the flow  (1 = inlet, 17 = exhaust)")
    ax.set_ylabel("temperature  [C]")
    ax.set_title("Temperature through the bank")
    ax.set_ylim(39, 62)
    ax.legend(fontsize=8, loc="center left")
    _save(fig, "march")


def main() -> None:
    model = PackFlowModel(geometry=D.geometry_for(D.Direction.FROM_12))
    print("writing report figures:")
    fig_network(model)
    fig_fan_count(model)
    fig_pareto(model)
    fig_march(model)
    fig_transient(model)


if __name__ == "__main__":
    main()

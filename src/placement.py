r"""
Mode 1 -- find the best fan arrangement over the box's real openings.

The search space is deliberately small and is enumerated EXHAUSTIVELY rather
than handed to an optimiser: each face is {off, in, out} with 0..3 fans (3 is
the measured limit -- an 80 mm frame fits three times across a 251 mm cutout),
which is a few dozen feasible arrangements, and each one costs a single Brent
solve of a few milliseconds.  Exhaustive enumeration is therefore both cheaper
than an optimiser and provably optimal within the space.

What the search is actually for
-------------------------------
Not "how many fans".  Running the solver over fan count on this axis shows the
objective is nearly flat:

    3 fans one-sided    93.2 CFM   47.0 C    75.6 W
    6 fans one-sided    94.4 CFM   46.9 C   151.2 W
   12 fans one-sided    94.7 CFM   46.9 C   302.4 W

Going 3 -> 12 buys 0.1 C for 227 W.  The cell bank is ~98% of the system
resistance and the fans sit at ~99% of their dead-head pressure, so adding fans
in PARALLEL adds capacity to a curve that has already collapsed.

The decision that matters is the TOPOLOGY -- which faces, and in which sense:

    1 push + 1 pull    107.1 CFM   46.2 C    50.4 W
    3 push + 3 pull    131.6 CFM   45.1 C   151.2 W

One fan on each face beats three on one face, at two thirds of the power,
because series staging adds pressure where the system is pressure-starved.

Because the objective is flat in fan count but steep in topology, reporting a
single "winner" would be misleading.  `pareto()` returns the non-dominated
front of (peak cell temperature, fan power) so the trade is visible.
"""

from __future__ import annotations

import itertools
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flow import PackFlowModel, FlowResult
from params import KELVIN, DELTA_FAN, SAN_ACE_FAN, Fan
from openings import FRONT_PLATE, REAR_PLATE
import network as N


@dataclass
class Candidate:
    """One evaluated arrangement."""

    topology: N.Topology
    result: FlowResult
    fan_name: str

    @property
    def t_cell(self) -> float:
        """Peak cell surface temperature [C] -- the objective."""
        return self.result.cell_surface_temperature - KELVIN

    @property
    def power(self) -> float:
        return self.topology.total_power

    @property
    def flow_cfm(self) -> float:
        return self.result.flow_cfm

    @property
    def n_fans(self) -> int:
        return self.topology.n_fans

    def row(self) -> str:
        return (f"{self.topology.describe()[:46]:46s} "
                f"{self.flow_cfm:7.1f} CFM  {self.t_cell:6.2f} C  "
                f"{self.power:6.1f} W")


def enumerate_topologies(fan: Fan = DELTA_FAN,
                         max_per_face: int | None = None
                         ) -> list[N.Topology]:
    """Every physically buildable arrangement over the two cut panels.

    `max_per_face` defaults to each opening's MEASURED fan capacity, so the
    search cannot propose something that will not fit.
    """
    lim_f = max_per_face or FRONT_PLATE.max_fans
    lim_r = max_per_face or REAR_PLATE.max_fans

    tops: list[N.Topology] = []
    for nf, nr in itertools.product(range(lim_f + 1), range(lim_r + 1)):
        if nf == 0 and nr == 0:
            continue
        for sf, sr in itertools.product((+1, -1), repeat=2):
            # Sense is meaningless on an unpowered face; fix it so the same
            # arrangement is not enumerated twice.
            if nf == 0 and sf != +1:
                continue
            if nr == 0 and sr != +1:
                continue
            top = N.Topology(
                name="",
                sets=[N.FanSet(fan, nf, FRONT_PLATE, sf),
                      N.FanSet(fan, nr, REAR_PLATE, sr)])
            if not top.is_through_flow:
                continue
            bits = []
            if nf:
                bits.append(f"{nf} blue {'in' if sf > 0 else 'out'}")
            if nr:
                bits.append(f"{nr} green {'in' if sr > 0 else 'out'}")
            top.name = " + ".join(bits)
            tops.append(top)
    return tops


def search(model: PackFlowModel | None = None,
           fans: tuple[Fan, ...] = (DELTA_FAN,),
           verbose: bool = False) -> list[Candidate]:
    """Evaluate every feasible arrangement, best (coolest) first."""
    if model is None:
        import direction as D
        model = PackFlowModel(geometry=D.geometry_for(D.Direction.FROM_12))

    out: list[Candidate] = []
    for fan in fans:
        for top in enumerate_topologies(fan):
            try:
                res = N.solve_topology(top, model)
            except N.InfeasibleTopology:
                continue
            out.append(Candidate(top, res, fan.name))
            if verbose:
                print("  " + out[-1].row())
    out.sort(key=lambda c: (c.t_cell, c.power, c.n_fans))
    return out


def pareto(cands: list[Candidate]) -> list[Candidate]:
    """Non-dominated front of (temperature, power), both minimised.

    Reported instead of a single winner because the objective is nearly flat in
    fan count -- a lone "best" would hide that most of the benefit arrives with
    the first fan on the second face.
    """
    front: list[Candidate] = []
    for c in sorted(cands, key=lambda c: (c.power, c.t_cell)):
        if not front or c.t_cell < front[-1].t_cell - 1e-9:
            front.append(c)
    return front


def report(model: PackFlowModel | None = None, top_n: int = 10) -> str:
    cands = search(model)
    best = cands[0]
    lines = ["FAN PLACEMENT SEARCH",
             f"  {len(cands)} feasible arrangements over the two measured cutouts",
             f"  (max {FRONT_PLATE.max_fans} fans per face -- 80 mm frames across "
             f"a {FRONT_PLATE.width*1e3:.0f} mm opening)", "",
             f"Best by peak cell temperature:", "  " + best.row(), "",
             f"Top {top_n}:"]
    for c in cands[:top_n]:
        lines.append("  " + c.row())
    lines += ["", "Pareto front (temperature vs fan power):"]
    for c in pareto(cands):
        lines.append("  " + c.row())

    one = [c for c in cands if c.n_fans == 3 and
           (not c.topology.inlets or not c.topology.outlets)]
    if one:
        base = one[0]
        lines += ["", "Against the current one-sided 3-fan design:",
                  f"  now  {base.t_cell:.2f} C at {base.power:.0f} W",
                  f"  best {best.t_cell:.2f} C at {best.power:.0f} W"
                  f"   ({base.t_cell - best.t_cell:+.2f} C)"]
    lines += ["",
              "Note: the objective is nearly flat in fan count and steep in",
              "topology. Series staging (one face in, the other out) adds",
              "pressure, which is what a bank at 98% of the system resistance",
              "needs; more fans in parallel on one face add almost nothing."]
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())

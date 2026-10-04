r"""
Fan network composition: how several fan sets combine against one system curve.

The existing `Fan.pressure()` implements only the PARALLEL law -- it divides the
total flow by `count` and reads the single-fan curve.  That is correct for fans
side by side on one face, and it is all the steady model ever needed.

The two-sided arrangement the user asked about is different, and it is the
reason this module exists.  When one set of fans pushes at the blue panel and
another pulls at the green panel, the two sets are **in series**: the same air
passes through both, so their pressure RISES ADD at equal flow,

$$\Delta p_{\rm push}(Q) + \Delta p_{\rm pull}(Q) \;=\; \Delta p_{\rm sys}(Q)$$

against fans side by side, where the FLOWS add at equal pressure,

$$Q_1(\Delta p) + Q_2(\Delta p) \;=\; Q .$$

Which law applies is not cosmetic here.  The cell bank is ~98% of the system
resistance and the fans sit at ~99% of their dead-head pressure, so adding fans
in parallel buys almost nothing (3 -> 12 fans is worth 0.1 C) while putting the
same number in series nearly reaches the design goal.  Series staging is the
whole result.

Mass balance
------------
Steady incompressible continuity over the box:

$$\sum_{i \in \text{inlets}} Q_i \;=\; \sum_{j \in \text{outlets}} Q_j$$

The box is sealed apart from the two cutouts, so if every fan blows INWARD there
is no exit path and the equation has no solution for $Q > 0$: the box simply
pressurises until one face reverses.  `Topology.solve` raises
`InfeasibleTopology` in that case rather than returning a plausible number,
because a silently-wrong answer is worse than an error.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flow import AirProperties, PackFlowModel, FlowResult
from params import KELVIN, MM_H2O_TO_PA, DELTA_FAN, Fan
from openings import Opening, FRONT_PLATE, REAR_PLATE


class InfeasibleTopology(ValueError):
    """Raised when a fan arrangement violates continuity.

    The common case is "both panels blow in": a sealed box with two inlets and
    no outlet has no steady solution.
    """


# =============================================================================
# One set of identical fans on one opening
# =============================================================================

@dataclass(frozen=True)
class FanSet:
    """`n` identical fans in parallel on a single opening.

    `sense` is +1 for a set that drives air INTO the box and -1 for one that
    draws air OUT.  Both senses contribute a positive pressure rise to a
    through-flow; the sign is used for the continuity bookkeeping, not to
    negate the curve.
    """

    fan: Fan
    n: int
    opening: Opening
    sense: int = +1            # +1 push in, -1 pull out

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("fan count must be >= 0")
        if self.n > self.opening.max_fans:
            raise ValueError(
                f"{self.n} fans will not fit {self.opening.name} "
                f"(max {self.opening.max_fans} by measured width)")
        if self.sense not in (+1, -1):
            raise ValueError("sense must be +1 (in) or -1 (out)")

    @property
    def active(self) -> bool:
        return self.n > 0

    @property
    def total_power(self) -> float:
        """Electrical power drawn by this set [W]."""
        return self.n * self.fan.rated_power

    def pressure(self, q_total: float) -> float:
        """Static rise [Pa] this set produces when passing `q_total` [m^3/s].

        Delegates to the existing `Fan.pressure`, which already implements the
        parallel law across `n` fans.
        """
        if not self.active:
            return 0.0
        staged = dataclasses.replace(self.fan, count=self.n)
        return staged.pressure(q_total)

    @property
    def free_delivery(self) -> float:
        if not self.active:
            return 0.0
        return self.fan.curve[-1, 0] / 60.0 * self.n

    def flow_at(self, dp: float) -> float:
        """Inverse curve: flow [m^3/s] this set delivers against `dp` [Pa].

        Needed for the parallel law, which composes on flow at equal pressure.
        The curve is monotonically decreasing but has a nearly flat shoulder
        near shut-off (150.0 -> 149.0 mmH2O over the first 0.5 m^3/min), so a
        plain root find can land anywhere along it.  Bracket on pressure and
        take the lowest-flow root.
        """
        if not self.active:
            return 0.0
        hi = self.free_delivery
        if dp >= self.pressure(0.0):
            return 0.0
        if dp <= self.pressure(hi):
            return hi
        return float(brentq(lambda q: self.pressure(q) - dp, 0.0, hi, xtol=1e-12))


# =============================================================================
# Composition laws
# =============================================================================

def compose_series(sets: list[FanSet], q: float) -> float:
    r"""Series: equal flow, pressures add.  $\Delta p = \sum_i \Delta p_i(Q)$."""
    return sum(s.pressure(q) for s in sets if s.active)


def compose_parallel(sets: list[FanSet], dp: float) -> float:
    r"""Parallel: equal pressure, flows add.  $Q = \sum_i Q_i(\Delta p)$."""
    return sum(s.flow_at(dp) for s in sets if s.active)


# =============================================================================
# A whole arrangement
# =============================================================================

@dataclass
class Topology:
    """A complete fan arrangement over the box's openings."""

    name: str
    sets: list[FanSet] = field(default_factory=list)

    @property
    def inlets(self) -> list[FanSet]:
        return [s for s in self.sets if s.active and s.sense > 0]

    @property
    def outlets(self) -> list[FanSet]:
        return [s for s in self.sets if s.active and s.sense < 0]

    @property
    def n_fans(self) -> int:
        return sum(s.n for s in self.sets)

    @property
    def total_power(self) -> float:
        return sum(s.total_power for s in self.sets)

    @property
    def passive_vents(self) -> list[FanSet]:
        """Openings carrying no fan.

        A cutout without a fan is still a hole: air can leave through it, driven
        by the pressure the powered face develops.  That is what makes the
        current one-sided design work -- 3 fans at the blue panel, exhausting
        through the open green panel.
        """
        return [s for s in self.sets if not s.active]

    @property
    def is_through_flow(self) -> bool:
        """True when air has both a way in and a way out.

        Either a powered inlet and a powered outlet (push-pull), or a powered
        face plus at least one unobstructed opening to vent through.
        """
        if self.inlets and self.outlets:
            return True
        powered = bool(self.inlets) or bool(self.outlets)
        return powered and bool(self.passive_vents)

    def check_feasible(self) -> None:
        """Raise unless continuity can be satisfied."""
        if not self.inlets and not self.outlets:
            raise InfeasibleTopology(f"{self.name}: no active fans")
        if not self.is_through_flow:
            blowing = "in" if self.inlets else "out"
            raise InfeasibleTopology(
                f"{self.name}: every opening has a fan and all of them blow "
                f"{blowing}. The box is sealed apart from these cutouts, so "
                f"continuity sum(Q_in) = sum(Q_out) has no solution for Q > 0 "
                f"-- the box would pressurise until one face reversed. Use one "
                f"inlet and one outlet (push-pull), or leave a face unpowered "
                f"so it can vent.")

    def net_pressure(self, q: float) -> float:
        r"""Total rise available at flow `q`.

        Inlet and outlet sets pass the same air, so they are in series and
        their rises add.  Within a face, the fans are in parallel, which
        `FanSet.pressure` already handles.
        """
        return compose_series(self.inlets, q) + compose_series(self.outlets, q)

    def mass_balance_residual(self, q: float) -> float:
        """sum(Q_in) - sum(Q_out).  Zero for any through-flow topology, since
        the same q passes every stage; non-zero means the arrangement cannot
        conserve mass."""
        if self.is_through_flow:
            return 0.0
        return q * (len(self.inlets) - len(self.outlets))

    @property
    def max_flow(self) -> float:
        """An upper bracket for the root find."""
        cands = [s.free_delivery for s in self.sets if s.active]
        return max(cands) * 1.5 if cands else 1.0

    def describe(self) -> str:
        bits = []
        for s in self.sets:
            if s.active:
                bits.append(f"{s.n}x {'in' if s.sense > 0 else 'out'} @ {s.opening.name}")
        return f"{self.name}: " + (", ".join(bits) if bits else "no fans")


# =============================================================================
# Solving
# =============================================================================

def find_operating_point(topology: Topology, model: PackFlowModel,
                         iterations: int = 4) -> float:
    r"""Solve $\Delta p_{\rm net}(Q) = \Delta p_{\rm sys}(Q)$ -> Q [m^3/s].

    This mirrors `PackFlowModel.find_operating_point` exactly, including its
    outer loop that iterates the mean air temperature to consistency -- the
    system curve depends on density, density on how much the air has warmed,
    and that on the flow.  Only the left-hand side differs: one fan curve there,
    the composed network characteristic here.

    Keeping the two procedures identical is what lets a single-face topology
    reproduce `flow.py`'s numbers to machine precision, which is the regression
    gate that proves this module added capability without changing physics.
    """
    topology.check_feasible()

    T_mean = model.cond.inlet_temperature + 5.0
    q = None

    for _ in range(iterations):
        air = AirProperties(T_mean)

        def residual(qq: float) -> float:
            dp_bank, dp_duct = model.system_pressure_drop(qq, air)
            return topology.net_pressure(qq) - (dp_bank + dp_duct)

        lo, hi = 1e-6, topology.max_flow
        if residual(lo) < 0:            # system already harder than shut-off
            return lo
        while residual(hi) > 0 and hi < 10.0:
            hi *= 1.5
        q_new = float(brentq(residual, lo, hi, xtol=1e-10))

        if q is not None and abs(q_new - q) < 1e-9:
            q = q_new
            break
        q = q_new

        mdot = q * AirProperties(model.cond.inlet_temperature).density
        cp = AirProperties(T_mean).specific_heat
        dT = model.heat_load / (mdot * cp)
        T_mean = model.cond.inlet_temperature + 0.5 * dT

    return q


def solve_topology(topology: Topology, model: PackFlowModel) -> FlowResult:
    """Operating point for an arbitrary fan arrangement.

    Returns the project's existing `FlowResult`, so every downstream consumer
    -- `plots.py`, `render.py`, `direction.cell_temperatures` -- keeps working
    unchanged.
    """
    return model.solve(find_operating_point(topology, model))


def network_curve(topology: Topology, n: int = 200
                  ) -> tuple[np.ndarray, np.ndarray]:
    """The composed characteristic, for plotting beside the system curve."""
    q = np.linspace(0.0, topology.max_flow / 1.5, n)
    return q, np.array([topology.net_pressure(x) for x in q])


# =============================================================================
# The named arrangements
# =============================================================================

def one_sided(n: int = 3, face: str = "front", fan: Fan = DELTA_FAN) -> Topology:
    """Fans on one panel only -- the current design.

    Physically this still needs an exit, which the opposite cutout provides as
    a passive vent; the vent's loss is folded into the duct K-factor.
    """
    opening = FRONT_PLATE if face == "front" else REAR_PLATE
    other = REAR_PLATE if face == "front" else FRONT_PLATE
    return Topology(
        name=f"one-sided ({face}, {n} fans)",
        sets=[FanSet(fan, n, opening, +1),
              FanSet(fan, 0, other, -1)])


def push_pull(n_push: int = 3, n_pull: int = 3, fan: Fan = DELTA_FAN) -> Topology:
    """Push at the blue panel, pull at the green -- the user's two-sided case.

    The two sets are in series, so their pressure rises add.  This is the
    arrangement that actually beats a high-resistance bank.
    """
    return Topology(
        name=f"push-pull ({n_push} blue in, {n_pull} green out)",
        sets=[FanSet(fan, n_push, FRONT_PLATE, +1),
              FanSet(fan, n_pull, REAR_PLATE, -1)])


def both_in(n: int = 3, fan: Fan = DELTA_FAN) -> Topology:
    """Both panels blowing inward -- infeasible, kept so the UI can show why."""
    return Topology(
        name=f"both-in ({n} + {n})",
        sets=[FanSet(fan, n, FRONT_PLATE, +1),
              FanSet(fan, n, REAR_PLATE, +1)])

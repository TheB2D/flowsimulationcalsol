r"""
Mode 3 -- the stochastic transient: heat spreading and randomising over time.

This is the one part of the project that the existing steady-state model cannot
supply.  `flow.py` solves an algebraic balance, so a cell's heat capacity never
enters and there is no time index anywhere.  Here each cell becomes a lumped
node with thermal mass and the system is integrated forward.

The equation
------------
For cell $i$, with $C_i = m c_p$ its heat capacity:

$$C_i\,\frac{dT_i}{dt} \;=\; \dot q_i
  \;-\; h_i A_i\,(T_i - T_{{\rm air},r(i)})
  \;+\; \sum_{j \in \mathcal N(i)} G_{ij}\,(T_j - T_i)
  \;+\; \underbrace{\sigma_\eta\,\xi_i(t)}_{\text{optional, NOT physics}}$$

Randomness enters two ways, and the user asked for both:

(a) PROPERTY SCATTER -- the physical one.  Manufacturing spread gives every
    cell a slightly different internal resistance and a different convective
    coupling:

    $$\dot q_i = I_{\rm cell}^2 R_i,\quad R_i = R_{\rm int}(1+\epsilon_i),
      \quad \epsilon_i \sim \mathcal N(0, \sigma_R^2)$$

    Drawn once from a seeded generator, so a run is reproducible.

(b) ADDITIVE NOISE -- a visual effect, off by default and labelled as such.
    Implemented as Ornstein-Uhlenbeck rather than white noise, because white
    noise just shimmers and reads as a rendering bug:

    $$d\xi_i = -\frac{\xi_i}{\tau}\,dt + \sqrt{\tfrac{2}{\tau}}\;dW_i$$

    Scaled by $\sqrt{\Delta t}$ so refining the timestep does not silently
    change the variance.

An honesty note that matters
----------------------------
The scatter is SMALL.  At sigma_R = 5% the per-cell spread is ~60 mK against a
~6 K rise along the flow -- about 1% of the visible gradient.  Rather than
quietly inflating sigma to make the animation lively, `scatter_gain` is a
separate, explicit multiplier that defaults to 1.0 and is labelled in the UI as
a visualisation aid rather than physics.

Numerics
--------
Explicit Euler.  Stability needs $\Delta t < \min_i C_i / G_i^{\rm tot}$, which
for this pack is 35-49 s; at the default dt = 0.5 s that is a ~70x margin.  The
system is not stiff -- convection dominates conduction by ~10x -- so an implicit
scheme would cost a 408x408 solve per step and buy nothing.  Measured at ~1 us
per step for all 408 nodes, which is why mode 3 can re-solve live instead of
replaying precomputed frames.

The air is treated as quasi-steady: its transit time through the bank is 0.31 s
against a cell time constant of 49 s, a 158x separation, so solving it
algebraically each step is correct to O(1/158).

Lumped-capacitance validity
---------------------------
$\mathrm{Bi} = h(D/4)/k_r \approx 0.6-3.1$ for this pack, which is ABOVE the
0.1 threshold.  The lumped node is therefore a VOLUME-MEAN temperature, not a
surface temperature, and `surface_temperatures()` adds the steady radial rise
$\dot q/(4\pi k L)$ before any comparison against the 60 C surface limit.
"""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from flow import PackFlowModel, FlowResult, AirProperties
from params import KELVIN, CELL_THERMAL, CellThermal, ELECTRONICS


@dataclass
class TransientState:
    """One instant of the integration."""

    t: float                       # s
    T: np.ndarray                  # (N,) cell volume-mean temperature [K]
    xi: np.ndarray                 # (N,) OU noise state
    step: int = 0


class StochasticTransient:
    r"""Lumped-node transient over the cell lattice.

    Parameters
    ----------
    model, result
        The steady solution supplies $h$, the per-row air temperatures and the
        flow.  They are reused rather than re-derived.
    order
        (row, col, bank) per cell, in `cell_positions.csv` order, so the result
        lines up with the renderer.
    """

    def __init__(self,
                 model: PackFlowModel,
                 result: FlowResult,
                 order: list[tuple[int, int, int]],
                 thermal: CellThermal = CELL_THERMAL,
                 seed: int = 0,
                 scatter_gain: float = 1.0,
                 noise_sigma: float = 0.0,
                 noise_tau: float = 20.0,
                 include_electronics: bool = True) -> None:
        self.model = model
        self.result = result
        self.order = order
        self.th = thermal
        self.seed = seed
        self.scatter_gain = scatter_gain
        self.noise_sigma = noise_sigma          # W, 0 = off (the default)
        self.noise_tau = noise_tau
        self.include_electronics = include_electronics

        self.n = len(order)
        self.rows = np.array([r for r, _, _ in order], dtype=int)

        g = model.geo
        self.area = math.pi * g.cell_diameter * g.tube_length   # m^2 per cell
        self.C = self.th.heat_capacity                           # J/K per cell

        # Air temperature entering each cell's row, from the steady march.
        rt = np.asarray(result.row_air_temperatures)
        if rt.size == 0:
            rt = np.full(max(self.rows) + 1, model.cond.inlet_temperature)
        idx = np.clip(self.rows, 0, len(rt) - 1)
        self.T_air = np.where(idx > 0, rt[np.maximum(idx - 1, 0)],
                              model.cond.inlet_temperature)

        self.h_nominal = result.h_conv
        self._build_neighbours()
        self.reset(seed)

    # -- lattice ------------------------------------------------------------

    def _build_neighbours(self) -> None:
        """Neighbour pairs from the (row, col, bank) lattice.

        Conduction is symmetric by construction -- each unordered pair is stored
        once and applied to both nodes with opposite sign -- which is what keeps
        the conduction term out of the global energy balance entirely.  An
        asymmetric G is the usual source of slow drift in this kind of model.
        """
        pos = {rcb: k for k, rcb in enumerate(self.order)}
        pairs = []
        for (r, c, b), k in pos.items():
            for dr, dc in ((1, 0), (0, 1)):
                j = pos.get((r + dr, c + dc, b))
                if j is not None:
                    pairs.append((k, j))
        self.pair_i = np.array([p[0] for p in pairs], dtype=int)
        self.pair_j = np.array([p[1] for p in pairs], dtype=int)

    # -- setup --------------------------------------------------------------

    def reset(self, seed: int | None = None) -> TransientState:
        """Redraw every random field from `seed` and restart at ambient.

        All scatter is drawn up front, in a fixed order, from one generator, so
        a run is bit-reproducible regardless of timestep or UI interaction.
        """
        if seed is not None:
            self.seed = seed
        rng = np.random.default_rng(self.seed)

        sR = self.th.sigma_R * self.scatter_gain
        sH = self.th.sigma_hA * self.scatter_gain
        sG = self.th.sigma_G * self.scatter_gain

        eps = np.clip(rng.normal(0.0, sR, self.n), -3 * sR, 3 * sR) if sR > 0 \
            else np.zeros(self.n)
        gam = np.clip(rng.normal(0.0, sH, self.n), -3 * sH, 3 * sH) if sH > 0 \
            else np.zeros(self.n)
        gg = np.clip(rng.normal(0.0, sG, self.pair_i.size), -3 * sG, 3 * sG) \
            if sG > 0 else np.zeros(self.pair_i.size)

        # Per-cell generation, scaled so the pack total stays at the model's
        # heat load whatever the scatter does.
        q_mean = self.model.heat_load / self.n
        self.q = q_mean * (1.0 + eps)
        self.q *= self.model.heat_load / self.q.sum()

        self.hA = self.h_nominal * self.area * (1.0 + gam)
        self.G = self.th.contact_conductance * (1.0 + gg)

        # Electronics heat is NOT added here: the contactors sit at the rear
        # plate and the boards on the shelf, both outside the cell air path, so
        # they raise the exhaust rather than any cell.  See `exhaust_rise`.

        T0 = np.full(self.n, self.model.cond.inlet_temperature)
        self.state = TransientState(t=0.0, T=T0, xi=np.zeros(self.n), step=0)
        self._hist_t: list[float] = [0.0]
        self._hist_max: list[float] = [float(T0.max() - KELVIN)]
        self._hist_mean: list[float] = [float(T0.mean() - KELVIN)]
        self._frames: list[np.ndarray] = [T0.copy()]
        # Running integral of heat carried off by the air, for the energy check.
        self._removed_int: list[float] = [0.0]
        return self.state

    # -- stability ----------------------------------------------------------

    @property
    def dt_max(self) -> float:
        r"""Explicit-Euler stability limit $\min_i C_i/G_i^{\rm tot}$ [s]."""
        gtot = self.hA.copy()
        np.add.at(gtot, self.pair_i, self.G)
        np.add.at(gtot, self.pair_j, self.G)
        return float(self.C / gtot.max())

    @property
    def tau(self) -> float:
        """Convective time constant C/(hA) [s] -- sets how long to run."""
        return float(self.C / np.mean(self.hA))

    @property
    def biot(self) -> float:
        return self.th.biot(self.h_nominal, self.model.geo.cell_diameter)

    # -- integration --------------------------------------------------------

    def step(self, dt: float = 0.5, n: int = 1) -> TransientState:
        """Advance `n` sub-steps of `dt` seconds."""
        if dt > 0.5 * self.dt_max:
            raise ValueError(
                f"dt = {dt:.2f} s exceeds half the stability limit "
                f"({self.dt_max:.1f} s). Lower dt or raise the conductance.")

        s = self.state
        T, xi = s.T, s.xi
        for _ in range(n):
            flux = self.q - self.hA * (T - self.T_air)

            # conduction, symmetric: equal and opposite on each pair
            dT_pair = T[self.pair_j] - T[self.pair_i]
            cond = np.zeros_like(T)
            np.add.at(cond, self.pair_i, self.G * dT_pair)
            np.add.at(cond, self.pair_j, -self.G * dT_pair)

            if self.noise_sigma > 0.0:
                rng = np.random.default_rng(self.seed + s.step)
                dW = rng.normal(0.0, 1.0, self.n) * math.sqrt(dt)
                xi = xi + (-xi / self.noise_tau) * dt \
                    + math.sqrt(2.0 / self.noise_tau) * dW
                flux = flux + self.noise_sigma * xi

            # Integrate the convective removal with the SAME flux the step
            # used, so the energy check is exact for this discretisation
            # rather than approximately right.
            self._removed_int.append(
                self._removed_int[-1]
                + float(np.sum(self.hA * (T - self.T_air))) * dt)

            T = T + dt / self.C * (flux + cond)
            s.step += 1

        s.T, s.xi, s.t = T, xi, s.t + dt * n
        self._hist_t.append(s.t)
        self._hist_max.append(float(T.max() - KELVIN))
        self._hist_mean.append(float(T.mean() - KELVIN))
        self._frames.append(T.copy())
        return s

    def run(self, t_end: float = 400.0, dt: float = 0.5,
            store_every: int = 1) -> TransientState:
        """Integrate to `t_end`, keeping every `store_every`-th frame."""
        n = int(round(t_end / dt))
        for k in range(n):
            self.step(dt, 1)
            if store_every > 1 and (k % store_every):
                self._frames.pop()
        return self.state

    # -- outputs ------------------------------------------------------------

    def temperatures_c(self) -> np.ndarray:
        """Volume-mean cell temperature [C], in cell_positions.csv order."""
        return self.state.T - KELVIN

    def surface_temperatures_c(self) -> np.ndarray:
        r"""Surface temperature [C] = volume mean + the radial rise.

        Bi > 0.1 here, so the lumped node is a volume mean. This is the value to
        compare against the 60 C SURFACE limit; conflating the two would
        under-report the peak by 0.9-4.5 K.
        """
        return self.temperatures_c() + self.th.radial_rise(
            float(np.mean(self.q)), self.model.geo.cell_height)

    def deviation_c(self) -> np.ndarray:
        """Each cell's departure from its own row's mean [K].

        Lets the scatter be seen on its own colour scale at true magnitude,
        instead of being swamped by the row-to-row gradient.
        """
        T = self.temperatures_c()
        out = np.zeros_like(T)
        for r in np.unique(self.rows):
            m = self.rows == r
            out[m] = T[m] - T[m].mean()
        return out

    def exhaust_rise(self) -> float:
        """Extra exhaust temperature [K] from the contactors and boards.

        They sit downstream of the bank, so this does not touch any cell.
        """
        if not self.include_electronics:
            return 0.0
        q = ELECTRONICS.total(self.model.elec.pack_current)
        mdot = self.result.mass_flow
        cp = AirProperties(self.result.outlet_temperature).specific_heat
        return q / (mdot * cp) if mdot > 0 else 0.0

    def history(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.array(self._hist_t), np.array(self._hist_max),
                np.array(self._hist_mean))

    def frame(self, k: int) -> np.ndarray:
        """Stored frame `k` as temperatures in C -- used for scrubbing."""
        k = max(0, min(k, len(self._frames) - 1))
        return self._frames[k] - KELVIN

    @property
    def n_frames(self) -> int:
        return len(self._frames)

    def energy_error(self) -> float:
        r"""Relative error in the running energy balance.

        $$\sum_i C_i\,\big(T_i(t)-T_i(0)\big)
          = \int_0^t\!\Big(\sum_i \dot q_i - \sum_i h_iA_i(T_i-T_{\rm air})\Big)dt'$$

        The removal term must be INTEGRATED, not evaluated once at the end and
        multiplied by $t$ -- the cells start at ambient and reject nothing, so
        the instantaneous flux at $t$ badly over-states what actually left.

        Conduction does not appear because $G_{ij}=G_{ji}$ makes it cancel
        exactly in the sum over nodes; that symmetry is enforced by construction
        in `_build_neighbours`, and it is the usual source of slow drift when it
        is not.
        """
        if not self._removed_int or self.state.t <= 0:
            return 0.0
        stored = float(self.C * (self.state.T - self._frames[0]).sum())
        gen = float(self.q.sum() * self.state.t)
        removed = float(self._removed_int[-1])
        denom = max(abs(gen), 1e-9)
        return abs(stored - (gen - removed)) / denom


def build(model: PackFlowModel | None = None,
          result: FlowResult | None = None,
          **kw) -> StochasticTransient:
    """Convenience constructor using the project's standard cell ordering."""
    import direction as D
    if model is None:
        model = PackFlowModel(geometry=D.geometry_for(D.Direction.FROM_12))
    if result is None:
        result = model.solve()
    cells = D.read_cells()
    order = [(int(c["row"]), int(c["col"]), int(c["bank"])) for c in cells]
    return StochasticTransient(model, result, order, **kw)


if __name__ == "__main__":
    tr = build()
    print(f"nodes {tr.n},  C = {tr.C:.1f} J/K,  tau = {tr.tau:.1f} s")
    print(f"stability dt_max = {tr.dt_max:.1f} s   (using dt = 0.5 s)")
    print(f"Biot = {tr.biot:.2f}  "
          f"{'-- lumped VIOLATED, node is a volume mean' if tr.biot > 0.1 else ''}")
    import time
    t0 = time.time()
    tr.run(t_end=400.0, dt=0.5)
    wall = time.time() - t0
    T = tr.temperatures_c()
    print(f"\nran 400 s in {wall*1e3:.0f} ms wall  "
          f"({wall/800*1e6:.1f} us/step)")
    print(f"  volume-mean  {T.min():.2f} .. {T.max():.2f} C")
    print(f"  surface      {tr.surface_temperatures_c().max():.2f} C")
    print(f"  deviation    {tr.deviation_c().min():+.3f} .. "
          f"{tr.deviation_c().max():+.3f} K")
    print(f"  exhaust rise from electronics  +{tr.exhaust_rise():.2f} K")

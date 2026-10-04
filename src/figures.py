r"""
The research-paper figure sheet, shaped for a side panel beside the 3D view.

Reuses `plots.py`'s panel functions rather than restyling anything: each of
them already takes an optional `ax` and draws in place when given one, so the
house palette, the limit lines, the operating-point marker and the two-space
unit labels all come along unchanged.  There is deliberately no second styling
system to keep in sync.

Updating has two tiers, because a full rebuild is ~120 ms and a slider emits
continuously while dragged:

    rebuild()        structure changed -- mode, fan count, direction   ~120 ms
    update_values()  only a scalar moved -- time step, ambient          ~30 ms

`update_values` mutates persistent Line2D artists via `set_ydata` and re-encodes
the same Figure, which is what keeps time-scrubbing responsive.
"""

from __future__ import annotations

import base64
import io
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import plots as P
from plots import INK, ACCENT, WARN, MUTED, GRID
from params import KELVIN

# Live sidebar renders at a lower dpi than the file export: at savefig.dpi 160
# a 7 x 9.8 in sheet is a ~1.4 MB PNG pushed over the websocket on every tick.
LIVE_DPI = 110
# While the transient is playing the sheet is re-encoded many times a second,
# and at 110 dpi that is a ~200 KB PNG per frame -- 3.3 MB/s over the websocket,
# which backs the transport up until it closes. During playback the sheet is
# encoded at this lower dpi instead (~55 KB), and returns to LIVE_DPI as soon
# as playback stops.
PLAY_DPI = 72
SIDEBAR_FIGSIZE = (7.0, 9.6)


def _suptitle(fig, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=0.985)
    fig.text(0.085, 0.958, subtitle, fontsize=7.8, color=MUTED)


def transient_panel(ax, tr) -> None:
    """Peak and mean cell temperature against time."""
    t, tmax, tmean = tr.history()
    ax.plot(t, tmax, color=WARN, lw=2.2, label="hottest cell")
    ax.plot(t, tmean, color=ACCENT, lw=1.8, label="pack mean")
    ax.axhline(60.0, color=WARN, ls="--", lw=1.2)
    ax.text(t[0] + 2, 60.6, "60 C limit", color=WARN, fontsize=8)
    ax.axhline(45.0, color=MUTED, ls=":", lw=1.2)
    ax.text(t[0] + 2, 45.6, "45 C design goal", color=MUTED, fontsize=8)
    if len(t) > 1:
        ax.plot([t[-1]], [tmax[-1]], "o", ms=8, color=INK, zorder=5)
        ax.annotate(f"t = {t[-1]:.0f} s\n{tmax[-1]:.2f} C",
                    xy=(t[-1], tmax[-1]), xytext=(-70, -28),
                    textcoords="offset points", fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.set_xlabel("time  [s]")
    ax.set_ylabel("temperature  [C]")
    ax.set_title("Transient response", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")


def scatter_panel(ax, tr) -> None:
    """Distribution of each cell's departure from its own row mean.

    Shown on its own scale because the scatter is 1-2% of the row-to-row
    gradient -- on the main colour map it is invisible, and inflating it would
    misrepresent the physics.
    """
    dev = tr.deviation_c()
    ax.hist(dev, bins=28, color=ACCENT, alpha=0.75, edgecolor="white", lw=0.5)
    ax.axvline(0.0, color=INK, ls=":", lw=1.2)
    ax.set_xlabel("departure from row mean  [K]")
    ax.set_ylabel("cells")
    ax.set_title(f"Cell-to-cell scatter  (sigma spread {dev.std():.3f} K)",
                 fontsize=10)


def placement_panel(ax, cands, best=None) -> None:
    """Peak temperature against fan power for every searched arrangement."""
    pw = [c.power for c in cands]
    tc = [c.t_cell for c in cands]
    ax.plot(pw, tc, "o", ms=5, color=MUTED, alpha=0.55, label="arrangements")

    import placement as PL
    front = PL.pareto(cands)
    ax.plot([c.power for c in front], [c.t_cell for c in front],
            "-s", color=ACCENT, lw=2.0, ms=6, label="Pareto front")
    if best is not None:
        ax.plot([best.power], [best.t_cell], "o", ms=9, color=INK, zorder=5)
        ax.annotate(f"{best.topology.name}\n{best.t_cell:.2f} C",
                    xy=(best.power, best.t_cell), xytext=(14, 16),
                    textcoords="offset points", fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.axhline(45.0, color=MUTED, ls=":", lw=1.2)
    ax.text(min(pw), 45.15, "45 C design goal", color=MUTED, fontsize=8)
    ax.set_xlabel("fan power  [W]")
    ax.set_ylabel("hottest cell  [C]")
    ax.set_title("Placement search", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")


def sidebar_figure(model, result, mode: str = "twosided",
                   transient=None, cands=None, best=None,
                   topology=None) -> plt.Figure:
    """A four-panel sheet in the `summary.png` idiom, shaped tall and narrow."""
    fig = plt.figure(figsize=SIDEBAR_FIGSIZE)
    gs = fig.add_gridspec(4, 1, hspace=0.52,
                          left=0.145, right=0.945, top=0.925, bottom=0.055)

    P.fan_curve_figure(model, fig.add_subplot(gs[0, 0]))
    P.thermal_march_figure(model, fig.add_subplot(gs[1, 0]))

    ax3 = fig.add_subplot(gs[2, 0])
    if mode == "placement" and cands:
        placement_panel(ax3, cands, best)
    else:
        P.velocity_sweep_figure(model, ax3)

    ax4 = fig.add_subplot(gs[3, 0])
    if mode == "transient" and transient is not None:
        transient_panel(ax4, transient)
    elif mode == "placement" and transient is None and cands:
        P.velocity_sweep_figure(model, ax4)
    else:
        P.velocity_sweep_figure(model, ax4)

    tc = result.cell_surface_temperature - KELVIN
    verdict = "PASS" if tc <= 60.0 else "FAIL"
    goal = "met" if tc <= 45.0 else "missed"
    name = topology.name if topology is not None else mode
    _suptitle(fig,
              f"{result.flow_cfm:.0f} CFM  |  hottest {tc:.1f} C  |  "
              f"60 C {verdict}  |  45 C {goal}",
              f"{name}  -  geometry measured from Main Battery.step, "
              f"{model.geo.n_cells} cells")
    return fig


def encode(fig, dpi: int = LIVE_DPI) -> str:
    """Figure -> data: URI an <img> can show."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class SidebarRenderer:
    """Keeps one Figure alive across updates.

    Rebuilding costs ~120 ms, which is fine on a mode switch but not on every
    frame of an animation; `update_transient` mutates the existing artists
    instead and only re-encodes.
    """

    def __init__(self) -> None:
        self.fig: plt.Figure | None = None
        self._lines: dict = {}

    def rebuild(self, model, result, **kw) -> str:
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = sidebar_figure(model, result, **kw)
        self._lines = {}
        tr = kw.get("transient")
        if tr is not None and len(self.fig.axes) >= 4:
            ax = self.fig.axes[-1]
            if len(ax.lines) >= 2:
                self._lines = {"max": ax.lines[0], "mean": ax.lines[1],
                               "ax": ax}
        return encode(self.fig)

    def update_transient(self, tr, dpi: int | None = None) -> str:
        """Fast path: rewrite the time series without rebuilding the sheet.

        `dpi` drops to PLAY_DPI during playback: re-encoding the full sheet at
        110 dpi costs ~48 ms and ~200 KB, which at playback rates saturates the
        websocket. The figure itself is unchanged -- only the raster is coarser.
        """
        if not self._lines or self.fig is None:
            raise RuntimeError("call rebuild() first")
        t, tmax, tmean = tr.history()
        self._lines["max"].set_data(t, tmax)
        self._lines["mean"].set_data(t, tmean)
        ax = self._lines["ax"]
        ax.relim()
        ax.autoscale_view()
        return encode(self.fig, dpi=dpi or LIVE_DPI)


if __name__ == "__main__":
    import time
    import direction as D
    from flow import PackFlowModel
    import transient as TRN

    m = PackFlowModel(geometry=D.geometry_for(D.Direction.FROM_12))
    r = m.solve()
    tr = TRN.build(m, r)
    tr.run(t_end=200.0, dt=0.5)

    sr = SidebarRenderer()
    t0 = time.time()
    uri = sr.rebuild(m, r, mode="transient", transient=tr)
    print(f"rebuild        {(time.time()-t0)*1e3:6.0f} ms   {len(uri)/1024:.0f} KB")
    t0 = time.time()
    for _ in range(5):
        tr.step(0.5, 20)
        uri = sr.update_transient(tr)
    print(f"update_values  {(time.time()-t0)/5*1e3:6.0f} ms   {len(uri)/1024:.0f} KB")

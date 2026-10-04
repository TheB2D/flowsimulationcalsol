r"""
Interactive battery-box thermal and airflow explorer.

    .venv/bin/python src/app.py              # opens in your browser
    .venv/bin/python src/app.py --source box # enclosure only, faster
    .venv/bin/python src/app.py --prebuild   # just warm the caches and exit

Three modes:

  1. PLACEMENT   exhaustively searches every buildable fan arrangement over the
                 two measured cutouts and shows the Pareto front of peak cell
                 temperature against fan power.
  2. TWO-SIDED   fans on the blue `Front Plate` and the green `Rear Plate` at
                 once.  Push-pull puts them in series so their pressure rises
                 add, which is what actually helps a bank that is 98% of the
                 system resistance.  Selecting "both in" is refused, with the
                 continuity argument, rather than quietly returning a number.
  3. TRANSIENT   the stochastic time march, with play/pause/scrub.

The 3D view, the research-paper figure sheet and the governing equations sit
side by side, and the panels, shelf and boards start translucent with a toggle
to make them solid.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
import time
import sys
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from trame.app import get_server, asynchronous
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as v3, html
from trame_pyvista.ui.vuetify3 import PyVistaRemoteView

import geometry as G
import packscene as PS
import network as N
import placement as PL
import figures as FIG
import equations as EQ
import fans as FANS
import camera as CAM
import transient as TRN
import direction as D
from flow import PackFlowModel
from params import (KELVIN, DELTA_FAN, SAN_ACE_FAN, OperatingConditions,
                    PackElectrical, CELL_THERMAL, ELECTRONICS)
from openings import FRONT_PLATE, REAR_PLATE, summary as openings_summary

server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller


def view_update() -> None:
    """Refresh the 3D view if the UI has been built.

    `ctrl.view_update` is only bound once `PyVistaRemoteView` exists, so calling
    it before `build_ui` -- which the headless smoke test does -- would raise.
    Everything in this module goes through here instead.
    """
    fn = getattr(ctrl, "view_update", None)
    if fn is None:
        return
    try:
        fn()
    except Exception:
        pass


# =============================================================================
# Model state, held outside trame state (these are not JSON-serialisable)
# =============================================================================

class App:
    def __init__(self, source: str = "full", tol: float = 0.5) -> None:
        self.source = source
        self.tol = tol
        self.data = None
        self.handles: PS.SceneHandles | None = None
        self.model: PackFlowModel | None = None
        self.result = None
        self.topology: N.Topology | None = None
        self.transient: TRN.StochasticTransient | None = None
        self.cands: list = []
        self.best = None
        self.sidebar = FIG.SidebarRenderer()
        self.panel_centres: dict = {}
        self.cell_order: list = []

    # -- geometry -------------------------------------------------------
    def load_geometry(self) -> None:
        self.data = G.load_or_build(self.source, tol=self.tol, verbose=True)
        self.handles = PS.build_scene(self.data, off_screen=True,
                                      decimate=PS.DECIMATE)
        self.panel_centres = FANS.panel_centres_from_bodies(self.data["bodies"])
        PS.set_opaque_mode(self.handles, False)

    # -- physics --------------------------------------------------------
    def build_model(self) -> PackFlowModel:
        cond = OperatingConditions(
            inlet_temperature=float(state.ambient_c) + KELVIN)
        geo = D.geometry_for(D.Direction.FROM_12 if state.direction == "from_12"
                             else D.Direction.FROM_17)
        fan = DELTA_FAN if state.fan_model == "delta" else SAN_ACE_FAN
        return PackFlowModel(geometry=geo, fan=fan, conditions=cond,
                             heat_load=float(state.heat_load))

    def current_topology(self) -> N.Topology:
        fan = DELTA_FAN if state.fan_model == "delta" else SAN_ACE_FAN
        nf, nr = int(state.n_front), int(state.n_rear)
        sf = +1 if state.sense_front == "in" else -1
        sr = +1 if state.sense_rear == "in" else -1
        nf = min(nf, FRONT_PLATE.max_fans)
        nr = min(nr, REAR_PLATE.max_fans)
        bits = []
        if nf:
            bits.append(f"{nf} blue {state.sense_front}")
        if nr:
            bits.append(f"{nr} green {state.sense_rear}")
        return N.Topology(
            name=" + ".join(bits) or "no fans",
            sets=[N.FanSet(fan, nf, FRONT_PLATE, sf),
                  N.FanSet(fan, nr, REAR_PLATE, sr)])


APP = App()


# =============================================================================
# Recomputation
# =============================================================================

def recompute(rebuild_sidebar: bool = True) -> None:
    """Tier C: re-solve the physics and refresh everything that depends on it."""
    APP.model = APP.model or APP.build_model()
    APP.model = APP.build_model()
    mode = state.mode

    try:
        if mode == "placement":
            APP.cands = PL.search(APP.model)
            APP.best = APP.cands[0] if APP.cands else None
            APP.topology = APP.best.topology if APP.best else None
            APP.result = APP.best.result if APP.best else APP.model.solve()
            state.infeasible = ""
        else:
            APP.topology = APP.current_topology()
            APP.result = N.solve_topology(APP.topology, APP.model)
            state.infeasible = ""
    except N.InfeasibleTopology as exc:
        state.infeasible = str(exc)
        return
    except ValueError as exc:
        state.infeasible = str(exc)
        return

    # transient rides on the steady solution
    if mode == "transient":
        cells = D.read_cells()
        order = [(int(c["row"]), int(c["col"]), int(c["bank"])) for c in cells]
        APP.transient = TRN.StochasticTransient(
            APP.model, APP.result, order,
            seed=int(state.seed),
            scatter_gain=float(state.scatter_gain),
            noise_sigma=float(state.noise_sigma),
            include_electronics=bool(state.include_electronics))
        state.t_now = 0.0
        state.n_frames = APP.transient.n_frames

    push_field()
    draw_fans()
    update_metrics()
    if rebuild_sidebar:
        refresh_sidebar(rebuild=True)
    view_update()


def push_field() -> None:
    """Tier B: write the current temperature field into the scene."""
    h = APP.handles
    if h is None or h.cell_mesh is None:
        return
    if state.mode == "transient" and APP.transient is not None:
        if state.show_deviation:
            T = APP.transient.deviation_c()
            PS.set_clim(h, (-0.6, 0.6))
        else:
            T = (APP.transient.surface_temperatures_c()
                 if state.show_surface else APP.transient.temperatures_c())
            PS.set_clim(h, tuple(state.clim) if state.lock_clim
                        else (float(T.min()), float(T.max())))
    else:
        T = D.cell_temperatures(
            D.Direction.FROM_12 if state.direction == "from_12"
            else D.Direction.FROM_17, APP.result)
        PS.set_clim(h, tuple(state.clim) if state.lock_clim
                    else (float(T.min()), float(T.max())))
    if T.size == h.n_cells:
        PS.set_cell_temperatures(h, T)


def draw_fans() -> None:
    h = APP.handles
    if h is None or APP.topology is None:
        return
    PS.clear_fans(h)
    if state.show_fans:
        # No panel_centres override: the fans go on each opening's MEASURED
        # cutout centre. Passing the panel centroid here put them ~80 mm high.
        h.fan_actors = FANS.build_fan_actors(
            h.plotter, APP.topology, None,
            show_arrows=bool(state.show_arrows))


def update_metrics() -> None:
    r = APP.result
    if r is None:
        return
    tc = r.cell_surface_temperature - KELVIN
    state.m_cfm = f"{r.flow_cfm:.0f} CFM"
    state.m_tmax = f"{tc:.1f} C"
    state.m_dp = f"{r.pressure_drop_total:.0f} Pa"
    state.m_h = f"h {r.h_conv:.0f}"
    state.m_re = f"Re {r.reynolds_max:.0f}"
    state.m_verdict = "PASS" if tc <= 60.0 else "FAIL"
    state.m_goal = "45 C met" if tc <= 45.0 else "45 C missed"
    state.m_power = f"{APP.topology.total_power:.0f} W" if APP.topology else "-"
    if APP.transient is not None and state.mode == "transient":
        state.m_bi = f"Bi {APP.transient.biot:.2f}"
        state.m_tau = f"tau {APP.transient.tau:.0f} s"
    else:
        state.m_bi = ""
        state.m_tau = ""


def refresh_sidebar(rebuild: bool = False, play: bool = False) -> None:
    """Re-encode the figure sheet.

    `play=True` uses the coarser playback dpi, which roughly halves the payload.
    The encode cost is dominated by matplotlib's draw rather than the raster, so
    dpi alone is not enough -- the throttle in `animate` is what actually keeps
    the websocket healthy.
    """
    if APP.model is None or APP.result is None:
        return
    kw = dict(mode=state.mode, topology=APP.topology)
    if state.mode == "transient":
        kw["transient"] = APP.transient
    if state.mode == "placement":
        kw["cands"] = APP.cands
        kw["best"] = APP.best
    try:
        if rebuild or APP.sidebar.fig is None:
            state.sidebar_src = APP.sidebar.rebuild(APP.model, APP.result, **kw)
        elif state.mode == "transient" and APP.transient is not None:
            state.sidebar_src = APP.sidebar.update_transient(
                APP.transient, dpi=FIG.PLAY_DPI if play else None)
    except Exception:
        state.sidebar_src = APP.sidebar.rebuild(APP.model, APP.result, **kw)


# =============================================================================
# State
# =============================================================================

state.update({
    "mode": "twosided",
    "ambient_c": 40.0,
    "heat_load": 295.69,
    "direction": "from_12",
    "fan_model": "delta",
    "n_front": 3, "n_rear": 3,
    "sense_front": "in", "sense_rear": "out",
    "opaque_mode": False,
    "show_cells": True, "show_case": True, "show_shelf": True,
    "show_pcb": True, "show_busbar": True, "show_structure": True,
    "show_fans": True, "show_arrows": True,
    "lock_clim": True, "clim": [40.0, 50.0],
    "playing": False, "t_now": 0.0, "dt": 0.5, "speed": 10,
    "t_end": 600.0,
    "seed": 0, "scatter_gain": 1.0, "noise_sigma": 0.0,
    "show_deviation": False, "show_surface": False,
    "include_electronics": True,
    "n_frames": 1,
    "sidebar_src": "", "infeasible": "",
    "m_cfm": "", "m_tmax": "", "m_dp": "", "m_h": "", "m_re": "",
    "m_verdict": "", "m_goal": "", "m_power": "", "m_bi": "", "m_tau": "",
    "equations": EQ.catalogue("Flow"), "eq_group": "Flow",
    "busy": False, "status": "",
    "cam_mode": "orbit",          # orbit | pan -- what the arrow keys do
    "cam_key": "",                # last key pressed, pushed from the browser
    "key_help": CAM.KEY_HELP,
    "open_panels": ["Physics", "Fans"],
})


@state.change("mode", "ambient_c", "heat_load", "direction", "fan_model",
              "n_front", "n_rear", "sense_front", "sense_rear",
              "seed", "scatter_gain", "noise_sigma", "include_electronics")
def _on_physics(**_):
    recompute()


@state.change("opaque_mode")
def _on_opaque(opaque_mode, **_):
    if APP.handles:
        PS.set_opaque_mode(APP.handles, bool(opaque_mode))
        view_update()


@state.change("show_cells", "show_case", "show_shelf", "show_pcb",
              "show_busbar", "show_structure")
def _on_visibility(**_):
    h = APP.handles
    if not h:
        return
    for cls, flag in (("cell", state.show_cells), ("case_plate", state.show_case),
                      ("shelf", state.show_shelf), ("pcb", state.show_pcb),
                      ("busbar", state.show_busbar),
                      ("structure", state.show_structure)):
        PS.set_class_visible(h, cls, bool(flag))
    view_update()


@state.change("show_fans", "show_arrows")
def _on_fans(**_):
    draw_fans()
    view_update()


@state.change("lock_clim", "show_deviation", "show_surface")
def _on_scale(**_):
    push_field()
    view_update()


@state.change("eq_group")
def _on_eq_group(eq_group, **_):
    state.equations = EQ.catalogue(eq_group)


# =============================================================================
# Time controls
# =============================================================================

# =============================================================================
# Camera
# =============================================================================

# Camera moves are cheap (~4 ms) but each one pushes a frame. While the
# transient is playing the animation loop is already pushing frames, so an
# unthrottled key-repeat (holding an arrow down) doubles the stream and is
# exactly what tipped the websocket over during panning.
_last_cam_push = 0.0
CAM_MIN_INTERVAL = 0.033        # s, ~30 fps ceiling on camera-driven frames


@state.change("cam_key")
def _on_cam_key(cam_key, **_):
    """Apply a keypress forwarded from the browser.

    The value carries a `#counter` suffix so that pressing the same key twice
    in a row still registers as a state change; trame would otherwise ignore
    the second press because the value did not differ.

    The camera always moves; only the PUSH is rate-limited, so held keys stay
    smooth without flooding the socket.
    """
    global _last_cam_push
    if not cam_key or APP.handles is None:
        return
    raw = cam_key.split("#", 1)[0]
    shift = raw.startswith("shift+")
    key = raw[6:] if shift else raw
    if not CAM.handle_key(APP.handles.plotter, key, mode=state.cam_mode,
                          shift=shift):
        return
    now = time.monotonic()
    if now - _last_cam_push >= CAM_MIN_INTERVAL or not state.playing:
        _last_cam_push = now
        view_update()


@ctrl.set("set_view")
def set_view(name: str = "iso"):
    if APP.handles is None:
        return
    fn = CAM.NAMED_VIEWS.get(name)
    if fn:
        fn(APP.handles.plotter)
        view_update()


@ctrl.set("cam_nudge")
def cam_nudge(what: str):
    """Button equivalents of the keyboard controls."""
    if APP.handles is None:
        return
    CAM.handle_key(APP.handles.plotter, what, mode=state.cam_mode)
    view_update()


@state.change("cam_mode")
def _on_cam_mode(cam_mode, **_):
    """Switch the MOUSE style to match the arrow-key mode.

    Keeping the two in sync matters: a user who picks PAN expects dragging to
    pan as well, not just the arrows.
    """
    if APP.handles is None:
        return
    p = APP.handles.plotter
    try:
        if cam_mode == "pan":
            # Left-drag pans; everything else stays conventional.
            p.enable_custom_trackball_style(left="pan", middle="pan",
                                            right="dolly")
        else:
            p.enable_trackball_style()
    except Exception:
        pass
    view_update()


@ctrl.set("play_pause")
def play_pause():
    state.playing = not state.playing
    if state.playing:
        asynchronous.create_task(animate())


@ctrl.set("step_once")
def step_once():
    if APP.transient is None:
        return
    APP.transient.step(float(state.dt), int(state.speed))
    state.t_now = APP.transient.state.t
    state.n_frames = APP.transient.n_frames
    push_field()
    update_metrics()
    refresh_sidebar()
    view_update()


@ctrl.set("reset_time")
def reset_time():
    state.playing = False
    if APP.transient is not None:
        APP.transient.reset(int(state.seed))
        state.t_now = 0.0
        state.n_frames = APP.transient.n_frames
        push_field()
        refresh_sidebar(rebuild=True)
        view_update()


# The sidebar sheet costs ~50 ms to re-encode and ~200 KB to push. Re-doing it
# on every animation frame is what saturated the websocket and closed the
# transport (1392 `Cannot write to closing transport` errors in one session).
# It is throttled to this interval instead, independent of the 3D frame rate --
# a time-series plot simply does not need 16 updates a second.
SIDEBAR_MIN_INTERVAL = 0.5      # s
FRAME_INTERVAL = 0.05           # s, target for the 3D field update


async def animate():
    """Drive the march, keeping the websocket inside its budget.

    The cost split is lopsided and the loop is shaped around it:

        physics step        0.01 ms
        push_field          0.18 ms   -> every frame
        update_metrics      0.01 ms   -> every frame
        refresh_sidebar    ~50 ms, ~200 KB  -> THROTTLED

    Pushing the sidebar every frame meant ~3.3 MB/s of PNG on top of the 3D
    view's own JPEG stream; the browser could not drain it, the send queue
    backed up and the connection died -- which looks to the user like the app
    freezing, especially while panning (interaction frames add to the same
    queue).

    Three things this has to get right, each of which failed once:

    * Use `asyncio.sleep`. There is NO `sleep` in `trame.app.asynchronous`, so
      `await asynchronous.sleep(...)` raises `AttributeError` on the first
      iteration -- inside a background task, where nothing surfaces it. The
      symptom is Play doing nothing while Step still works.
    * Do the render INSIDE the `with state:` block, so the flushed state and the
      new 3D frame arrive together.
    * Yield to the event loop every frame, so camera interaction stays
      responsive while playing.

    Any exception is logged rather than swallowed, and `state.playing` is always
    cleared, so the button can never be left stuck.
    """
    last_sidebar = -1e9
    try:
        while state.playing:
            if APP.transient is None:
                break
            t_frame = time.monotonic()

            APP.transient.step(float(state.dt), max(1, int(state.speed)))

            due = (time.monotonic() - last_sidebar) >= SIDEBAR_MIN_INTERVAL
            with state:
                state.t_now = APP.transient.state.t
                state.n_frames = APP.transient.n_frames
                push_field()
                update_metrics()
                if due:
                    refresh_sidebar(play=True)
                    last_sidebar = time.monotonic()
                view_update()

            if state.t_now >= float(state.t_end):
                state.playing = False
                break

            # Sleep the remainder of the frame budget, never less than a tick:
            # a bare `continue` would starve the event loop and freeze the UI.
            spent = time.monotonic() - t_frame
            await asyncio.sleep(max(0.01, FRAME_INTERVAL - spent))
    except Exception:
        traceback.print_exc()
    finally:
        with state:
            state.playing = False
            # Leave a crisp, fully up-to-date sheet behind.
            refresh_sidebar()
            view_update()


# =============================================================================
# UI
# =============================================================================

def build_ui():
    with SinglePageWithDrawerLayout(server, full_height=True) as layout:
        layout.title.set_text("CalSol battery box  -  airflow and thermal")

        with layout.toolbar:
            v3.VSpacer()
            # Use the context-manager form, not `children=[...]`: constructing
            # a widget already registers it with the open parent, so passing the
            # same instances via `children` renders every button TWICE.
            with v3.VBtnToggle(v_model=("mode", "twosided"), mandatory=True,
                               density="compact", variant="outlined",
                               divided=True):
                v3.VBtn("Placement", value="placement", size="small")
                v3.VBtn("Two-sided", value="twosided", size="small")
                v3.VBtn("Transient", value="transient", size="small")
            v3.VSpacer()
            for key, colour in (("m_cfm", "primary"), ("m_tmax", "error"),
                                ("m_power", "grey"), ("m_verdict", "success"),
                                ("m_goal", "grey")):
                v3.VChip(f"{{{{ {key} }}}}", size="small", class_="mx-1",
                         color=colour, variant="flat",
                         v_show=f"{key} && {key}.length")
            v3.VProgressLinear(indeterminate=True, v_show="busy",
                               absolute=True, location="bottom")

        with layout.drawer as drawer:
            drawer.width = 350
            # NOTE: do NOT pass `model_value=[0, 1, 2]` here. trame tries
            # `.startswith()` on each entry, raises on the ints, and silently
            # renders the whole subtree as `<VExpansionPanels html-error />` --
            # i.e. an empty drawer with no error shown to the user. Open the
            # panels via the string state variable below instead.
            with v3.VExpansionPanels(v_model=("open_panels", ["Physics", "Fans"]),
                                     multiple=True, variant="accordion"):
                # -- physics -------------------------------------------
                with v3.VExpansionPanel(title="Physics", value="Physics"):
                    with v3.VExpansionPanelText():
                        v3.VSlider(v_model=("heat_load", 295.69),
                                   min=100, max=700, step=5,
                                   label="pack heat  [W]", thumb_label=True,
                                   density="compact", hide_details=True)
                        v3.VSlider(v_model=("ambient_c", 40.0),
                                   min=15, max=55, step=1,
                                   label="inlet air  [C]", thumb_label=True,
                                   density="compact", hide_details=True)
                        v3.VSelect(v_model=("direction", "from_12"),
                                   items=("[{'title':'across the 17 rows (from the 12)','value':'from_12'},"
                                          "{'title':'across the 12 cols (from the 17)','value':'from_17'}]",),
                                   label="flow axis", density="compact",
                                   hide_details=True, class_="mt-3")
                        v3.VSelect(v_model=("fan_model", "delta"),
                                   items=("[{'title':'Delta GFB0812ES-E (in the CAD)','value':'delta'},"
                                          "{'title':'San Ace 80 (the written spec)','value':'sanace'}]",),
                                   label="fan", density="compact",
                                   hide_details=True, class_="mt-3")
                        v3.VSwitch(v_model=("include_electronics", True),
                                   label="contactor + PCB heat (ASSUMED)",
                                   density="compact", hide_details=True,
                                   color="primary")

                # -- fans ----------------------------------------------
                with v3.VExpansionPanel(title="Fans", value="Fans"):
                    with v3.VExpansionPanelText():
                        html.Div("blue  Front Plate  (max 3)",
                                 classes="text-caption text-medium-emphasis")
                        v3.VSlider(v_model=("n_front", 3), min=0, max=3, step=1,
                                   thumb_label="always", density="compact",
                                   hide_details=True, color="blue")
                        with v3.VBtnToggle(v_model=("sense_front", "in"),
                                           mandatory=True, density="compact",
                                           variant="outlined", class_="mb-3"):
                            v3.VBtn("blow in", value="in", size="x-small")
                            v3.VBtn("draw out", value="out", size="x-small")
                        html.Div("green  Rear Plate  (max 3)",
                                 classes="text-caption text-medium-emphasis")
                        v3.VSlider(v_model=("n_rear", 3), min=0, max=3, step=1,
                                   thumb_label="always", density="compact",
                                   hide_details=True, color="green")
                        with v3.VBtnToggle(v_model=("sense_rear", "out"),
                                           mandatory=True, density="compact",
                                           variant="outlined"):
                            v3.VBtn("blow in", value="in", size="x-small")
                            v3.VBtn("draw out", value="out", size="x-small")
                        v3.VAlert(text=("infeasible",), type="error",
                                  density="compact", class_="mt-3",
                                  v_show="infeasible && infeasible.length")

                # -- camera --------------------------------------------
                with v3.VExpansionPanel(title="Camera / view", value="Camera"):
                    with v3.VExpansionPanelText():
                        html.Div("arrow keys", classes="text-caption text-medium-emphasis")
                        with v3.VBtnToggle(v_model=("cam_mode", "orbit"),
                                           mandatory=True, density="compact",
                                           variant="outlined", class_="mb-3"):
                            v3.VBtn("orbit", value="orbit", size="x-small")
                            v3.VBtn("pan", value="pan", size="x-small")

                        html.Div("nudge", classes="text-caption text-medium-emphasis")
                        with v3.VRow(classes="ma-0 pa-0 justify-center"):
                            v3.VBtn(icon="mdi-chevron-up", size="x-small",
                                    variant="text",
                                    click=(ctrl.cam_nudge, "['arrowup']"))
                        with v3.VRow(classes="ma-0 pa-0 justify-center align-center"):
                            v3.VBtn(icon="mdi-chevron-left", size="x-small",
                                    variant="text",
                                    click=(ctrl.cam_nudge, "['arrowleft']"))
                            v3.VBtn(icon="mdi-restore", size="x-small",
                                    variant="text",
                                    click=(ctrl.set_view, "['iso']"))
                            v3.VBtn(icon="mdi-chevron-right", size="x-small",
                                    variant="text",
                                    click=(ctrl.cam_nudge, "['arrowright']"))
                        with v3.VRow(classes="ma-0 pa-0 justify-center"):
                            v3.VBtn(icon="mdi-chevron-down", size="x-small",
                                    variant="text",
                                    click=(ctrl.cam_nudge, "['arrowdown']"))

                        with v3.VRow(classes="ma-0 pa-0 mt-2 justify-center"):
                            v3.VBtn("roll -", size="x-small", variant="text",
                                    click=(ctrl.cam_nudge, "['q']"))
                            v3.VBtn("-", size="x-small", variant="text",
                                    click=(ctrl.cam_nudge, "['-']"))
                            v3.VBtn("+", size="x-small", variant="text",
                                    click=(ctrl.cam_nudge, "['+']"))
                            v3.VBtn("roll +", size="x-small", variant="text",
                                    click=(ctrl.cam_nudge, "['e']"))

                        html.Div("views", classes="text-caption text-medium-emphasis mt-3")
                        with v3.VRow(classes="ma-0 pa-0"):
                            for _v, _lbl in (("iso", "iso"), ("flow", "flow"),
                                             ("inlet", "inlet"), ("exhaust", "exhaust"),
                                             ("top", "top"), ("side", "side")):
                                v3.VBtn(_lbl, size="x-small", variant="outlined",
                                        class_="ma-1",
                                        click=(ctrl.set_view, f"['{_v}']"))

                        v3.VDivider(classes="my-3")
                        with v3.VTable(density="compact"):
                            with html.Tbody():
                                with html.Tr(v_for="(h, i) in key_help", key="i"):
                                    html.Td("{{ h[0] }}",
                                            classes="text-caption font-weight-medium")
                                    html.Td("{{ h[1] }}",
                                            classes="text-caption text-medium-emphasis")

                # -- appearance ----------------------------------------
                with v3.VExpansionPanel(title="Appearance", value="Appearance"):
                    with v3.VExpansionPanelText():
                        v3.VSwitch(v_model=("opaque_mode", False),
                                   label="make panels + shelf opaque",
                                   color="primary", density="compact",
                                   hide_details=True)
                        for lbl, key in (("cells", "show_cells"),
                                         ("enclosure panels", "show_case"),
                                         ("electronics shelf", "show_shelf"),
                                         ("PCBs", "show_pcb"),
                                         ("busbars", "show_busbar"),
                                         ("structure", "show_structure"),
                                         ("fans", "show_fans"),
                                         ("flow arrows", "show_arrows")):
                            v3.VSwitch(v_model=(key, True), label=lbl,
                                       density="compact", hide_details=True,
                                       color="primary")
                        v3.VSwitch(v_model=("lock_clim", True),
                                   label="lock colour scale (comparable modes)",
                                   density="compact", hide_details=True,
                                   color="primary")

                # -- time ----------------------------------------------
                with v3.VExpansionPanel(title="Time  (transient mode)", value="Time"):
                    with v3.VExpansionPanelText():
                        with v3.VRow(classes="pa-0 ma-0"):
                            v3.VBtn("play / pause", click=ctrl.play_pause,
                                    size="small", color="primary",
                                    variant="flat", class_="mr-2")
                            v3.VBtn("step", click=ctrl.step_once,
                                    size="small", variant="outlined",
                                    class_="mr-2")
                            v3.VBtn("reset", click=ctrl.reset_time,
                                    size="small", variant="outlined")
                        html.Div("t = {{ t_now.toFixed(0) }} s",
                                 classes="text-caption mt-3")
                        v3.VSlider(v_model=("speed", 10), min=1, max=60, step=1,
                                   label="steps / frame", thumb_label=True,
                                   density="compact", hide_details=True)
                        v3.VSlider(v_model=("t_end", 600.0), min=60, max=1800,
                                   step=30, label="run until  [s]",
                                   thumb_label=True, density="compact",
                                   hide_details=True)
                        v3.VTextField(v_model=("seed", 0), label="random seed",
                                      type="number", density="compact",
                                      hide_details=True, class_="mt-2")
                        v3.VSlider(v_model=("scatter_gain", 1.0),
                                   min=0.0, max=8.0, step=0.5,
                                   label="scatter gain (>1 NOT physical)",
                                   thumb_label=True, density="compact",
                                   hide_details=True)
                        v3.VSlider(v_model=("noise_sigma", 0.0),
                                   min=0.0, max=0.3, step=0.01,
                                   label="OU noise  [W]  (visual only)",
                                   thumb_label=True, density="compact",
                                   hide_details=True)
                        v3.VSwitch(v_model=("show_deviation", False),
                                   label="colour by departure from row mean",
                                   density="compact", hide_details=True,
                                   color="primary")
                        v3.VSwitch(v_model=("show_surface", False),
                                   label="surface T (adds radial rise)",
                                   density="compact", hide_details=True,
                                   color="primary")

                # -- equations -----------------------------------------
                with v3.VExpansionPanel(title="Equations", value="Equations"):
                    with v3.VExpansionPanelText():
                        with v3.VBtnToggle(v_model=("eq_group", "Flow"),
                                           mandatory=True, density="compact",
                                           variant="outlined", class_="mb-2"):
                            for _g in EQ.GROUPS:
                                v3.VBtn(_g, value=_g, size="x-small")
                        with v3.VCard(variant="flat", v_for="(e, i) in equations",
                                      key="i", class_="mb-3"):
                            html.Div("{{ e.caption }}",
                                     classes="text-caption font-weight-medium")
                            html.Img(src=("e.uri",),
                                     style="max-width:100%; margin:4px 0;")
                            html.Div("{{ e.source }}",
                                     classes="text-caption text-medium-emphasis")

        with layout.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                with v3.VRow(classes="pa-0 ma-0 fill-height", no_gutters=True):
                    with v3.VCol(cols="8", classes="pa-0 fill-height"):
                        # The div is focusable (tabindex) so it can receive key
                        # events, and grabs focus on mouseenter so the arrows
                        # work as soon as the pointer is over the view -- no
                        # click-to-focus step. `@keydown.prevent` stops the
                        # arrows from scrolling the page.
                        #
                        # The "#" + counter suffix matters: trame skips a state
                        # change when the value is unchanged, so without it a
                        # second press of the same key would be ignored.
                        with html.Div(
                                tabindex="0",
                                style="outline:none; width:100%; height:100%;",
                                mouseenter="$event.currentTarget.focus()",
                                # Raw JS string, NOT the `(expr,)` tuple form:
                                # a tuple makes trame wrap the expression in
                                # `trigger(...)`, which fires a trigger NAMED by
                                # the expression instead of assigning to
                                # `cam_key` -- so no key would ever reach the
                                # server. A bare string is emitted verbatim.
                                keydown=("$event.preventDefault(); "
                                         "cam_key = (($event.shiftKey ? 'shift+' : '') "
                                         "+ $event.key) + '#' + Date.now()"),
                                # preventDefault is called inside the handler
                                # rather than via a `.prevent` modifier: trame
                                # renders `keydown_prevent` as the hyphenated
                                # `@keydown-prevent`, which Vue does not treat
                                # as a modifier, so the arrows would still
                                # scroll the page.
                                __events=["keydown"]):
                            # Server-side rendering: with ~6 M triangles a local
                            # view would serialise the whole scene to the
                            # browser. Remote keeps the payload independent of
                            # mesh size and keeps VTK state in Python, where the
                            # fast path lives.
                            view = PyVistaRemoteView(APP.handles.plotter,
                                                     interactive_ratio=1)
                            ctrl.view_update = view.update
                            ctrl.view_reset_camera = view.reset_camera
                    with v3.VCol(cols="4", classes="pa-1",
                                 style="overflow-y:auto; max-height:100vh;"):
                        html.Img(src=("sidebar_src",),
                                 style="width:100%; display:block;")
        return layout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="full",
                    choices=["full", "box", "legacy", "pack"],
                    help="which STEP to show (default: full)")
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--prebuild", action="store_true",
                    help="warm the geometry caches and exit")
    args = ap.parse_args()

    if args.prebuild:
        for key in ("box", "full"):
            G.load_or_build(key, tol=args.tol, verbose=True)
        print("caches warm")
        return

    APP.source, APP.tol = args.source, args.tol
    print(openings_summary())
    print(f"\nloading {args.source} geometry ...")
    APP.load_geometry()
    print(PS.scene_stats(APP.data))

    state.equations = EQ.catalogue("Flow")
    recompute()
    build_ui()
    server.start(port=args.port)


if __name__ == "__main__":
    main()

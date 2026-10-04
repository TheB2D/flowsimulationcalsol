"""
Guard against the UI silently rendering nothing.

trame catches exceptions raised while building a widget and emits
`<Widget html-error />` in place of the whole subtree.  Nothing is printed to
the browser and the server still starts, so a blank control panel looks exactly
like a working app from the server side.  That is how an empty drawer shipped
once already.

Two concrete traps this locks down:

  * `VExpansionPanels(model_value=[0, 1, 2])` -- trame calls `.startswith()` on
    each entry and dies on the ints, discarding every panel.
  * `Container(children=[...])` -- constructing a widget already registers it
    with the open parent, so passing the same instances via `children` renders
    each one TWICE.

    python src/test_ui.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Every control the drawer is supposed to expose, by its state binding.
REQUIRED_BINDINGS = [
    "heat_load", "ambient_c", "direction", "fan_model", "include_electronics",
    "n_front", "n_rear", "sense_front", "sense_rear",
    "opaque_mode", "show_cells", "show_case", "show_shelf", "show_pcb",
    "show_fans", "show_arrows", "lock_clim",
    "speed", "t_end", "seed", "scatter_gain", "noise_sigma",
    "show_deviation", "show_surface",
    "mode", "eq_group", "open_panels",
    "cam_mode",
]

REQUIRED_MARKUP = [
    "VNavigationDrawer", "VExpansionPanels", "vtk-remote-view", "sidebar_src",
    'title="Physics"', 'title="Fans"', 'title="Appearance"',
    'title="Time', 'title="Equations"', 'title="Camera / view"',
    # the keyboard path: a focusable div that forwards keydown to `cam_key`
    'tabindex="0"', "key_help",
    # the keyboard path: the handler must ASSIGN cam_key (not be wrapped in
    # trigger(), which fires a trigger named by the expression instead) and
    # must call preventDefault so the arrows do not scroll the page.
    "cam_key =", "preventDefault",
    # NOTE: assert on the RENDERED form. trame rewrites `click=(ctrl.fn, ...)`
    # into `trigger('trigger__N', [...])`, so grepping for the function name
    # would spuriously fail even though the button is there.
    "trigger(",
]


def main() -> int:
    import app as A

    A.APP.source = "box"          # structure is what matters; load the small file
    A.APP.load_geometry()
    A.recompute()
    html = A.build_ui().html

    failures: list[str] = []

    # 1. the silent-failure marker
    n_err = html.count("html-error")
    if n_err:
        failures.append(f"{n_err} widget(s) rendered as html-error "
                        f"-- a subtree was swallowed")

    # 2. every control present
    for key in REQUIRED_BINDINGS:
        if f'v-model="{key}"' not in html and f'"{key}"' not in html:
            failures.append(f"missing binding: {key}")

    for needle in REQUIRED_MARKUP:
        if needle not in html:
            failures.append(f"missing markup: {needle}")

    # 3. nothing duplicated -- three mode buttons, not six
    n_mode = sum(html.count(f'value="{v}"')
                 for v in ("placement", "twosided", "transient"))
    if n_mode != 3:
        failures.append(f"mode toggle has {n_mode} buttons, expected 3 "
                        f"(children=[...] duplicates widgets)")

    # 4. the animation loop must actually advance -- `animate()` once died on
    #    its first iteration (no `sleep` in trame.app.asynchronous), which
    #    looks exactly like Play doing nothing while Step still works.
    import asyncio
    import time

    A.state.mode = "transient"
    A.recompute()
    t0 = float(A.state.t_now)

    async def _drive():
        A.state.playing = True
        task = asyncio.ensure_future(A.animate())
        await asyncio.sleep(0.6)
        A.state.playing = False
        await asyncio.sleep(0.2)
        return task

    asyncio.new_event_loop().run_until_complete(_drive())
    advanced = float(A.state.t_now) - t0
    if advanced <= 0:
        failures.append("animate() did not advance time -- Play is dead")
    else:
        print(f"animate(): advanced {advanced:.0f} s of simulated time "
              f"in 0.6 s of wall clock")

    # 5. the camera controls must actually move the camera. Calling the state
    #    handler directly, because change listeners only fire inside a running
    #    server -- a test-harness limit, not a code path difference.
    import numpy as np

    cam = A.APP.handles.plotter.camera
    before = np.array(cam.position, dtype=float)
    A._on_cam_key("arrowleft#1")
    if np.allclose(before, np.array(cam.position, dtype=float)):
        failures.append("arrow key did not orbit the camera")

    A.state.cam_mode = "pan"
    foc = np.array(cam.focal_point, dtype=float)
    A._on_cam_key("arrowleft#2")
    if np.allclose(foc, np.array(cam.focal_point, dtype=float)):
        failures.append("pan mode did not translate the focal point")
    A.state.cam_mode = "orbit"

    foc2 = np.array(cam.focal_point, dtype=float)
    A._on_cam_key("arrowright#3")
    if not np.allclose(foc2, np.array(cam.focal_point, dtype=float)):
        failures.append("orbit mode moved the focal point (it should not)")

    for _name in A.CAM.NAMED_VIEWS:
        A.ctrl.set_view(_name)
    print(f"camera: orbit, pan and {len(A.CAM.NAMED_VIEWS)} named views all work")

    # 6. fans must sit INSIDE their cutouts. They were once placed on the
    #    panel CENTROID, which is 75-85 mm above the hole because the plate has
    #    more material above the cutout than below -- so every fan floated clear
    #    of the opening it was supposed to be mounted in.
    import fans as _FANS
    import openings as _OP

    for _o in (_OP.FRONT_PLATE, _OP.REAR_PLATE, _OP.FRONT_DUCT):
        _cx, _cy, _ = _o.centre_mm
        _hw, _hh = _o.width * 1e3 / 2, _o.height * 1e3 / 2
        _r = 40.0                       # an 80 mm frame
        for _x, _y, _ in _FANS.fan_centres(_o, _o.max_fans):
            if not (_cx - _hw <= _x - _r + 1e-6 and _x + _r <= _cx + _hw + 1e-6
                    and _cy - _hh <= _y - _r + 1e-6
                    and _y + _r <= _cy + _hh + 1e-6):
                failures.append(f"fan at ({_x:.0f},{_y:.0f}) is outside the "
                                f"{_o.name} cutout")
    print("fans: all frames fit inside their measured cutouts")

    # 7. the CalSol logo watermark: present, correctly proportioned, and not
    #    sitting on top of the scalar bar (which spans the bottom of the view).
    import packscene as _PS

    if not _PS.LOGO_PATH.exists():
        failures.append(f"logo asset missing: {_PS.LOGO_PATH}")
    else:
        _w, _h = 1500, 1000
        _hf = _PS.LOGO_WIDTH_FRAC * (_w / _h) / _PS.LOGO_ASPECT
        _drawn = (_PS.LOGO_WIDTH_FRAC * _w) / (_hf * _h)
        if abs(_drawn - _PS.LOGO_ASPECT) > 0.02:
            failures.append(f"logo aspect distorted: drawn {_drawn:.2f} "
                            f"vs artwork {_PS.LOGO_ASPECT:.2f}")
        if _PS.LOGO_POSITION[1] < 0.25:
            failures.append("logo sits low enough to collide with the "
                            "horizontal scalar bar")
        print(f"logo: {_PS.LOGO_WIDTH_FRAC:.2f}w at {_PS.LOGO_POSITION}, "
              f"aspect {_drawn:.2f} (artwork {_PS.LOGO_ASPECT:.2f})")

    # 8. websocket budget. Re-encoding the ~200 KB sidebar on every animation
    #    frame pushed ~3.3 MB/s and closed the transport ("Cannot write to
    #    closing transport" x1392). The sheet must be throttled, not per-frame.
    _enc = {"n": 0, "bytes": 0}
    _orig_refresh = A.refresh_sidebar

    def _counting(rebuild=False, play=False):
        _orig_refresh(rebuild=rebuild, play=play)
        _enc["n"] += 1
        _enc["bytes"] += len(A.state.sidebar_src)

    A.state.mode = "transient"
    A.recompute()
    A.refresh_sidebar = _counting
    try:
        async def _play(sec):
            A.state.playing = True
            asyncio.ensure_future(A.animate())
            await asyncio.sleep(sec)
            A.state.playing = False
            await asyncio.sleep(0.3)

        _t0 = time.monotonic()
        asyncio.new_event_loop().run_until_complete(_play(1.5))
        _wall = time.monotonic() - _t0
    finally:
        A.refresh_sidebar = _orig_refresh

    _rate = _enc["bytes"] / 1e6 / max(_wall, 1e-6)
    if _rate > 1.0:
        failures.append(f"sidebar pushes {_rate:.2f} MB/s during playback "
                        f"-- the websocket will back up")
    print(f"websocket: sidebar {_enc['n'] / _wall:.1f} encodes/s, "
          f"{_rate:.2f} MB/s during playback")

    print(f"rendered HTML: {len(html)} chars, {n_err} html-error markers")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("   -", f)
        return 1
    print(f"all {len(REQUIRED_BINDINGS)} bindings and "
          f"{len(REQUIRED_MARKUP)} markup checks pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

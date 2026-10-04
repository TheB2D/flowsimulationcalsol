r"""
Camera control for the 3D view: keyboard orbit, pan mode, and named views.

The default VTK interactor gives you mouse orbit and not much else. This adds
the two things that actually make a box like this easy to inspect:

  * **Arrow keys drive the camera.** Left/right yaw (azimuth), up/down pitch
    (elevation). In PAN mode the same keys translate instead. Shift makes any
    step 4x coarser; Q/E roll; +/- dolly.
  * **A pan mode.** Orbiting about a fixed focal point is the wrong gesture when
    you want to look along the 590 mm flow axis at one module. Pan moves the
    camera and its focal point together so the view slides sideways.

Everything here mutates the camera in place and is cheap -- no mesh touches, no
re-solve -- so it sits in the app's render-only update tier (~4 ms).

The rotation calls go through the underlying `vtkCamera` methods (`Azimuth`,
`Elevation`, `Roll`, `Dolly`), which apply a RELATIVE delta. PyVista also
exposes `azimuth`/`elevation` as properties holding an absolute angle; mixing
the two is an easy way to get a camera that drifts, so this module uses the
method form exclusively and always re-orthogonalises the view-up afterwards to
stop the horizon tumbling after repeated pitch steps.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Degrees per arrow-key press, and the multiplier when Shift is held.
STEP_DEG = 6.0
STEP_PAN = 0.045        # fraction of the scene diagonal per press
STEP_ZOOM = 1.12
FAST = 4.0

MODES = ("orbit", "pan")


def _basis(cam) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, forward) unit vectors for the current camera."""
    pos = np.asarray(cam.position, dtype=float)
    foc = np.asarray(cam.focal_point, dtype=float)
    up = np.asarray(cam.up, dtype=float)

    fwd = foc - pos
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n else np.array([0.0, 0.0, -1.0])

    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    right = right / n if n else np.array([1.0, 0.0, 0.0])

    true_up = np.cross(right, fwd)
    return right, true_up, fwd


def orbit(plotter, d_az: float = 0.0, d_el: float = 0.0) -> None:
    """Rotate about the focal point.

    `OrthogonalizeViewUp` after each step is what keeps the horizon level --
    without it, repeated elevation steps slowly roll the scene.
    """
    cam = plotter.camera
    if d_az:
        cam.Azimuth(d_az)
    if d_el:
        # Stop short of the poles; going through one flips the view-up and the
        # controls invert, which feels broken even though it is "correct".
        cam.Elevation(max(-85.0, min(85.0, d_el)))
    cam.OrthogonalizeViewUp()
    plotter.render()


def pan(plotter, dx: float = 0.0, dy: float = 0.0, scale: float = 1.0) -> None:
    """Slide the camera and its focal point together.

    `dx`/`dy` are in units of the scene diagonal, so a press moves the same
    apparent distance whatever the model's size.
    """
    cam = plotter.camera
    right, up, _ = _basis(cam)

    span = _scene_span(plotter) * scale
    shift = right * (dx * span) + up * (dy * span)

    cam.position = tuple(np.asarray(cam.position, dtype=float) + shift)
    cam.focal_point = tuple(np.asarray(cam.focal_point, dtype=float) + shift)
    plotter.render()


def roll(plotter, deg: float) -> None:
    plotter.camera.Roll(deg)
    plotter.camera.OrthogonalizeViewUp()
    plotter.render()


def dolly(plotter, factor: float) -> None:
    """Move along the view axis. Preferred over `zoom`, which only changes the
    view angle and eventually distorts the perspective."""
    plotter.camera.Dolly(factor)
    plotter.renderer.ResetCameraClippingRange()
    plotter.render()


def _scene_span(plotter) -> float:
    try:
        b = plotter.renderer.ComputeVisiblePropBounds()
        d = math.dist((b[0], b[2], b[4]), (b[1], b[3], b[5]))
        return d if d > 0 else 1.0
    except Exception:
        return 1.0


# =============================================================================
# Named views
# =============================================================================

def _look(plotter, eye_dir, up=(0.0, 1.0, 0.0), zoom: float = 1.25) -> None:
    """Point the camera at the scene centre from a given direction."""
    b = plotter.renderer.ComputeVisiblePropBounds()
    centre = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
    span = _scene_span(plotter)
    d = np.asarray(eye_dir, dtype=float)
    d = d / (np.linalg.norm(d) or 1.0)

    cam = plotter.camera
    cam.focal_point = tuple(centre)
    cam.position = tuple(centre + d * span)
    cam.up = up
    plotter.reset_camera()
    cam.Zoom(zoom)
    plotter.render()


def view_iso(plotter) -> None:
    """Three-quarter view down the flow axis -- the default.

    PyVista's own "iso" lands end-on for this box, which hides the row-to-row
    temperature gradient the colour map exists to show.
    """
    _look(plotter, (1.0, 0.75, 0.55))


def view_flow(plotter) -> None:
    """Side-on, looking across the 590 mm flow axis: the gradient view."""
    _look(plotter, (0.15, 0.25, 1.0))


def view_inlet(plotter) -> None:
    """Head-on at the blue Front Plate (the intake cutout)."""
    _look(plotter, (0.0, 0.0, 1.0))


def view_exhaust(plotter) -> None:
    """Head-on at the green Rear Plate (the exhaust cutout)."""
    _look(plotter, (0.0, 0.0, -1.0))


def view_top(plotter) -> None:
    _look(plotter, (0.0, 1.0, 0.05), up=(0.0, 0.0, -1.0))


def view_side(plotter) -> None:
    _look(plotter, (1.0, 0.0, 0.0))


NAMED_VIEWS = {
    "iso": view_iso, "flow": view_flow, "inlet": view_inlet,
    "exhaust": view_exhaust, "top": view_top, "side": view_side,
}


# =============================================================================
# The key map
# =============================================================================

def handle_key(plotter, key: str, mode: str = "orbit",
               shift: bool = False) -> bool:
    """Apply one keypress. Returns True if the camera moved.

    Arrow keys orbit in ORBIT mode and translate in PAN mode. Everything else
    behaves the same in both, so switching modes only ever changes what the
    arrows do -- which is the whole point of having a mode.
    """
    k = (key or "").lower()
    f = FAST if shift else 1.0
    deg = STEP_DEG * f
    pad = STEP_PAN * f

    if k in ("arrowleft", "left", "a"):
        pan(plotter, -pad, 0.0) if mode == "pan" else orbit(plotter, d_az=+deg)
    elif k in ("arrowright", "right", "d"):
        pan(plotter, +pad, 0.0) if mode == "pan" else orbit(plotter, d_az=-deg)
    elif k in ("arrowup", "up", "w"):
        pan(plotter, 0.0, +pad) if mode == "pan" else orbit(plotter, d_el=+deg)
    elif k in ("arrowdown", "down", "s"):
        pan(plotter, 0.0, -pad) if mode == "pan" else orbit(plotter, d_el=-deg)
    elif k in ("q",):
        roll(plotter, -deg)
    elif k in ("e",):
        roll(plotter, +deg)
    elif k in ("+", "=", "add"):
        dolly(plotter, STEP_ZOOM)
    elif k in ("-", "_", "subtract"):
        dolly(plotter, 1.0 / STEP_ZOOM)
    elif k in ("r",):
        view_iso(plotter)
    elif k in ("f",):
        view_flow(plotter)
    else:
        return False
    return True


KEY_HELP = [
    ("arrow keys", "orbit  (pan in PAN mode)"),
    ("shift + arrows", "4x coarser steps"),
    ("Q / E", "roll"),
    ("+ / -", "zoom in / out"),
    ("R", "reset to the default view"),
    ("F", "side-on flow view"),
    ("drag", "orbit"),
    ("shift + drag", "pan"),
    ("scroll", "zoom"),
]

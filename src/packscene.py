r"""
The 3D scene: build the VTK pipeline once, then only write into arrays.

This module exists to own one performance property.  `render.py` builds its
scene from scratch on every call -- it copies each of the 408 cell meshes, fills
a constant scalar array, merges them, and adds the result -- which is fine for a
one-shot PNG and far too slow for an animation.  Here the merge happens once at
construction, and every later update is a single vectorised write into an
existing VTK buffer, measured at well under a millisecond.

If that logic lived next to the UI callbacks it would drift back into re-adding
meshes the first time something needed changing, so it is isolated behind a
handle object whose only mutators are the fast ones.

Two bugs in the existing render path are fixed here because an app that
compares modes cannot live with either:

  * **No `clim`.**  `render.py` lets VTK auto-scale the colour map to each
    frame's range, so a cell sitting at 45 C changes colour as its neighbours
    warm, and two modes are not visually comparable.  The range is pinned.

  * **Alpha-blending order.**  `DRAW_ORDER` works only while opacity is fixed
    at build time: the correct order depends on the opacity values, and the
    actors are already added by the time a user drags a slider.  Depth peeling
    makes blending order-independent, so opacity becomes a free runtime
    property.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import geometry as G
from geometry import Body
from render import build_blocks, cell_meshes, _ordered_cells, _n_faces


# Per-class appearance.  Panels, shelf, PCBs and internal structure start
# translucent so the cells read clearly; the cells are the only solid class.
CLASS_STYLE = {
    "cell":       {"color": "#3B82F6", "opacity": 1.00},
    "case_plate": {"color": "#94A3B8", "opacity": 0.10},
    "busbar":     {"color": "#F59E0B", "opacity": 0.45},
    "structure":  {"color": "#64748B", "opacity": 0.10},
    "fastener":   {"color": "#475569", "opacity": 0.10},
    "pcb":        {"color": "#10B981", "opacity": 0.12},
    "shelf":      {"color": "#A1A1AA", "opacity": 0.10},
    "other":      {"color": "#A78BFA", "opacity": 0.12},
}

# Display colours for the named panels.  These are the sRGB values stored in the
# STEP -- the ones the user recognises from the CAD.  They are NOT what OCCT
# returns (it gives linear RGB, so the blue panel reads back as #000BA4), which
# is exactly why panels are keyed by NAME here and never matched on colour.
PANEL_COLOR = {
    "front plate": "#023DD2",   # the blue panel with the cutout
    "rear plate":  "#49A954",   # the green panel with the cutout
    "side plate":  "#B11919",
    "base plate":  "#E87108",
    "top brace":   "#FFEF23",
    "rear panel":  "#A747FF",
    "front duct":  "#CBD5E1",
}

# Opaque first, most transparent last.  Kept as the add order so the scene is
# still correct if depth peeling is unavailable on the user's driver.
DRAW_ORDER = ["cell", "busbar", "pcb", "other", "structure",
              "fastener", "shelf", "case_plate"]

# Decimation for the full 3374-solid model (~5.5 M triangles). Cells are never
# decimated -- they carry the result, and they are only 0.34 M triangles.
DECIMATE = {"structure": 0.06, "busbar": 0.12, "fastener": 0.04,
            "other": 0.10, "pcb": 0.20, "shelf": 0.15}

# Default colour range. Deliberately NOT the full 35-65 C span: a realistic
# result occupies ~40-48 C, and stretching the map to the 60 C limit renders
# every cell near-black and hides the gradient the map exists to show. The
# limit is still marked on the figures; this is about making the 3D readable.
CLIM_DEFAULT = (40.0, 50.0)     # C

# CalSol team logo, shown as a watermark in the 3D view.
LOGO_PATH = pathlib.Path(__file__).resolve().parent.parent / "assets" / "calsol_logo.png"
LOGO_ASPECT = 774.0 / 240.0     # the artwork's own width/height
LOGO_WIDTH_FRAC = 0.13          # fraction of the window width
# Top-right. The bottom-right corner looks tempting but the horizontal scalar
# bar runs the full width down there, and the logo lands straight on top of its
# upper end.  Top-right is the only free corner: the scene title is top-left and
# the orientation axes are bottom-left.
LOGO_POSITION = (0.845, 0.915)
LOGO_OPACITY = 0.85


@dataclass
class SceneHandles:
    """Everything the app needs to mutate the scene without rebuilding it."""

    plotter: pv.Plotter
    class_actors: dict = field(default_factory=dict)
    panel_actors: dict = field(default_factory=dict)
    cell_mesh: pv.PolyData | None = None
    cell_actor: object | None = None
    cell_index: np.ndarray | None = None     # point -> cell ordinal
    cell_order: list = field(default_factory=list)
    fan_actors: list = field(default_factory=list)
    clim: tuple = CLIM_DEFAULT
    depth_peeling: bool = False
    n_cells: int = 0


def enable_order_independent_transparency(p: pv.Plotter) -> bool:
    """Turn on depth peeling so opacity can change at runtime.

    MSAA and depth peeling are mutually exclusive in VTK, so multisampling is
    switched off and FXAA used instead -- it composites after blending and is
    compatible.

    Returns whether the request was accepted.  Note that
    `GetLastRenderingUsedDepthPeeling()` only reports true after a real render
    pass has run, which on an off-screen plotter has not happened yet at build
    time -- so it is NOT a usable readiness check here.  `verify_depth_peeling`
    below does that after the first real frame.
    """
    try:
        ren = p.renderer
        ren.SetUseDepthPeeling(True)
        ren.SetMaximumNumberOfPeels(4)
        ren.SetOcclusionRatio(0.1)
        if getattr(p, "ren_win", None) is not None:
            p.ren_win.SetAlphaBitPlanes(1)
            p.ren_win.SetMultiSamples(0)
        p.enable_anti_aliasing("fxaa")
        return bool(ren.GetUseDepthPeeling())
    except Exception:
        return False


def verify_depth_peeling(p: pv.Plotter) -> bool:
    """Whether the driver actually used depth peeling on the last frame.

    Call after a real render.  If this comes back false the scene is still
    correct -- `DRAW_ORDER` remains the add order -- but transparency may show
    ordering artefacts once opacity is changed at runtime, which is worth
    surfacing in the UI rather than leaving the user to wonder.
    """
    try:
        return bool(p.renderer.GetLastRenderingUsedDepthPeeling())
    except Exception:
        return False


def build_cell_block(data: dict) -> tuple[pv.PolyData, np.ndarray, list]:
    r"""Merge the 408 cells ONCE, tagging every point with its cell ordinal.

    After this, scattering a per-cell temperature vector `T` to points is the
    single expression ``T[cell_index]`` -- no copy, no merge, no re-add. That is
    what makes the animation hot path ~0.7 ms instead of ~0.5 s.
    """
    meshes, bodies = cell_meshes(data)
    ordered = _ordered_cells(data)
    pos = {b.index: k for k, b in enumerate(ordered)}

    parts = []
    for m, b in zip(meshes, bodies):
        if b.index not in pos:
            continue
        mm = m.copy()
        mm["cid"] = np.full(mm.n_points, pos[b.index], dtype=np.int32)
        parts.append(mm)

    merged = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
    cid = np.asarray(merged["cid"], dtype=np.int32)
    merged["T"] = np.zeros(merged.n_points, dtype=np.float64)
    return merged, cid, [b.index for b in ordered]


def build_scene(data: dict,
                off_screen: bool = True,
                decimate: dict | None = None,
                clim: tuple = CLIM_DEFAULT,
                window_size: tuple = (1500, 1000)) -> SceneHandles:
    """Assemble the scene once and return the handles to mutate it."""
    p = pv.Plotter(off_screen=off_screen, window_size=window_size)
    p.set_background("#0B1220")

    h = SceneHandles(plotter=p, clim=clim)
    h.depth_peeling = enable_order_independent_transparency(p)

    # -- cells: one actor, one scalar array -------------------------------
    cells = [b for b in data["bodies"] if b.cls == "cell"]
    if cells:
        mesh, cid, order = build_cell_block(data)
        h.cell_mesh, h.cell_index, h.cell_order = mesh, cid, order
        h.n_cells = len(order)
        h.cell_actor = p.add_mesh(
            mesh, scalars="T", cmap="inferno", clim=clim,
            smooth_shading=True, opacity=CLASS_STYLE["cell"]["opacity"],
            scalar_bar_args={"title": "Cell temperature  [C]", "color": "white",
                             "n_labels": 6, "title_font_size": 15,
                             "label_font_size": 12})

    # -- everything else: one merged actor per class ----------------------
    blocks = build_blocks(data, decimate=decimate)
    for cls in [c for c in DRAW_ORDER if c in blocks] + \
               [c for c in blocks if c not in DRAW_ORDER]:
        if cls == "cell":
            continue
        st = CLASS_STYLE.get(cls, {"color": "#9CA3AF", "opacity": 0.12})
        h.class_actors[cls] = p.add_mesh(
            blocks[cls], color=st["color"], opacity=st["opacity"],
            smooth_shading=False)

    p.add_axes(line_width=2, color="white")
    add_logo(p, window_size)
    set_default_camera(p, data)
    return h


def add_logo(p: pv.Plotter, window_size: tuple = (1500, 1000)) -> bool:
    """Place the CalSol logo as a watermark in the 3D view.

    The widget takes INDEPENDENT width and height fractions of the window, so
    the height has to be derived from both the artwork's aspect and the
    window's -- passing a square size would stretch a 3.2:1 logo badly.

    Returns False and leaves the scene untouched if the asset is missing, since
    a missing logo should never stop the app from rendering.
    """
    if not LOGO_PATH.exists():
        return False
    try:
        w, h = window_size
        logo = pv.read(str(LOGO_PATH))
        h_frac = LOGO_WIDTH_FRAC * (w / h) / LOGO_ASPECT
        p.add_logo_widget(logo, position=LOGO_POSITION,
                          size=(LOGO_WIDTH_FRAC, h_frac),
                          opacity=LOGO_OPACITY)
        return True
    except Exception:
        return False


def set_default_camera(p: pv.Plotter, data: dict | None = None) -> None:
    """A three-quarter view down the flow axis.

    The box is long in z (771 mm) and the air crosses it along z, so the
    row-to-row temperature gradient only reads if the camera is off to the side
    rather than looking straight down an end. PyVista's "iso" lands end-on here,
    which hides the very thing the colour map is showing.
    """
    p.camera_position = [(900.0, 900.0, 250.0),     # eye
                         (0.0, 120.0, -300.0),      # focus: centre of the pack
                         (0.0, 1.0, 0.0)]           # +y up
    p.reset_camera()
    p.camera.zoom(1.25)


# =============================================================================
# The fast paths -- these are what run during an animation
# =============================================================================

def set_cell_temperatures(h: SceneHandles, T: np.ndarray) -> None:
    """Write a per-cell temperature vector [C]. Sub-millisecond.

    Mutates the existing VTK-backed buffer in place and flags it modified.
    Never rebind ``mesh["T"] = ...`` -- that allocates a fresh array and can
    drop the mapper's binding.
    """
    if h.cell_mesh is None or h.cell_index is None:
        return
    T = np.asarray(T, dtype=np.float64)
    if T.size != h.n_cells:
        raise ValueError(f"got {T.size} temperatures for {h.n_cells} cells")
    arr = h.cell_mesh.point_data["T"]
    arr[:] = T[h.cell_index]
    h.cell_mesh.point_data["T"].Modified()


def set_clim(h: SceneHandles, clim: tuple) -> None:
    """Pin or re-pin the colour range so modes stay comparable."""
    h.clim = tuple(clim)
    if h.cell_actor is not None:
        h.cell_actor.mapper.scalar_range = h.clim


def set_class_opacity(h: SceneHandles, cls: str, opacity: float) -> None:
    if cls == "cell" and h.cell_actor is not None:
        h.cell_actor.GetProperty().SetOpacity(float(opacity))
        return
    a = h.class_actors.get(cls)
    if a is not None:
        a.GetProperty().SetOpacity(float(opacity))


def set_class_visible(h: SceneHandles, cls: str, visible: bool) -> None:
    if cls == "cell" and h.cell_actor is not None:
        h.cell_actor.SetVisibility(bool(visible))
        return
    a = h.class_actors.get(cls)
    if a is not None:
        a.SetVisibility(bool(visible))


def set_opaque_mode(h: SceneHandles, opaque: bool,
                    groups: tuple = ("case_plate", "shelf", "pcb", "structure")
                    ) -> None:
    """The master transparency toggle.

    Default (False) leaves the panels, shelf, PCBs and internal structure
    translucent so the cells are visible; True makes them solid. Correct at any
    opacity because depth peeling removed the ordering constraint.
    """
    for cls in groups:
        set_class_opacity(h, cls, 1.0 if opaque else
                          CLASS_STYLE.get(cls, {}).get("opacity", 0.12))


def clear_fans(h: SceneHandles) -> None:
    for a in h.fan_actors:
        try:
            h.plotter.remove_actor(a)
        except Exception:
            pass
    h.fan_actors = []


def scene_stats(data: dict) -> dict:
    import collections
    counts = collections.Counter(b.cls for b in data["bodies"])
    tris = sum(len(f) // 4 for _, f in data["meshes"])
    return {"solids": len(data["bodies"]), "triangles": tris,
            "classes": dict(counts)}
